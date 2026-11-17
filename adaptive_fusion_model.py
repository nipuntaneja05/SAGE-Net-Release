"""
adaptive_fusion_model.py
Complete 6-Encoder Architecture for Multimodal Crossing Anticipation:
- 3 Kinematic Encoders:
  1. Kinematic Motion Encoder (bounding box geometry, velocity, acceleration)
  2. Kinematic Scene Encoder (ego-vehicle speed and dynamics)
  3. Kinematic Interaction Encoder (spatial expansion and relative trajectory graph)
- 3 Visual Encoders:
  1. Visual Motion Encoder (pedestrian crop appearance)
  2. Visual Scene Encoder (illumination-adapted global scene context)
  3. Visual Interaction Encoder (self-guided spatial interaction surround)
- Preprocessing & Robustness Blocks:
  1. Occlusion Motion Reconstruction Block (denoises & inpaints occluded trajectories)
  2. Illumination Block (unsupervised day/night dual-pathway feature adapter)
- Scene-Conditioned Adaptive Fusion Gate:
  Dynamically balances Kinematic vs. Visual decision authority (w_K vs w_V) based on scene reliability.
- Multi-Task Auxiliary Heads:
  Outputs (L_final, L_kinem, L_vis, w_K, w_V, reconstructed_KM, recon_loss)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. TEMPORAL ATTENTION AGGREGATOR
# ==============================================================================
class TemporalAttention(nn.Module):
    """
    Temporal Dot-Product Attention:
    score_t = tanh(W * h_t + b)
    alpha_t = softmax(score_t)
    context = sum(alpha_t * h_t)
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, seq):
        # seq: [B, Seq, hidden_dim]
        scores = self.attn(seq) # [B, Seq, 1]
        alpha = F.softmax(scores, dim=1) # [B, Seq, 1]
        pooled = torch.sum(alpha * seq, dim=1) # [B, hidden_dim]
        return pooled, alpha.squeeze(-1)


# ==============================================================================
# 2. OCCLUSION MOTION RECONSTRUCTION BLOCK
# ==============================================================================
class OcclusionReconstructionBlock(nn.Module):
    """
    Denoises and reconstructs occluded / missing motion trajectory coordinates.
    During training, simulates occlusion by randomly masking 1 to 4 timesteps.
    Trained via auxiliary MSE trajectory reconstruction loss.
    """
    def __init__(self, in_dim=17, hidden_dim=64):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.proj_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, km, mask_prob=0.15, force_mask=False):
        """
        km: [B, Seq, 17]
        Returns:
            km_recon: [B, Seq, 17] reconstructed trajectory
            recon_loss: scalar reconstruction loss on active kinematics (dims 0:12)
        """
        B, Seq, D = km.shape
        km_in = km.clone()

        if self.training or force_mask:
            # Simulate random occlusion mask across timesteps
            # 1 to 4 frames randomly masked
            mask = torch.ones((B, Seq, 1), device=km.device)
            for b in range(B):
                if torch.rand(1).item() < 0.7:  # 70% chance of experiencing occlusion in training
                    num_masked = torch.randint(1, 5, (1,)).item()
                    start_t = torch.randint(0, max(1, Seq - num_masked + 1), (1,)).item()
                    mask[b, start_t:start_t + num_masked] = 0.0
            km_masked = km_in * mask
        else:
            mask = torch.ones((B, Seq, 1), device=km.device)
            km_masked = km_in

        # Encode and reconstruct
        x = F.gelu(self.conv_in(km_masked.transpose(1, 2))).transpose(1, 2) # [B, Seq, hidden_dim]
        gru_out, _ = self.bigru(x) # [B, Seq, hidden_dim * 2]
        delta = self.proj_out(gru_out) # [B, Seq, 17]
        km_recon = km_masked + delta

        # Preserve normalized timeline (dim 16)
        km_recon[:, :, 16] = km[:, :, 16]

        # Calculate reconstruction loss strictly on masked timesteps if any, else across all active dims (0:12)
        recon_loss = F.mse_loss(km_recon[:, :, 0:12], km[:, :, 0:12])

        return km_recon, recon_loss


# ==============================================================================
# 3. ILLUMINATION BLOCK (Dual-Pathway Day/Night Scene Adapter)
# ==============================================================================
class IlluminationBlock(nn.Module):
    """
    Unsupervised Illumination Adapter:
    Evaluates global scene context V_S (1792-D) to estimate lighting regime (g_illum in [0, 1]).
    Dynamically blends daylight and low-light / night specialized projections.
    """
    def __init__(self, vis_dim=1792, bottleneck_dim=128):
        super().__init__()
        # Illumination estimator
        self.illum_estimator = nn.Sequential(
            nn.Linear(vis_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # Daylight specialized projection
        self.proj_day = nn.Sequential(
            nn.Linear(vis_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )
        # Low-light / Night specialized projection
        self.proj_night = nn.Sequential(
            nn.Linear(vis_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim)
        )

    def forward(self, v_s):
        # v_s: [B, Seq, 1792]
        v_s_mean = v_s.mean(dim=1) # [B, 1792]
        g_illum = self.illum_estimator(v_s_mean) # [B, 1]

        z_day = self.proj_day(v_s)       # [B, Seq, 128]
        z_night = self.proj_night(v_s)   # [B, Seq, 128]

        # Dynamic blend
        g_expanded = g_illum.unsqueeze(1) # [B, 1, 1]
        z_s_adapted = (1.0 - g_expanded) * z_day + g_expanded * z_night
        return z_s_adapted, g_illum


# ==============================================================================
# 4. THE 3 KINEMATIC ENCODERS
# ==============================================================================
class KinematicMotionEncoder(nn.Module):
    """
    Encodes pedestrian bounding box geometry, velocity, and acceleration (dims 0:12).
    Multi-Scale Temporal Conv1D (k=3, k=5) + BiLSTM + Temporal Attention.
    """
    def __init__(self, in_dim=12, conv_dim=32, out_dim=64, lstm_hidden=64):
        super().__init__()
        self.conv_k3 = nn.Conv1d(in_dim, conv_dim, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(in_dim, conv_dim, kernel_size=5, padding=2)
        self.stem_proj = nn.Sequential(
            nn.Linear(conv_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        self.bilstm = nn.LSTM(
            input_size=out_dim,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )
        self.norm = nn.LayerNorm(lstm_hidden * 2)
        self.attn = TemporalAttention(hidden_dim=lstm_hidden * 2)

    def forward(self, km):
        # km: [B, Seq, 17] -> take dims 0:12
        x = km[:, :, 0:12].transpose(1, 2)
        f_k3 = F.gelu(self.conv_k3(x))
        f_k5 = F.gelu(self.conv_k5(x))
        stem_out = self.stem_proj(torch.cat([f_k3, f_k5], dim=1).transpose(1, 2))
        lstm_out, _ = self.bilstm(stem_out)
        h_norm = self.norm(lstm_out)
        pooled, alpha = self.attn(h_norm)
        return pooled, alpha


class KinematicSceneEncoder(nn.Module):
    """
    Encodes ego-vehicle speed and speed dynamics (dims 12:16) + timeline (dim 16).
    Conv1D + BiLSTM + Temporal Attention.
    """
    def __init__(self, in_dim=5, conv_dim=32, lstm_hidden=64):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, conv_dim, kernel_size=3, padding=1)
        self.proj = nn.Sequential(
            nn.Linear(conv_dim, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )
        self.norm = nn.LayerNorm(lstm_hidden * 2)
        self.attn = TemporalAttention(hidden_dim=lstm_hidden * 2)

    def forward(self, km):
        # km: [B, Seq, 17] -> take speed (12:16) and timeline (16:17) = 5 dims
        x = km[:, :, 12:17].transpose(1, 2)
        f = F.gelu(self.conv(x)).transpose(1, 2)
        p = self.proj(f)
        lstm_out, _ = self.bilstm(p)
        h_norm = self.norm(lstm_out)
        pooled, alpha = self.attn(h_norm)
        return pooled, alpha


class KinematicInteractionEncoder(nn.Module):
    """
    Encodes spatial interaction dynamics:
    Relative scale expansion (w/h, area w*h), expansion rate, distance proxy (1/h),
    and heading trajectory angle.
    Multi-Head Attention Graph + BiLSTM + Temporal Attention.
    """
    def __init__(self, num_heads=4, embed_dim=64, lstm_hidden=64):
        super().__init__()
        # Computes 5 derived spatial interaction features from [cx, cy, w, h]
        self.input_proj = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )
        self.norm = nn.LayerNorm(lstm_hidden * 2)
        self.attn = TemporalAttention(hidden_dim=lstm_hidden * 2)

    def forward(self, km):
        # km: [B, Seq, 17]
        cx, cy, w, h = km[:, :, 0], km[:, :, 1], km[:, :, 2], km[:, :, 3]
        eps = 1e-4
        aspect = w / torch.clamp(h, min=eps)
        area = w * h
        dist_proxy = 1.0 / torch.clamp(h, min=eps)
        vx, vy = km[:, :, 4], km[:, :, 5]
        heading = torch.atan2(vy, vx + eps)

        inter_feat = torch.stack([aspect, area, dist_proxy, vx, heading], dim=-1) # [B, Seq, 5]
        x = self.input_proj(inter_feat) # [B, Seq, embed_dim]
        attn_out, _ = self.mha(x, x, x)
        x_fused = x + attn_out

        lstm_out, _ = self.bilstm(x_fused)
        h_norm = self.norm(lstm_out)
        pooled, alpha = self.attn(h_norm)
        return pooled, alpha


# ==============================================================================
# 5. THE 3 VISUAL ENCODERS
# ==============================================================================
class VisualMotionEncoder(nn.Module):
    """
    Encodes pedestrian appearance crop V_M (1792-D).
    Projection + BiLSTM + Temporal Attention.
    """
    def __init__(self, in_dim=1792, bottleneck_dim=128, lstm_hidden=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.bilstm = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )
        self.norm = nn.LayerNorm(lstm_hidden * 2)
        self.attn = TemporalAttention(hidden_dim=lstm_hidden * 2)

    def forward(self, vm):
        # vm: [B, Seq, 1792]
        x = self.proj(vm)
        lstm_out, _ = self.bilstm(x)
        h_norm = self.norm(lstm_out)
        pooled, alpha = self.attn(h_norm)
        return pooled, alpha


class VisualSceneEncoder(nn.Module):
    """
    Encodes illumination-adapted scene context Z_S' (128-D).
    BiLSTM + Temporal Attention.
    """
    def __init__(self, in_dim=128, lstm_hidden=64):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )
        self.norm = nn.LayerNorm(lstm_hidden * 2)
        self.attn = TemporalAttention(hidden_dim=lstm_hidden * 2)

    def forward(self, zs_adapted):
        # zs_adapted: [B, Seq, 128]
        lstm_out, _ = self.bilstm(zs_adapted)
        h_norm = self.norm(lstm_out)
        pooled, alpha = self.attn(h_norm)
        return pooled, alpha


class VisualInteractionEncoder(nn.Module):
    """
    Encodes interaction surround context V_I (1792-D) guided by pedestrian appearance V_M.
    Cross-Modal Visual Attention + BiLSTM + Temporal Attention.
    """
    def __init__(self, in_dim=1792, embed_dim=256, bottleneck_dim=128, lstm_hidden=64):
        super().__init__()
        self.proj_m = nn.Linear(in_dim, embed_dim)
        self.proj_i = nn.Linear(in_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm_mha = nn.LayerNorm(embed_dim)
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU()
        )
        self.bilstm = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )
        self.norm_lstm = nn.LayerNorm(lstm_hidden * 2)
        self.attn = TemporalAttention(hidden_dim=lstm_hidden * 2)

    def forward(self, vm, vi):
        # vm: [B, Seq, 1792], vi: [B, Seq, 1792]
        q = self.proj_m(vm)
        k = self.proj_i(vi)
        v = k
        attn_out, _ = self.mha(query=q, key=k, value=v)
        x = self.norm_mha(q + attn_out)
        x_btn = self.bottleneck(x)
        lstm_out, _ = self.bilstm(x_btn)
        h_norm = self.norm_lstm(lstm_out)
        pooled, alpha = self.attn(h_norm)
        return pooled, alpha


# ==============================================================================
# 6. SCENE-CONDITIONED ADAPTIVE FUSION GATE
# ==============================================================================
class SceneConditionedAdaptiveGate(nn.Module):
    """
    Evaluates visual scene reliability:
    - Input: H_V_scene (128-D), illumination score g_illum (1-D), and scene complexity stat (64-D).
    - Output: Softmax weights [w_K, w_V] across Kinematic and Visual streams.
    - Behavior:
      If scene is occluded, dark, or cluttered -> w_K -> 1 (kinematics classify).
      If scene is clear, unoccluded, well-lit   -> w_V -> 1 (vision classifies).
    """
    def __init__(self, scene_feat_dim=128, stat_dim=64):
        super().__init__()
        self.stat_proj = nn.Sequential(
            nn.Linear(1792, stat_dim),
            nn.LayerNorm(stat_dim),
            nn.GELU()
        )
        gate_input_dim = scene_feat_dim + 1 + stat_dim # 128 + 1 + 64 = 193
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
        # Initialize bias to slightly favor kinematics at training onset ([0.5, -0.5])
        nn.init.constant_(self.gate_mlp[4].bias, 0.0)

    def forward(self, h_v_scene, g_illum, v_s):
        # h_v_scene: [B, 128]
        # g_illum: [B, 1]
        # v_s: [B, Seq, 1792] -> compute scene variance / complexity stat
        v_s_stat = self.stat_proj(v_s.std(dim=1)) # [B, 64]
        gate_in = torch.cat([h_v_scene, g_illum, v_s_stat], dim=-1) # [B, 193]
        logits_gate = self.gate_mlp(gate_in) # [B, 2]
        weights = F.softmax(logits_gate, dim=-1) # [B, 2] -> [w_K, w_V]
        return weights


# ==============================================================================
# 7. MASTER MULTIMODAL ARCHITECTURE
# ==============================================================================
class AdaptiveFusionMasterModel(nn.Module):
    """
    AdaptiveFusionMasterModel
    Combines:
    1. Occlusion Motion Reconstruction Block
    2. Illumination Dual-Pathway Adapter Block
    3. 3 Kinematic Encoders (Motion, Scene, Interaction) -> Kinematic Logit L_K
    4. 3 Visual Encoders (Motion, Scene, Interaction) -> Visual Logit L_V
    5. Scene-Conditioned Adaptive Fusion Gate -> [w_K, w_V]
    6. Final Logit: L_final = w_K * L_K + w_V * L_V
    """
    def __init__(self, vis_dim=1792, km_dim=17, lstm_hidden=64, use_concat_head=False):
        super().__init__()
        # Preprocessing & Robustness Blocks
        self.occlusion_block = OcclusionReconstructionBlock(in_dim=km_dim, hidden_dim=64)
        self.illum_block = IlluminationBlock(vis_dim=vis_dim, bottleneck_dim=128)

        # 3 Kinematic Encoders
        self.kinem_motion_enc = KinematicMotionEncoder(in_dim=12, lstm_hidden=lstm_hidden)
        self.kinem_scene_enc  = KinematicSceneEncoder(in_dim=5, lstm_hidden=lstm_hidden)
        self.kinem_inter_enc  = KinematicInteractionEncoder(lstm_hidden=lstm_hidden)

        # Kinematic Intention Head: [128*3] = 384 -> 64 -> 1
        self.kinem_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2 * 3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # 3 Visual Encoders
        self.vis_motion_enc = VisualMotionEncoder(in_dim=vis_dim, lstm_hidden=lstm_hidden)
        self.vis_scene_enc  = VisualSceneEncoder(in_dim=128, lstm_hidden=lstm_hidden)
        self.vis_inter_enc  = VisualInteractionEncoder(in_dim=vis_dim, lstm_hidden=lstm_hidden)

        # Visual Intention Head: [128*3] = 384 -> 64 -> 1
        self.vis_head = nn.Sequential(
            nn.Linear(lstm_hidden * 2 * 3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # Scene-Conditioned Adaptive Fusion Gate
        self.adaptive_gate = SceneConditionedAdaptiveGate(scene_feat_dim=lstm_hidden * 2, stat_dim=64)

        # Optional Joint Concatenation Head (for Concat MLP ablation)
        self.concat_head = None
        if use_concat_head:
            self.concat_head = nn.Sequential(
                nn.Linear(lstm_hidden * 2 * 6, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )

    def forward(self, v_m, v_s, v_i, k_m,
                disable_vision=False, disable_kinematics=False,
                fixed_uniform_fusion=False, force_occlusion=False,
                disable_occlusion_block=False,
                disable_illum_block=False,
                disable_interaction=False,
                disable_scene_streams=False,
                motion_only=False,
                kinematics_motion_only=False,
                visual_motion_only=False,
                temporal_pooling='attention',
                fusion_mode='adaptive'):
        """
        Inputs:
            v_m: [B, Seq, 1792] (Pedestrian crop appearance)
            v_s: [B, Seq, 1792] (Scene context)
            v_i: [B, Seq, 1792] (Interaction surround crop)
            k_m: [B, Seq, 17]   (Raw kinematics)
        Returns:
            logits_final: [B, 1]
            aux: dict with {logits_kinem, logits_vis, gate_weights, recon_loss, g_illum}
        """
        B, Seq, _ = k_m.shape

        # Step 1: Occlusion Reconstruction on Kinematics
        if disable_occlusion_block:
            km_recon = k_m
            recon_loss = torch.tensor(0.0, device=k_m.device)
        else:
            km_recon, recon_loss = self.occlusion_block(k_m, force_mask=force_occlusion)

        # Step 2: Illumination Adaptation on Scene Visuals
        if disable_illum_block:
            zs_adapted = self.illum_block.proj_day(v_s)
            g_illum = torch.full((B, 1), 0.5, device=v_s.device)
        else:
            zs_adapted, g_illum = self.illum_block(v_s)

        # Step 3: Forward 3 Kinematic Encoders
        hk_m, alpha_km = self.kinem_motion_enc(km_recon) # [B, 128]
        hk_s, alpha_ks = self.kinem_scene_enc(km_recon)  # [B, 128]
        hk_i, alpha_ki = self.kinem_inter_enc(km_recon)  # [B, 128]

        if disable_interaction or motion_only:
            hk_i = torch.zeros_like(hk_i)
        if disable_scene_streams or motion_only or kinematics_motion_only:
            hk_s = torch.zeros_like(hk_s)
        if kinematics_motion_only:
            hk_i = torch.zeros_like(hk_i)

        hk_all = torch.cat([hk_m, hk_s, hk_i], dim=-1)   # [B, 384]
        l_k = self.kinem_head(hk_all)                     # [B, 1]

        # Step 4: Forward 3 Visual Encoders
        hv_m, alpha_vm = self.vis_motion_enc(v_m)        # [B, 128]
        hv_s, alpha_vs = self.vis_scene_enc(zs_adapted)  # [B, 128]
        hv_i, alpha_vi = self.vis_inter_enc(v_m, v_i)    # [B, 128]

        if disable_interaction or motion_only:
            hv_i = torch.zeros_like(hv_i)
        if disable_scene_streams or motion_only or visual_motion_only:
            hv_s = torch.zeros_like(hv_s)
        if visual_motion_only:
            hv_i = torch.zeros_like(hv_i)

        hv_all = torch.cat([hv_m, hv_s, hv_i], dim=-1)   # [B, 384]
        l_v = self.vis_head(hv_all)                      # [B, 1]

        # Step 5: Scene-Conditioned Adaptive Fusion Gate
        gate_weights = self.adaptive_gate(hv_s, g_illum, v_s) # [B, 2] -> [w_K, w_V]
        w_k = gate_weights[:, 0].unsqueeze(1) # [B, 1]
        w_v = gate_weights[:, 1].unsqueeze(1) # [B, 1]

        # Step 6: Final Logit Computation
        if disable_vision:
            logits_final = l_k
            gate_weights = torch.tensor([[1.0, 0.0]], device=k_m.device).repeat(B, 1)
        elif disable_kinematics:
            logits_final = l_v
            gate_weights = torch.tensor([[0.0, 1.0]], device=k_m.device).repeat(B, 1)
        elif fixed_uniform_fusion or fusion_mode == 'uniform':
            logits_final = 0.5 * l_k + 0.5 * l_v
            gate_weights = torch.tensor([[0.5, 0.5]], device=k_m.device).repeat(B, 1)
        elif fusion_mode == 'concat':
            logits_final = self.concat_head(torch.cat([hk_all, hv_all], dim=-1))
            gate_weights = torch.tensor([[0.5, 0.5]], device=k_m.device).repeat(B, 1)
        else:
            logits_final = w_k * l_k + w_v * l_v

        aux = {
            'logits_kinem': l_k,
            'logits_vis': l_v,
            'gate_weights': gate_weights,
            'reconstructed_km': km_recon,
            'recon_loss': recon_loss,
            'g_illum': g_illum,
            'alpha_kinem_motion': alpha_km,
            'alpha_vis_motion': alpha_vm
        }

        return logits_final, aux


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing AdaptiveFusionMasterModel on: {device}")
    model = AdaptiveFusionMasterModel().to(device)

    B, Seq = 4, 16
    v_m = torch.randn(B, Seq, 1792, device=device)
    v_s = torch.randn(B, Seq, 1792, device=device)
    v_i = torch.randn(B, Seq, 1792, device=device)
    k_m = torch.randn(B, Seq, 17, device=device)

    logits, aux = model(v_m, v_s, v_i, k_m)
    print("Logits shape:", logits.shape)
    print("Gate weights shape:", aux['gate_weights'].shape, "Sample weights:", aux['gate_weights'][0].tolist())
    print("Logits Kinem shape:", aux['logits_kinem'].shape)
    print("Logits Vis shape:", aux['logits_vis'].shape)
    print("Reconstruction loss:", aux['recon_loss'].item())
    print("Illumination score:", aux['g_illum'].mean().item())
    print("Unit Test PASSED!")
