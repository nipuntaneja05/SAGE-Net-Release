"""
test_adaptive_architecture_unit.py
Comprehensive automated test suite for the Adaptive Fusion Architecture:
- Verifies shapes of all 3 Kinematic Encoders and 3 Visual Encoders
- Verifies Occlusion Motion Reconstruction Block masking and denoising
- Verifies Illumination Block dual-pathway modulation
- Verifies Scene-Conditioned Adaptive Fusion Gate weights
- Verifies full gradient backpropagation across multi-task loss components
"""

import unittest
import torch
import torch.nn.functional as F

from adaptive_fusion_model import (
    OcclusionReconstructionBlock,
    IlluminationBlock,
    KinematicMotionEncoder,
    KinematicSceneEncoder,
    KinematicInteractionEncoder,
    VisualMotionEncoder,
    VisualSceneEncoder,
    VisualInteractionEncoder,
    SceneConditionedAdaptiveGate,
    AdaptiveFusionMasterModel
)

class TestAdaptiveArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.B = 4
        cls.Seq = 16
        cls.vis_dim = 1792
        cls.km_dim = 17

    def test_occlusion_reconstruction_block(self):
        block = OcclusionReconstructionBlock(in_dim=self.km_dim).to(self.device)
        km = torch.randn(self.B, self.Seq, self.km_dim, device=self.device)
        km_recon, recon_loss = block(km, force_mask=True)
        self.assertEqual(km_recon.shape, (self.B, self.Seq, self.km_dim))
        self.assertTrue(torch.is_tensor(recon_loss))
        self.assertGreater(recon_loss.item(), 0.0)

    def test_illumination_block(self):
        illum = IlluminationBlock(vis_dim=self.vis_dim).to(self.device)
        v_s = torch.randn(self.B, self.Seq, self.vis_dim, device=self.device)
        zs_adapted, g_illum = illum(v_s)
        self.assertEqual(zs_adapted.shape, (self.B, self.Seq, 128))
        self.assertEqual(g_illum.shape, (self.B, 1))
        self.assertTrue((g_illum >= 0.0).all() and (g_illum <= 1.0).all())

    def test_3_kinematic_encoders(self):
        k_m = torch.randn(self.B, self.Seq, self.km_dim, device=self.device)
        enc_m = KinematicMotionEncoder().to(self.device)
        enc_s = KinematicSceneEncoder().to(self.device)
        enc_i = KinematicInteractionEncoder().to(self.device)

        h_m, a_m = enc_m(k_m)
        h_s, a_s = enc_s(k_m)
        h_i, a_i = enc_i(k_m)

        self.assertEqual(h_m.shape, (self.B, 128))
        self.assertEqual(h_s.shape, (self.B, 128))
        self.assertEqual(h_i.shape, (self.B, 128))
        self.assertEqual(a_m.shape, (self.B, self.Seq))

    def test_3_visual_encoders(self):
        v_m = torch.randn(self.B, self.Seq, self.vis_dim, device=self.device)
        v_s_adapt = torch.randn(self.B, self.Seq, 128, device=self.device)
        v_i = torch.randn(self.B, self.Seq, self.vis_dim, device=self.device)

        enc_vm = VisualMotionEncoder().to(self.device)
        enc_vs = VisualSceneEncoder().to(self.device)
        enc_vi = VisualInteractionEncoder().to(self.device)

        h_vm, _ = enc_vm(v_m)
        h_vs, _ = enc_vs(v_s_adapt)
        h_vi, _ = enc_vi(v_m, v_i)

        self.assertEqual(h_vm.shape, (self.B, 128))
        self.assertEqual(h_vs.shape, (self.B, 128))
        self.assertEqual(h_vi.shape, (self.B, 128))

    def test_adaptive_gate_weights_sum(self):
        gate = SceneConditionedAdaptiveGate().to(self.device)
        h_vs = torch.randn(self.B, 128, device=self.device)
        g_illum = torch.rand(self.B, 1, device=self.device)
        v_s = torch.randn(self.B, self.Seq, self.vis_dim, device=self.device)

        weights = gate(h_vs, g_illum, v_s)
        self.assertEqual(weights.shape, (self.B, 2))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(self.B, device=self.device), atol=1e-5))

    def test_master_model_gradient_flow(self):
        model = AdaptiveFusionMasterModel().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        v_m = torch.randn(self.B, self.Seq, self.vis_dim, device=self.device, requires_grad=True)
        v_s = torch.randn(self.B, self.Seq, self.vis_dim, device=self.device, requires_grad=True)
        v_i = torch.randn(self.B, self.Seq, self.vis_dim, device=self.device, requires_grad=True)
        k_m = torch.randn(self.B, self.Seq, self.km_dim, device=self.device, requires_grad=True)
        target = torch.randint(0, 2, (self.B, 1), device=self.device).float()

        optimizer.zero_grad()
        logits, aux = model(v_m, v_s, v_i, k_m)

        l_final = F.binary_cross_entropy_with_logits(logits, target)
        l_kinem = F.binary_cross_entropy_with_logits(aux['logits_kinem'], target)
        l_vis   = F.binary_cross_entropy_with_logits(aux['logits_vis'], target)
        loss = l_final + 0.3 * l_kinem + 0.2 * l_vis + 0.1 * aux['recon_loss']

        loss.backward()

        # Check gradients exist and no NaN in parameters
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Gradient is None for {name}")
                self.assertFalse(torch.isnan(p.grad).any(), f"NaN gradient in {name}")
                self.assertFalse(torch.isinf(p.grad).any(), f"Inf gradient in {name}")

        optimizer.step()

    def test_all_ablation_variants(self):
        model = AdaptiveFusionMasterModel(use_concat_head=True).to(self.device)
        v_m = torch.randn(self.B, self.Seq, self.vis_dim, device=self.device)
        v_s = torch.randn(self.B, self.Seq, self.vis_dim, device=self.device)
        v_i = torch.randn(self.B, self.Seq, self.vis_dim, device=self.device)
        k_m = torch.randn(self.B, self.Seq, self.km_dim, device=self.device)

        ablation_settings = [
            ("Full Model", {}),
            ("Fixed Uniform", {"fixed_uniform_fusion": True}),
            ("Concat MLP", {"fusion_mode": "concat"}),
            ("w/o Occlusion Block", {"disable_occlusion_block": True}),
            ("w/o Illumination Block", {"disable_illum_block": True}),
            ("w/o Both Robustness Blocks", {"disable_occlusion_block": True, "disable_illum_block": True}),
            ("Kinematics Only", {"disable_vision": True}),
            ("Kinematics Motion Only", {"disable_vision": True, "kinematics_motion_only": True}),
            ("Visual Only", {"disable_kinematics": True}),
            ("Visual Motion Only", {"disable_kinematics": True, "visual_motion_only": True}),
            ("w/o Interaction", {"disable_interaction": True}),
            ("w/o Scene Streams", {"disable_scene_streams": True}),
            ("Motion Only", {"motion_only": True})
        ]

        for name, kwargs in ablation_settings:
            logits, aux = model(v_m, v_s, v_i, k_m, **kwargs)
            self.assertEqual(logits.shape, (self.B, 1), f"Failed for {name}")
            self.assertFalse(torch.isnan(logits).any(), f"NaN in logits for {name}")

if __name__ == '__main__':
    unittest.main()
