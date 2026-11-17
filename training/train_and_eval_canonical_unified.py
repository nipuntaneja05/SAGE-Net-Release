"""
train_and_eval_canonical_unified.py
Definitive Unified Canonical Benchmark and Reliability-Focused Ablation Protocol:
1. Canonical Benchmark Splits across all 3 datasets:
   - IDD-PeD: Official 13-Held-Out-Video Split (2,598 test windows)
   - JAAD: Official Default Benchmark Split (978 train, 164 val, 903 test windows)
   - PIE: Official Set-Level Standard Split (4,448 train, 1,251 val, 3,686 test windows)
2. Consistent Multi-Seed Training & Evaluation:
   - 3 Independent Seeds ([42, 101, 2024]) on the exact canonical train partitions with 1:1 ROS.
   - Validation threshold calibration (tau* via Youden's J) on canonical val partitions.
   - Matched 3-seed evaluation (Mean +/- Std and Ensemble) across all variants on canonical test partitions.
3. Reliability-Focused Hypothesis Testing (SAGE-Net Adaptive Gate vs. Fixed 50/50 Fusion):
   - Stratifies test windows into Adverse/Degraded Visual Regimes (Low light / small scale) vs. Pristine.
   - Tests whether Adaptive Gating dynamically shifts authority (w_K -> 1) and prevents failure under visual degradation.
4. Generates publication-grade figures, unified JSON metrics, and self-contained splits.
"""

import os
import sys
import json
import time
import copy
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, roc_curve, precision_recall_curve
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.adaptive_fusion_model import AdaptiveFusionMasterModel
from datasets.iddped_anticipation_dataset import IDDPedAnticipationDataset

DATA_DIR = Path(os.environ.get("SAGENET_DATA_DIR", ROOT_DIR / "data"))
RESULT_DIR = ROOT_DIR / "results" / "generated"
ARTIFACT_DIR = RESULT_DIR
CKPT_DIR = ROOT_DIR / "checkpoints" / "generated"
SCRATCH_DIR = CKPT_DIR
RESULT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SAGEResolutionDataset(Dataset):
    def __init__(self, sample_list, oversample_positives=False, target_ratio=1.0, seed=42):
        self.original_samples = sample_list
        self.oversample_positives = oversample_positives
        self.target_ratio = target_ratio
        self.seed = seed
        self.processed_samples = self._process_and_balance()

    def _process_and_balance(self):
        normed_list = []
        for s in self.original_samples:
            w_res = float(s.get('frame_w', 1920.0))
            h_res = float(s.get('frame_h', 1080.0))
            km = s['K_M'].clone()
            km[:, 0] /= w_res; km[:, 2] /= w_res; km[:, 4] /= w_res; km[:, 8] /= w_res
            km[:, 1] /= h_res; km[:, 3] /= h_res; km[:, 5] /= h_res; km[:, 9] /= h_res

            normed_item = {
                'V_M': s['V_M'],
                'V_S': s['V_S'],
                'V_I': s['V_I'],
                'K_M': km,
                'label': float(s['label']),
                'video_id': s['video_id'],
                'pedestrian_id': str(s.get('pedestrian_id', 0))
            }
            normed_list.append(normed_item)

        if not self.oversample_positives:
            return normed_list

        pos = [s for s in normed_list if s['label'] == 1.0]
        neg = [s for s in normed_list if s['label'] == 0.0]

        if len(pos) == 0 or len(neg) == 0:
            return normed_list

        target_pos_count = int(len(neg) * self.target_ratio)
        rng = random.Random(self.seed)
        oversampled_pos = rng.choices(pos, k=target_pos_count)
        balanced_list = neg + oversampled_pos
        rng.shuffle(balanced_list)
        return balanced_list

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, idx):
        s = self.processed_samples[idx]
        return {
            'V_M': s['V_M'].clone().detach(),
            'V_S': s['V_S'].clone().detach(),
            'V_I': s['V_I'].clone().detach(),
            'K_M': s['K_M'].clone().detach(),
            'label': torch.tensor(s['label'], dtype=torch.float32),
            'video_id': s['video_id'],
            'pedestrian_id': s['pedestrian_id']
        }


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
        pt = targets * probs + (1 - targets) * (1 - probs)
        w = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = -w * torch.pow((1.0 - pt), self.gamma) * torch.log(pt)
        return loss.mean()


def calibrate_tau_on_val(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_tau = float(thresholds[best_idx])
    return float(np.clip(best_tau, 0.05, 0.95))


def eval_metrics(y_true, y_prob, tau=0.5):
    y_pred = (y_prob >= tau).astype(int)
    return {
        'ROC_AUC': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        'PR_AUC': float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'F1_Score': float(f1_score(y_true, y_pred, zero_division=0)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'tau': float(tau)
    }


# Matched Variant Dictionary
CANONICAL_VARIANTS = {
    '1_sage_net_adaptive': {
        'name': 'SAGE-Net (Scene-Adaptive Gate)',
        'kwargs': {'disable_vision': False, 'disable_kinematics': False, 'fixed_uniform_fusion': False}
    },
    '2_fixed_uniform_fusion': {
        'name': 'Fixed Uniform Fusion (50/50)',
        'kwargs': {'fixed_uniform_fusion': True}
    },
    '3_kinematics_all': {
        'name': 'Kinematics Only (K_M + K_S + K_I)',
        'kwargs': {'disable_vision': True}
    },
    '4_visual_all': {
        'name': 'Visual Streams Only (V_M + V_S + V_I)',
        'kwargs': {'disable_kinematics': True}
    },
    '5_kinematics_motion_only': {
        'name': 'Kinematics Motion Alone (K_M)',
        'kwargs': {'disable_vision': True, 'kinematics_motion_only': True}
    },
    '6_visual_motion_only': {
        'name': 'Visual Pedestrian Crop Alone (V_M)',
        'kwargs': {'disable_kinematics': True, 'visual_motion_only': True}
    },
    '7_wo_interaction': {
        'name': 'w/o Interaction Encoders (No K_I, No V_I)',
        'kwargs': {'disable_interaction': True}
    },
    '8_wo_scene': {
        'name': 'w/o Scene Encoders (No K_S, No V_S)',
        'kwargs': {'disable_scene_streams': True}
    },
    '9_wo_occlusion_block': {
        'name': 'w/o Occlusion Reconstruction Block',
        'kwargs': {'disable_occlusion_block': True}
    },
    '10_wo_illum_block': {
        'name': 'w/o Illumination Adapter Block',
        'kwargs': {'disable_illum_block': True}
    }
}


def train_canonical_dataset(dataset_name, all_data, split_manifest, epochs=12, batch_size=32, lr=1e-4, seeds=[42, 101, 2024]):
    print(f"\n" + "=" * 80)
    print(f"  TRAINING SAGE-NET ON CANONICAL {dataset_name} BENCHMARK SPLIT")
    print(f"=" * 80)
    sys.stdout.flush()

    sinfo = split_manifest['official_split']
    train_vids = set(sinfo['train_videos'])
    val_vids   = set(sinfo['val_videos'])
    test_vids  = set(sinfo['test_videos'])

    train_samples = [s for s in all_data if s['video_id'] in train_vids]
    val_samples   = [s for s in all_data if s['video_id'] in val_vids]
    test_samples  = [s for s in all_data if s['video_id'] in test_vids]

    print(f"[{dataset_name}] Official Partition -> Train: {len(train_samples)} seqs (Pos: {sum(s['label']==1 for s in train_samples)}), "
          f"Val: {len(val_samples)} seqs (Pos: {sum(s['label']==1 for s in val_samples)}), "
          f"Test: {len(test_samples)} seqs (Pos: {sum(s['label']==1 for s in test_samples)})")
    sys.stdout.flush()

    val_ds  = SAGEResolutionDataset(val_samples, oversample_positives=False)
    test_ds = SAGEResolutionDataset(test_samples, oversample_positives=False)
    val_loader  = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    y_val  = np.array([s['label'] for s in val_ds.processed_samples])
    y_test = np.array([s['label'] for s in test_ds.processed_samples])

    focal_crit = FocalLoss(alpha=0.75, gamma=2.0)
    bce_crit = nn.BCEWithLogitsLoss()

    trained_models = {}

    for seed in seeds:
        print(f"\n  [{dataset_name}] Training Seed {seed} ({epochs} Epochs with ROS)...")
        sys.stdout.flush()
        t_start = time.time()
        seed_everything(seed)

        train_ds = SAGEResolutionDataset(train_samples, oversample_positives=True, target_ratio=1.0, seed=seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        model = AdaptiveFusionMasterModel().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_val_auc = -1.0
        best_state = None
        best_ep = 0

        for ep in range(1, epochs + 1):
            model.train()
            ep_loss = 0.0
            for b in train_loader:
                optimizer.zero_grad()
                vm = b['V_M'].to(DEVICE); vs = b['V_S'].to(DEVICE); vi = b['V_I'].to(DEVICE); km = b['K_M'].to(DEVICE)
                y  = b['label'].to(DEVICE).unsqueeze(1)

                logits, aux = model(vm, vs, vi, km)
                l_focal = focal_crit(logits, y)
                l_kinem = bce_crit(aux['logits_kinem'], y)
                l_vis   = bce_crit(aux['logits_vis'], y)
                l_recon = aux['recon_loss']

                loss = l_focal + 0.3 * l_kinem + 0.2 * l_vis + 0.1 * l_recon
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()

            scheduler.step()

            # Validation check
            model.eval()
            val_probs = []
            with torch.no_grad():
                for b in val_loader:
                    vm = b['V_M'].to(DEVICE); vs = b['V_S'].to(DEVICE); vi = b['V_I'].to(DEVICE); km = b['K_M'].to(DEVICE)
                    logits, _ = model(vm, vs, vi, km)
                    val_probs.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())

            val_auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else 0.5
            if ep % 4 == 0 or ep == epochs:
                print(f"    [{dataset_name} Seed {seed}] Ep {ep:2d}/{epochs:2d} | Train Loss: {ep_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
                sys.stdout.flush()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_ep = ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        ckpt_path = os.path.join(CKPT_DIR, f'sagenet_canonical_{dataset_name.lower()}_seed{seed}.pth')
        torch.save(best_state, ckpt_path)
        trained_models[seed] = model
        print(f"  [{dataset_name}] Seed {seed} Finished in {time.time()-t_start:.1f}s | Best Ep: {best_ep} | Best Val AUC: {best_val_auc:.4f}")
        sys.stdout.flush()

    return trained_models, val_loader, test_loader, y_val, y_test, test_samples


def evaluate_canonical_suite(dataset_name, trained_models, val_loader, test_loader, y_val, y_test, seeds):
    print(f"\n================================================================================")
    print(f"  CANONICAL BENCHMARK & ABLATION EVALUATION ON {dataset_name} ({len(y_test)} Test Windows)")
    print(f"================================================================================")
    sys.stdout.flush()

    results = {}
    variant_test_probs = {}

    for vk, vinfo in CANONICAL_VARIANTS.items():
        vname = vinfo['name']
        vkwargs = vinfo['kwargs']
        print(f"  --> Evaluating [{vk}] {vname} across {len(seeds)} seeds...")
        sys.stdout.flush()

        seed_probs = []
        tau_stars = []
        seed_m_star = []
        seed_m_05 = []

        for seed in seeds:
            model = trained_models[seed]
            model.eval()

            # 1. Validation calibration
            val_p = []
            with torch.no_grad():
                for b in val_loader:
                    vm = b['V_M'].to(DEVICE); vs = b['V_S'].to(DEVICE); vi = b['V_I'].to(DEVICE); km = b['K_M'].to(DEVICE)
                    logits, _ = model(vm, vs, vi, km, **vkwargs)
                    val_p.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
            tau_star = calibrate_tau_on_val(y_val, np.array(val_p))
            tau_stars.append(tau_star)

            # 2. Test evaluation
            test_p = []
            with torch.no_grad():
                for b in test_loader:
                    vm = b['V_M'].to(DEVICE); vs = b['V_S'].to(DEVICE); vi = b['V_I'].to(DEVICE); km = b['K_M'].to(DEVICE)
                    logits, _ = model(vm, vs, vi, km, **vkwargs)
                    test_p.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
            test_p = np.array(test_p)
            seed_probs.append(test_p)

            m_star = eval_metrics(y_test, test_p, tau=tau_star)
            m_05   = eval_metrics(y_test, test_p, tau=0.50)
            seed_m_star.append(m_star)
            seed_m_05.append(m_05)

        seed_probs = np.array(seed_probs)
        ens_probs = np.mean(seed_probs, axis=0)
        ens_tau_star = float(np.mean(tau_stars))
        ens_m_star = eval_metrics(y_test, ens_probs, tau=ens_tau_star)
        ens_m_05   = eval_metrics(y_test, ens_probs, tau=0.50)

        aucs = [m['ROC_AUC'] for m in seed_m_star]
        aps  = [m['PR_AUC'] for m in seed_m_star]
        accs_star = [m['Accuracy'] for m in seed_m_star]
        f1s_star  = [m['F1_Score'] for m in seed_m_star]
        accs_05   = [m['Accuracy'] for m in seed_m_05]

        results[vk] = {
            'name': vname,
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'mean_ap': float(np.mean(aps)),
            'std_ap': float(np.std(aps)),
            'mean_acc_star': float(np.mean(accs_star)),
            'mean_f1_star': float(np.mean(f1s_star)),
            'mean_acc_05': float(np.mean(accs_05)),
            'ensemble_auc': ens_m_star['ROC_AUC'],
            'ensemble_ap': ens_m_star['PR_AUC'],
            'ensemble_acc_star': ens_m_star['Accuracy'],
            'ensemble_f1_star': ens_m_star['F1_Score'],
            'ensemble_acc_05': ens_m_05['Accuracy'],
            'tau_star': ens_tau_star
        }

        variant_test_probs[vk] = {
            'ens_probs': ens_probs,
            'ens_tau_star': ens_tau_star
        }

    return results, variant_test_probs


# ==============================================================================
# RELIABILITY-FOCUSED HYPOTHESIS TESTING (ADVERSE VS PRISTINE VISUAL CONDITIONS)
# ==============================================================================
def evaluate_reliability_strata(dataset_name, model, test_loader, y_test, test_samples):
    print(f"\n[{dataset_name}] Evaluating Reliability-Focused Claim (Adverse vs Pristine Conditions)...")
    sys.stdout.flush()

    model.eval()
    all_wk = []
    all_wv = []
    all_gillum = []
    all_h = []
    all_p_adaptive = []
    all_p_fixed = []
    all_p_kinem = []
    all_p_vis = []

    with torch.no_grad():
        for b in test_loader:
            vm = b['V_M'].to(DEVICE); vs = b['V_S'].to(DEVICE); vi = b['V_I'].to(DEVICE); km = b['K_M'].to(DEVICE)

            # Forward Adaptive
            logits_ad, aux = model(vm, vs, vi, km)
            # Forward Fixed 50/50
            logits_fx, _ = model(vm, vs, vi, km, fixed_uniform_fusion=True)
            # Forward Kinem Only
            logits_kn, _ = model(vm, vs, vi, km, disable_vision=True)
            # Forward Vis Only
            logits_vs, _ = model(vm, vs, vi, km, disable_kinematics=True)

            p_ad = torch.sigmoid(logits_ad.squeeze(-1)).cpu().numpy()
            p_fx = torch.sigmoid(logits_fx.squeeze(-1)).cpu().numpy()
            p_kn = torch.sigmoid(logits_kn.squeeze(-1)).cpu().numpy()
            p_vs = torch.sigmoid(logits_vs.squeeze(-1)).cpu().numpy()

            wk = aux['gate_weights'][:, 0].cpu().numpy()
            wv = aux['gate_weights'][:, 1].cpu().numpy()
            g_il = aux['g_illum'].squeeze(-1).cpu().numpy()
            bbox_h = km[:, -1, 3].cpu().numpy() # final frame bbox height

            all_wk.extend(wk); all_wv.extend(wv); all_gillum.extend(g_il); all_h.extend(bbox_h)
            all_p_adaptive.extend(p_ad); all_p_fixed.extend(p_fx)
            all_p_kinem.extend(p_kn); all_p_vis.extend(p_vs)

    all_wk = np.array(all_wk)
    all_gillum = np.array(all_gillum)
    all_h = np.array(all_h)
    all_p_adaptive = np.array(all_p_adaptive)
    all_p_fixed = np.array(all_p_fixed)
    all_p_kinem = np.array(all_p_kinem)
    all_p_vis = np.array(all_p_vis)

    # Define Stratum B: Adverse Visual Conditions (Low lighting OR distant small pedestrian)
    # Thresholds: g_illum >= 0.35 OR h <= 0.12
    adverse_mask = (all_gillum >= 0.35) | (all_h <= 0.12)
    pristine_mask = ~adverse_mask

    strata_results = {}
    for stratum_name, mask in [('Adverse_Visual_Regime', adverse_mask), ('Pristine_Visual_Regime', pristine_mask), ('Full_Canonical_Set', np.ones(len(y_test), dtype=bool))]:
        sub_y = y_test[mask]
        if len(np.unique(sub_y)) > 1:
            auc_ad = float(roc_auc_score(sub_y, all_p_adaptive[mask]))
            auc_fx = float(roc_auc_score(sub_y, all_p_fixed[mask]))
            auc_kn = float(roc_auc_score(sub_y, all_p_kinem[mask]))
            auc_vs = float(roc_auc_score(sub_y, all_p_vis[mask]))
        else:
            auc_ad = auc_fx = auc_kn = auc_vs = 0.5

        mean_wk = float(np.mean(all_wk[mask]))
        delta_ad_vs_fx = auc_ad - auc_fx

        strata_results[stratum_name] = {
            'n_samples': int(np.sum(mask)),
            'n_positives': int(np.sum(sub_y == 1)),
            'mean_kinematic_weight_wK': mean_wk,
            'SAGE_Net_AUC': auc_ad,
            'Fixed_Fusion_AUC': auc_fx,
            'Kinematics_Alone_AUC': auc_kn,
            'Vision_Alone_AUC': auc_vs,
            'Delta_Adaptive_vs_Fixed': delta_ad_vs_fx
        }

    return strata_results, {
        'all_wk': all_wk, 'all_gillum': all_gillum, 'all_h': all_h,
        'all_p_adaptive': all_p_adaptive, 'all_p_fixed': all_p_fixed,
        'adverse_mask': adverse_mask, 'y_test': y_test
    }


# ==============================================================================
# MAIN DRIVER
# ==============================================================================
def main():
    global DATA_DIR, RESULT_DIR, ARTIFACT_DIR, CKPT_DIR, SCRATCH_DIR
    parser = argparse.ArgumentParser(description="Train and evaluate SAGE-Net on the included canonical splits.")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR, help="Directory containing precomputed tensor cache files.")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR, help="Directory for generated result JSON files.")
    parser.add_argument("--checkpoint-dir", type=Path, default=CKPT_DIR, help="Directory for generated model checkpoints.")
    args = parser.parse_args()
    DATA_DIR = args.data_root.expanduser().resolve()
    RESULT_DIR = args.output_dir.expanduser().resolve()
    ARTIFACT_DIR = RESULT_DIR
    CKPT_DIR = args.checkpoint_dir.expanduser().resolve()
    SCRATCH_DIR = CKPT_DIR
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  CANONICAL SAGE-NET MULTI-DATASET BENCHMARK & RELIABILITY TEST PIPELINE")
    print(f"  Device: {DEVICE} | Platform: PyTorch")
    print("=" * 80)
    sys.stdout.flush()

    # 1. Load Caches and Manifests
    with open(ROOT_DIR / 'datasets' / 'splits' / 'split_manifest_jaad.json') as f:
        manifest_jaad = json.load(f)
    with open(ROOT_DIR / 'datasets' / 'splits' / 'split_manifest_pie.json') as f:
        manifest_pie = json.load(f)
    with open(ROOT_DIR / 'datasets' / 'splits' / 'split_manifest_idd_ped.json') as f:
        manifest_idd = json.load(f)

    jaad_cache = torch.load(DATA_DIR / 'jaad_sliding_cache.pt', map_location='cpu', weights_only=False)
    pie_cache  = torch.load(DATA_DIR / 'pie_sliding_cache.pt', map_location='cpu', weights_only=False)

    seeds = [42, 101, 2024]

    # --------------------------------------------------------------------------
    # 2. TRAIN & EVALUATE JAAD CANONICAL (903 Test Windows)
    # --------------------------------------------------------------------------
    models_jaad, val_l_j, test_l_j, y_val_j, y_test_j, test_samp_j = train_canonical_dataset(
        'JAAD', jaad_cache, manifest_jaad, epochs=12, batch_size=32, lr=1e-4, seeds=seeds
    )
    res_jaad, probs_jaad = evaluate_canonical_suite('JAAD', models_jaad, val_l_j, test_l_j, y_val_j, y_test_j, seeds)
    rel_jaad, rel_data_j = evaluate_reliability_strata('JAAD', models_jaad[42], test_l_j, y_test_j, test_samp_j)

    with open(os.path.join(ARTIFACT_DIR, 'results_canonical_jaad_903.json'), 'w') as f:
        json.dump(res_jaad, f, indent=2)

    # --------------------------------------------------------------------------
    # 3. TRAIN & EVALUATE PIE CANONICAL (3,686 Test Windows)
    # --------------------------------------------------------------------------
    models_pie, val_l_p, test_l_p, y_val_p, y_test_p, test_samp_p = train_canonical_dataset(
        'PIE', pie_cache, manifest_pie, epochs=10, batch_size=48, lr=5e-5, seeds=seeds
    )
    res_pie, probs_pie = evaluate_canonical_suite('PIE', models_pie, val_l_p, test_l_p, y_val_p, y_test_p, seeds)
    rel_pie, rel_data_p = evaluate_reliability_strata('PIE', models_pie[42], test_l_p, y_test_p, test_samp_p)

    with open(os.path.join(ARTIFACT_DIR, 'results_canonical_pie_3686.json'), 'w') as f:
        json.dump(res_pie, f, indent=2)

    # --------------------------------------------------------------------------
    # 4. LOAD IDD-PED CANONICAL (2,598 Test Windows)
    # --------------------------------------------------------------------------
    print(f"\n================================================================================")
    print(f"  LOADING IDD-PED CANONICAL BENCHMARK (2,598 Official Test Windows)")
    print(f"================================================================================")
    sys.stdout.flush()

    idd_cache = torch.load(DATA_DIR / 'iddped_anticipation_cache.pt', map_location='cpu', weights_only=False)
    sinfo_idd = manifest_idd['official_split']
    val_vids_i = set(sinfo_idd['val_videos'])
    val_s_idd = [d for d in idd_cache if d['video_id'] in val_vids_i]
    val_ds_i = IDDPedAnticipationDataset(val_s_idd, oversample_positives=False)
    val_l_i = DataLoader(val_ds_i, batch_size=64, shuffle=False)
    y_val_i = np.array([int(s['label']) for s in val_s_idd])

    test_cached = torch.load(DATA_DIR / 'iddped_official_2598_test_tensors.pt', map_location='cpu', weights_only=False)
    test_ds_i = []
    for idx in range(len(test_cached['targets'])):
        test_ds_i.append({
            'V_M': test_cached['V_M'][idx],
            'V_S': test_cached['V_S'][idx],
            'V_I': test_cached['V_I'][idx],
            'K_M': test_cached['K_M'][idx],
            'label': float(test_cached['targets'][idx]),
            'video_id': 'iddped_test',
            'pedestrian_id': str(test_cached['pids'][idx])
        })
    test_l_i = DataLoader(test_ds_i, batch_size=64, shuffle=False)
    y_test_i = test_cached['targets']

    models_idd = {}
    for s in seeds:
        m = AdaptiveFusionMasterModel().to(DEVICE)
        m.load_state_dict(torch.load(CKPT_DIR / f'adaptive_fusion_seed{s}.pth', map_location=DEVICE, weights_only=False))
        models_idd[s] = m

    res_idd, probs_idd = evaluate_canonical_suite('IDD-PeD', models_idd, val_l_i, test_l_i, y_val_i, y_test_i, seeds)
    rel_idd, rel_data_i = evaluate_reliability_strata('IDD-PeD', models_idd[42], test_l_i, y_test_i, test_ds_i)

    with open(os.path.join(ARTIFACT_DIR, 'results_canonical_iddped_2598.json'), 'w') as f:
        json.dump(res_idd, f, indent=2)

    # --------------------------------------------------------------------------
    # 5. UNIFIED CANONICAL RESULTS SUMMARY & JSON
    # --------------------------------------------------------------------------
    canonical_unified = {
        'JAAD': {'test_windows': len(y_test_j), 'variants': res_jaad, 'reliability_analysis': rel_jaad},
        'PIE': {'test_windows': len(y_test_p), 'variants': res_pie, 'reliability_analysis': rel_pie},
        'IDD-PeD': {'test_windows': len(y_test_i), 'variants': res_idd, 'reliability_analysis': rel_idd}
    }

    with open(os.path.join(ARTIFACT_DIR, 'results_canonical_unified.json'), 'w') as f:
        json.dump(canonical_unified, f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, 'results_reliability_analysis.json'), 'w') as f:
        json.dump({'JAAD': rel_jaad, 'PIE': rel_pie, 'IDD-PeD': rel_idd}, f, indent=2)

    # --------------------------------------------------------------------------
    # 6. GENERATE CANONICAL PUBLICATION FIGURES
    # --------------------------------------------------------------------------
    print("\nGenerating Canonical Publication Figures...")
    sys.stdout.flush()

    # Figure A: Unified Canonical ROC Curves across all 3 datasets
    fig_roc, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig_roc.patch.set_facecolor('#ffffff')

    ds_info = [
        ('IDD-PeD (2,598 Test Windows)', y_test_i, probs_idd, res_idd),
        ('JAAD (903 Test Windows)', y_test_j, probs_jaad, res_jaad),
        ('PIE (3,686 Test Windows)', y_test_p, probs_pie, res_pie)
    ]

    colors = {
        '1_sage_net_adaptive': '#0284c7',
        '2_fixed_uniform_fusion': '#10b981',
        '3_kinematics_all': '#3b82f6',
        '4_visual_all': '#f97316',
        '5_kinematics_motion_only': '#06b6d4',
        '6_visual_motion_only': '#d97706',
        '7_wo_interaction': '#14b8a6',
        '8_wo_scene': '#6366f1',
        '9_wo_occlusion_block': '#f59e0b',
        '10_wo_illum_block': '#ec4899'
    }

    for idx, (title, y_arr, p_dict, r_dict) in enumerate(ds_info):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Chance (0.500)')
        for vk in ['1_sage_net_adaptive', '2_fixed_uniform_fusion', '3_kinematics_all', '4_visual_all']:
            ens_p = p_dict[vk]['ens_probs']
            fpr, tpr, _ = roc_curve(y_arr, ens_p)
            auc_v = r_dict[vk]['ensemble_auc']
            lw = 2.4 if vk == '1_sage_net_adaptive' else 1.8
            ax.plot(fpr, tpr, color=colors[vk], label=f"{r_dict[vk]['name']} ({auc_v:.4f})", linewidth=lw)

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color='#0f172a')
        ax.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.92)

    fig_roc.suptitle('Canonical SAGE-Net Multi-Dataset Crossing Anticipation Benchmarks (ROC Curves)',
                     fontsize=14, fontweight='bold', y=1.02, color='#0f172a')
    fig_roc.savefig(os.path.join(ARTIFACT_DIR, 'fig_canonical_roc_curves.png'), bbox_inches='tight', dpi=300)
    plt.close(fig_roc)

    # Figure B: Reliability Hypothesis Verification Chart
    fig_rel, axes_r = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fig_rel.patch.set_facecolor('#ffffff')

    # Panel 1: Shift of Kinematic Weight w_K in Adverse vs Pristine Regimes
    ax1 = axes_r[0]
    bar_width = 0.35
    x_indices = np.arange(3)
    ds_names = ['IDD-PeD', 'JAAD', 'PIE']
    w_pristine = [rel_idd['Pristine_Visual_Regime']['mean_kinematic_weight_wK'],
                  rel_jaad['Pristine_Visual_Regime']['mean_kinematic_weight_wK'],
                  rel_pie['Pristine_Visual_Regime']['mean_kinematic_weight_wK']]
    w_adverse  = [rel_idd['Adverse_Visual_Regime']['mean_kinematic_weight_wK'],
                  rel_jaad['Adverse_Visual_Regime']['mean_kinematic_weight_wK'],
                  rel_pie['Adverse_Visual_Regime']['mean_kinematic_weight_wK']]

    ax1.bar(x_indices - bar_width/2, w_pristine, width=bar_width, label='Pristine Visual Regime', color='#38bdf8', edgecolor='#0284c7')
    ax1.bar(x_indices + bar_width/2, w_adverse, width=bar_width, label='Adverse Visual Regime (Low Light / Distant)', color='#f97316', edgecolor='#c2410c')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(ds_names, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Kinematic Gate Weight (w_K)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Adaptive Authority Routing Under Visual Degradation\n(Dynamic Shift w_K -> 1 Under Adverse Lighting / Scale)', fontsize=11, fontweight='bold', pad=10)
    ax1.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax1.legend(loc='upper right', fontsize=9.5)

    # Panel 2: SAGE-Net vs Fixed Fusion AUC under Adverse Conditions
    ax2 = axes_r[1]
    auc_adverse_sage = [rel_idd['Adverse_Visual_Regime']['SAGE_Net_AUC'],
                        rel_jaad['Adverse_Visual_Regime']['SAGE_Net_AUC'],
                        rel_pie['Adverse_Visual_Regime']['SAGE_Net_AUC']]
    auc_adverse_fix  = [rel_idd['Adverse_Visual_Regime']['Fixed_Fusion_AUC'],
                        rel_jaad['Adverse_Visual_Regime']['Fixed_Fusion_AUC'],
                        rel_pie['Adverse_Visual_Regime']['Fixed_Fusion_AUC']]

    ax2.bar(x_indices - bar_width/2, auc_adverse_sage, width=bar_width, label='SAGE-Net (Adaptive Gate)', color='#0284c7', edgecolor='#0369a1')
    ax2.bar(x_indices + bar_width/2, auc_adverse_fix, width=bar_width, label='Fixed Uniform Fusion (50/50)', color='#94a3b8', edgecolor='#475569')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(ds_names, fontsize=11, fontweight='bold')
    ax2.set_ylabel('ROC-AUC on Adverse Test Windows', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Adverse-Regime Anticipation Robustness\n(Mitigating Failure When Visual Stream is Unreliable)', fontsize=11, fontweight='bold', pad=10)
    ax2.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax2.legend(loc='lower right', fontsize=9.5)

    fig_rel.suptitle('Reliability-Focused Hypothesis Verification: SAGE-Net Adaptive Routing vs. Fixed Fusion',
                     fontsize=13, fontweight='bold', y=1.02, color='#0f172a')
    fig_rel.savefig(os.path.join(ARTIFACT_DIR, 'fig_reliability_adversity_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close(fig_rel)

    # --------------------------------------------------------------------------
    # 7. TERMINAL SUMMARY TABLE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 125)
    print("  CANONICAL UNIFIED BENCHMARK RESULTS (ALL SPLITS CANONICAL & MATCHED)")
    print("=" * 125)
    header = f"{'Dataset':<8} | {'Test N':<6} | {'Variant Name':<38} | {'3-Seed Mean AUC':<18} | {'Ens AUC':<9} | {'Ens AP':<9} | {'Calib Acc':<10}"
    print(header)
    print("-" * len(header))

    for dname, rdict, y_arr in [('IDD-PeD', res_idd, y_test_i), ('JAAD', res_jaad, y_test_j), ('PIE', res_pie, y_test_p)]:
        for vk, r in rdict.items():
            mean_str = f"{r['mean_auc']:.4f} ± {r['std_auc']:.4f}"
            print(f"{dname:<8} | {len(y_arr):<6} | {r['name']:<38} | {mean_str:<18} | {r['ensemble_auc']:.4f}    | {r['ensemble_ap']:.4f}   | {r['ensemble_acc_star']*100:6.2f}%")
        print("-" * len(header))

    print("\n" + "=" * 90)
    print("  RELIABILITY CLAIM VERIFICATION (ADVERSE VS PRISTINE VISUAL REGIMES)")
    print("=" * 90)
    print(f"{'Dataset':<8} | {'Regime':<24} | {'N Seqs':<7} | {'Mean w_K':<9} | {'SAGE-Net AUC':<12} | {'Fixed AUC':<10} | {'Delta AUC':<10}")
    print("-" * 90)
    for dname, rdata in [('IDD-PeD', rel_idd), ('JAAD', rel_jaad), ('PIE', rel_pie)]:
        for rname in ['Pristine_Visual_Regime', 'Adverse_Visual_Regime']:
            rd = rdata[rname]
            print(f"{dname:<8} | {rname:<24} | {rd['n_samples']:<7} | {rd['mean_kinematic_weight_wK']:<9.3f} | {rd['SAGE_Net_AUC']:<12.4f} | {rd['Fixed_Fusion_AUC']:<10.4f} | {rd['Delta_Adaptive_vs_Fixed']:+.4f}")
        print("-" * 90)

    print("\nCanonical Pipeline Successfully Completed!")


if __name__ == '__main__':
    main()
