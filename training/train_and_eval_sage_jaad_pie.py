"""
train_and_eval_sage_jaad_pie.py
Master Training, Evaluation, and Plotting Driver for SAGE-Net (Scene-Adaptive Gated Encoders Network)
on JAAD and PIE Datasets:
1. Multi-Seed Training (Seeds: [42, 101, 2024]) on JAAD and PIE with Random Over-Sampling (ROS)
2. Evaluation of 4 Matched Variants:
   - Kinematics Only (K_M)
   - Visual Streams Only (V_M, V_S, V_I)
   - Fixed Uniform Fusion (50/50)
   - SAGE-Net Full Architecture (Scene-Conditioned Adaptive Fusion Gate)
   - SAGE-Net Multi-Seed Ensemble
3. Validation Threshold Calibration (tau* via Youden's J)
4. Full Metrics: ROC-AUC, PR-AUC (AP), Accuracy (tau*), F1 (tau*), Precision, Recall, Accuracy (0.50)
5. Generation of Publication-Grade Graphs, Heatmaps, and ROC/PR Curves for each dataset.
"""

import os
import sys
import json
import time
import random
import tempfile
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, roc_curve, precision_recall_curve, confusion_matrix
)

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from models.adaptive_fusion_model import AdaptiveFusionMasterModel

BASE_DIR = Path(os.environ.get('SAGENET_DATA_DIR', ROOT_DIR / 'data'))
ARTIFACT_DIR = ROOT_DIR / 'results' / 'generated'
CKPT_DIR = ROOT_DIR / 'checkpoints' / 'generated'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
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
    """
    Dataset for JAAD and PIE with resolution coordinate normalization and Random Over-Sampling.
    """
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
            # Normalize spatial box coordinates and velocities to [0, 1]
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
        oversampled_pos = [rng.choice(pos) for _ in range(target_pos_count)]

        combined = list(neg) + oversampled_pos
        rng.shuffle(combined)
        return combined

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, idx):
        s = self.processed_samples[idx]
        return {
            'V_M': s['V_M'],
            'V_S': s['V_S'],
            'V_I': s['V_I'],
            'K_M': s['K_M'],
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


def train_and_eval_dataset(dataset_name, cache_filename, manifest_filename, epochs=12, batch_size=32, lr=1e-4):
    print("\n" + "=" * 80)
    print(f"  RUNNING SAGE-NET MULTI-SEED EXPERIMENTS ON {dataset_name}")
    print("=" * 80)
    sys.stdout.flush()

    cache_path = BASE_DIR / cache_filename
    all_data = torch.load(cache_path, map_location='cpu', weights_only=False)

    manifest_path = ROOT_DIR / 'datasets' / 'splits' / manifest_filename

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    split_key = 'frozen_unseen_split' if 'frozen_unseen_split' in manifest else 'official_split'
    train_vids = set(manifest[split_key]['train_videos'])
    val_vids   = set(manifest[split_key]['val_videos'])
    test_vids  = set(manifest[split_key]['test_videos'])

    train_samples = [s for s in all_data if s['video_id'] in train_vids]
    val_samples   = [s for s in all_data if s['video_id'] in val_vids]
    test_samples  = [s for s in all_data if s['video_id'] in test_vids]

    print(f"[{dataset_name}] Partition -> Train: {len(train_samples)} seqs (Pos: {sum(s['label']==1 for s in train_samples)}), "
          f"Val: {len(val_samples)} seqs (Pos: {sum(s['label']==1 for s in val_samples)}), "
          f"Test: {len(test_samples)} seqs (Pos: {sum(s['label']==1 for s in test_samples)})")
    sys.stdout.flush()

    seeds = [42, 101, 2024]
    trained_models = {}
    focal_crit = FocalLoss(alpha=0.75, gamma=2.0)
    bce_crit = nn.BCEWithLogitsLoss()

    val_y_arr = np.array([int(s['label']) for s in val_samples])
    test_y_arr = np.array([int(s['label']) for s in test_samples])

    # 1. Multi-Seed Training with Random Over-Sampling
    for seed in seeds:
        print(f"\n[{dataset_name}] Training SAGE-Net Seed {seed} ({epochs} Epochs with ROS)...")
        sys.stdout.flush()
        t_start = time.time()
        seed_everything(seed)

        train_ds = SAGEResolutionDataset(train_samples, oversample_positives=True, target_ratio=1.0, seed=seed)
        val_ds   = SAGEResolutionDataset(val_samples, oversample_positives=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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
                vm = b['V_M'].to(DEVICE)
                vs = b['V_S'].to(DEVICE)
                vi = b['V_I'].to(DEVICE)
                km = b['K_M'].to(DEVICE)
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

            # Validation evaluation
            model.eval()
            val_probs = []
            with torch.no_grad():
                for b in val_loader:
                    vm = b['V_M'].to(DEVICE); vs = b['V_S'].to(DEVICE); vi = b['V_I'].to(DEVICE); km = b['K_M'].to(DEVICE)
                    logits, _ = model(vm, vs, vi, km)
                    val_probs.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())

            val_auc = roc_auc_score(val_y_arr, val_probs) if len(np.unique(val_y_arr)) > 1 else 0.5
            if ep % 4 == 0 or ep == epochs:
                print(f"    [{dataset_name} Seed {seed}] Ep {ep:2d}/{epochs:2d} | Train Loss: {ep_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
                sys.stdout.flush()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_ep = ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        ckpt_path = os.path.join(CKPT_DIR, f'sagenet_{dataset_name.lower()}_seed{seed}.pth')
        torch.save(best_state, ckpt_path)
        trained_models[seed] = model
        print(f"  [{dataset_name}] Seed {seed} Finished in {time.time()-t_start:.1f}s | Best Ep: {best_ep} | Best Val AUC: {best_val_auc:.4f} | Saved: {ckpt_path}")
        sys.stdout.flush()

    # 2. Benchmark Evaluation on Held-Out Test Sequences
    print(f"\n[{dataset_name}] Evaluating 4 Matched Variants on Held-Out Test Set ({len(test_samples)} Windows)...")
    sys.stdout.flush()

    variant_configs = {
        '1_kinematics_only': {
            'name': 'Kinematics Only (K_M)',
            'kwargs': {'disable_vision': True, 'disable_kinematics': False, 'fixed_uniform_fusion': False}
        },
        '2_visual_only': {
            'name': 'Visual Streams Only (V_M, V_S, V_I)',
            'kwargs': {'disable_vision': False, 'disable_kinematics': True, 'fixed_uniform_fusion': False}
        },
        '3_fixed_uniform_fusion': {
            'name': 'Fixed Uniform Fusion (50/50)',
            'kwargs': {'disable_vision': False, 'disable_kinematics': False, 'fixed_uniform_fusion': True}
        },
        '4_sage_net_adaptive': {
            'name': 'SAGE-Net (Scene-Adaptive Gate)',
            'kwargs': {'disable_vision': False, 'disable_kinematics': False, 'fixed_uniform_fusion': False}
        }
    }

    test_ds = SAGEResolutionDataset(test_samples, oversample_positives=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    results_dict = {}
    variant_test_probs = {}

    for v_key, v_info in variant_configs.items():
        v_name = v_info['name']
        v_kw = v_info['kwargs']

        seed_probs = []
        seed_metrics_star = []
        seed_metrics_05   = []
        tau_stars = []

        for seed in seeds:
            model = trained_models[seed]
            model.eval()

            # Calibrate tau* on validation
            val_ds = SAGEResolutionDataset(val_samples, oversample_positives=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            val_p = []
            with torch.no_grad():
                for b in val_loader:
                    vm = b['V_M'].to(DEVICE); vs = b['V_S'].to(DEVICE); vi = b['V_I'].to(DEVICE); km = b['K_M'].to(DEVICE)
                    logits, _ = model(vm, vs, vi, km, **v_kw)
                    val_p.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
            tau_star = calibrate_tau_on_val(val_y_arr, np.array(val_p))
            tau_stars.append(tau_star)

            # Test evaluation
            test_p = []
            with torch.no_grad():
                for b in test_loader:
                    vm = b['V_M'].to(DEVICE); vs = b['V_S'].to(DEVICE); vi = b['V_I'].to(DEVICE); km = b['K_M'].to(DEVICE)
                    logits, _ = model(vm, vs, vi, km, **v_kw)
                    test_p.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
            test_p = np.array(test_p)
            seed_probs.append(test_p)

            seed_metrics_star.append(eval_metrics(test_y_arr, test_p, tau=tau_star))
            seed_metrics_05.append(eval_metrics(test_y_arr, test_p, tau=0.50))

        seed_probs = np.array(seed_probs)
        ens_probs = np.mean(seed_probs, axis=0)
        variant_test_probs[v_key] = ens_probs

        ens_tau_star = float(np.mean(tau_stars))
        ens_metrics_star = eval_metrics(test_y_arr, ens_probs, tau=ens_tau_star)
        ens_metrics_05   = eval_metrics(test_y_arr, ens_probs, tau=0.50)

        aucs = [m['ROC_AUC'] for m in seed_metrics_star]
        aps  = [m['PR_AUC'] for m in seed_metrics_star]
        f1s_star = [m['F1_Score'] for m in seed_metrics_star]
        accs_star = [m['Accuracy'] for m in seed_metrics_star]
        accs_05 = [m['Accuracy'] for m in seed_metrics_05]

        results_dict[v_key] = {
            'name': v_name,
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'mean_ap': float(np.mean(aps)),
            'std_ap': float(np.std(aps)),
            'mean_f1_star': float(np.mean(f1s_star)),
            'mean_acc_star': float(np.mean(accs_star)),
            'mean_acc_05': float(np.mean(accs_05)),
            'seed_aucs': {str(s): float(a) for s, a in zip(seeds, aucs)},
            'ensemble_auc': float(ens_metrics_star['ROC_AUC']),
            'ensemble_ap': float(ens_metrics_star['PR_AUC']),
            'ensemble_f1_star': float(ens_metrics_star['F1_Score']),
            'ensemble_acc_star': float(ens_metrics_star['Accuracy']),
            'ensemble_acc_05': float(ens_metrics_05['Accuracy']),
            'tau_star': ens_tau_star
        }
        print(f"  [{dataset_name} - {v_name}] 3-Seed Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} | Ens AUC: {ens_metrics_star['ROC_AUC']:.4f} | Acc: {ens_metrics_star['Accuracy']*100:.2f}%")
        sys.stdout.flush()

    # Extract Mechanistic Gating & Attention Tensors on Test Set
    print(f"\n[{dataset_name}] Extracting Empirical Gating Weights & Temporal Attention...")
    gate_weights_list, illum_scores_list, alpha_km_list, alpha_vm_list = [], [], [], []
    with torch.no_grad():
        for b in test_loader:
            vm = b['V_M'].to(DEVICE); vs = b['V_S'].to(DEVICE); vi = b['V_I'].to(DEVICE); km = b['K_M'].to(DEVICE)
            _, aux = trained_models[seeds[0]](vm, vs, vi, km)
            gate_weights_list.extend(aux['gate_weights'][:, 0].cpu().numpy())
            illum_scores_list.extend(aux['g_illum'].squeeze(-1).cpu().numpy())
            alpha_km_list.extend(aux['alpha_kinem_motion'].cpu().numpy())
            alpha_vm_list.extend(aux['alpha_vis_motion'].cpu().numpy())

    emp_tensors = {
        'w_kinem': np.array(gate_weights_list),
        'g_illum': np.array(illum_scores_list),
        'alpha_km': np.array(alpha_km_list),
        'alpha_vm': np.array(alpha_vm_list),
        'y_true': test_y_arr,
        'probs': variant_test_probs['4_sage_net_adaptive'],
        'bbox_heights': np.array([s['K_M'][-1, 3].item() for s in test_samples])
    }

    # 3. Generate Publication-Grade Figures, Heatmaps, and ROC/PR Curves
    print(f"\n[{dataset_name}] Generating Publication-Grade Figures & Heatmaps...")
    generate_dataset_figures(dataset_name, results_dict, variant_test_probs, test_y_arr, emp_tensors)

    # Save JSON results
    out_json = os.path.join(ARTIFACT_DIR, f'results_sagenet_{dataset_name.lower()}.json')
    with open(out_json, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"[{dataset_name}] Results saved to: {out_json}")
    sys.stdout.flush()

    return results_dict


def generate_dataset_figures(dataset_name, results_dict, variant_test_probs, test_y_arr, emp_tensors):
    """
    Renders ROC, PR, Accuracy Bar, and 4-Panel Gating Heatmaps for a given dataset.
    """
    # 1. Standalone ROC Curves
    fig_roc, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    fig_roc.patch.set_facecolor('#ffffff')

    var_styles = {
        '1_kinematics_only': ('Kinematics Only (K_M)', '#0284c7', '-.'),
        '2_visual_only': ('Visual Streams Only', '#f97316', ':'),
        '3_fixed_uniform_fusion': ('Fixed Uniform Fusion (50/50)', '#8b5cf6', '--'),
        '4_sage_net_adaptive': ('SAGE-Net (Scene-Adaptive)', '#16a34a', '-')
    }

    for vk, (label_name, color, ls) in var_styles.items():
        fpr, tpr, _ = roc_curve(test_y_arr, variant_test_probs[vk])
        res = results_dict[vk]
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2.4,
                label=f"{label_name} [Mean: {res['mean_auc']:.3f}±{res['std_auc']:.3f} | Ens: {res['ensemble_auc']:.3f}]")

    ax.plot([0, 1], [0, 1], color='#cbd5e1', linestyle=':', linewidth=1.2, label='Chance (AUC = 0.500)')
    ax.set_title(f"{dataset_name} Benchmark: SAGE-Net ROC Curves", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight='bold')
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=9.0, framealpha=0.95, edgecolor='#cbd5e1')
    fig_roc.savefig(os.path.join(ARTIFACT_DIR, f'fig_{dataset_name.lower()}_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_roc)

    # 2. Standalone PR Curves
    fig_pr, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    fig_pr.patch.set_facecolor('#ffffff')
    prev = float(np.mean(test_y_arr))

    for vk, (label_name, color, ls) in var_styles.items():
        prec, rec, _ = precision_recall_curve(test_y_arr, variant_test_probs[vk])
        res = results_dict[vk]
        ax.plot(rec, prec, color=color, linestyle=ls, linewidth=2.4,
                label=f"{label_name} [AP: {res['mean_ap']:.3f} | Ens: {res['ensemble_ap']:.3f}]")

    ax.axhline(prev, color='#ef4444', linestyle=':', linewidth=1.2, label=f'Crossing Prevalence ({prev*100:.1f}%)')
    ax.set_title(f"{dataset_name} Benchmark: SAGE-Net Precision-Recall Curves", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight='bold')
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=9.0, framealpha=0.95, edgecolor='#cbd5e1')
    fig_pr.savefig(os.path.join(ARTIFACT_DIR, f'fig_{dataset_name.lower()}_pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_pr)

    # 3. Accuracy Bar Chart
    fig_acc, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    fig_acc.patch.set_facecolor('#ffffff')

    bar_labels = ['Kinematics\nOnly', 'Visual\nOnly', 'Fixed 50/50\nFusion', 'SAGE-Net\n(Ours @ tau*)', 'SAGE-Net\n(Ours @ 0.50)']
    bar_vals = [
        results_dict['1_kinematics_only']['mean_acc_star'] * 100,
        results_dict['2_visual_only']['mean_acc_star'] * 100,
        results_dict['3_fixed_uniform_fusion']['mean_acc_star'] * 100,
        results_dict['4_sage_net_adaptive']['mean_acc_star'] * 100,
        results_dict['4_sage_net_adaptive']['mean_acc_05'] * 100
    ]
    bar_colors = ['#0284c7', '#f97316', '#8b5cf6', '#16a34a', '#15803d']

    bars = ax.bar(range(len(bar_vals)), bar_vals, color=bar_colors, width=0.55, edgecolor='#0f172a', linewidth=1.2)
    ax.set_xticks(range(len(bar_vals)))
    ax.set_xticklabels(bar_labels, fontsize=10, fontweight='bold')
    ax.set_ylabel("Classification Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_ylim([max(0, min(bar_vals) - 15), 100])
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_title(f"{dataset_name} Benchmark: Classification Accuracy Across Architecture Variants", fontsize=12.5, fontweight='bold', pad=12)

    for bar, val in zip(bars, bar_vals):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{val:.2f}%",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#0f172a')

    fig_acc.savefig(os.path.join(ARTIFACT_DIR, f'fig_{dataset_name.lower()}_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_acc)

    # 4. 4-Panel Gating & Attention Heatmaps
    fig_hm = plt.figure(figsize=(16, 12), dpi=300)
    fig_hm.patch.set_facecolor('#ffffff')
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28, left=0.08, right=0.94, top=0.92, bottom=0.08)

    fig_hm.suptitle(
        f"SAGE-Net Adaptive Gating & Multi-Modal Dynamics on {dataset_name}\n"
        f"Empirical Modality Weighting, Illumination Adaptation, and Temporal Focus Profiles",
        fontsize=14, fontweight='bold', color='#0f172a'
    )

    # Panel A: Heatmap of Kinematic Weight w_K by Illumination Score & BBox Scale
    ax_a = fig_hm.add_subplot(gs[0, 0])
    illum_bins = np.linspace(0.0, 1.0, 5)
    scale_bins = np.linspace(0.0, max(0.4, float(np.percentile(emp_tensors['bbox_heights'], 95))), 5)

    grid_wk = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            m_ij = (emp_tensors['g_illum'] >= illum_bins[i]) & (emp_tensors['g_illum'] < illum_bins[i+1]) & \
                   (emp_tensors['bbox_heights'] >= scale_bins[j]) & (emp_tensors['bbox_heights'] < scale_bins[j+1])
            if np.sum(m_ij) > 0:
                grid_wk[i, j] = np.mean(emp_tensors['w_kinem'][m_ij])
            else:
                grid_wk[i, j] = np.nan

    sns.heatmap(grid_wk, ax=ax_a, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Kinematic Weight (w_K)'},
                xticklabels=[f"{scale_bins[j]:.2f}-{scale_bins[j+1]:.2f}" for j in range(4)],
                yticklabels=[f"{illum_bins[i]:.2f}-{illum_bins[i+1]:.2f}" for i in range(4)])
    ax_a.set_title(f"(a) Adaptive Gate Weight (w_K) vs. Scene Luminance & Pedestrian Scale", fontsize=11, fontweight='bold', pad=8)
    ax_a.set_xlabel("Normalized Pedestrian Bounding Box Height", fontsize=10, fontweight='bold')
    ax_a.set_ylabel("Estimated Low-Light Illumination Score (g_illum)", fontsize=10, fontweight='bold')

    # Panel B: Temporal Attention Profiles Across 15 Pre-Crossing Frames
    ax_b = fig_hm.add_subplot(gs[0, 1])
    Seq = emp_tensors['alpha_km'].shape[1]
    t_steps = np.arange(Seq)
    ak_m = np.mean(emp_tensors['alpha_km'], axis=0)
    ak_s = np.std(emp_tensors['alpha_km'], axis=0)
    av_m = np.mean(emp_tensors['alpha_vm'], axis=0)
    av_s = np.std(emp_tensors['alpha_vm'], axis=0)

    ax_b.plot(t_steps, ak_m, color='#0284c7', marker='o', linewidth=2.2, label='Kinematic Temporal Attention (alpha_K)')
    ax_b.fill_between(t_steps, ak_m - ak_s, ak_m + ak_s, color='#0284c7', alpha=0.18)
    ax_b.plot(t_steps, av_m, color='#f97316', marker='s', linewidth=2.2, label='Visual Appearance Attention (alpha_V)')
    ax_b.fill_between(t_steps, av_m - av_s, av_m + av_s, color='#f97316', alpha=0.18)

    ax_b.set_title(f"(b) Learned Temporal Attention Focus Across {Seq} Pre-Crossing Frames", fontsize=11, fontweight='bold', pad=8)
    ax_b.set_xlabel(f"Observation Frame Index t in [0, {Seq-1}] (Approaching Horizon)", fontsize=10, fontweight='bold')
    ax_b.set_ylabel("Attention Weight alpha_t", fontsize=10, fontweight='bold')
    ax_b.grid(True, linestyle='--', alpha=0.4)
    ax_b.legend(loc='upper left', fontsize=9.2)

    # Panel C: Kinematic Routing Weight Distribution
    ax_c = fig_hm.add_subplot(gs[1, 0])
    wk_all = emp_tensors['w_kinem']
    ax_c.hist(wk_all, bins=25, color='#3b82f6', alpha=0.75, edgecolor='#1e3a8a', density=True)
    ax_c.axvline(np.mean(wk_all), color='#15803d', linestyle='-', linewidth=2.2, label=f'Mean w_K: {np.mean(wk_all):.2f}')
    ax_c.axvline(0.50, color='#ef4444', linestyle='--', linewidth=1.5, label='Fixed 50/50 Baseline')

    ax_c.set_title(f"(c) Empirical Distribution of Assigned Kinematic Weights (w_K)", fontsize=11, fontweight='bold', pad=8)
    ax_c.set_xlabel("Assigned Kinematic Weight w_K", fontsize=10, fontweight='bold')
    ax_c.set_ylabel("Density", fontsize=10, fontweight='bold')
    ax_c.grid(True, linestyle='--', alpha=0.4)
    ax_c.legend(loc='upper right', fontsize=9.0)

    # Panel D: Confusion Matrix
    ax_d = fig_hm.add_subplot(gs[1, 1])
    tau = results_dict['4_sage_net_adaptive']['tau_star']
    y_pred = (emp_tensors['probs'] >= tau).astype(int)
    cm = confusion_matrix(test_y_arr, y_pred)

    sns.heatmap(cm, ax=ax_d, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred Non-Cross (0)', 'Pred Cross (1)'],
                yticklabels=['Actual Non-Cross (0)', 'Actual Cross (1)'])
    acc_cm = float(np.mean(y_pred == test_y_arr)) * 100
    ax_d.set_title(f"(d) Confusion Matrix at tau* = {tau:.2f} (Accuracy: {acc_cm:.2f}%)", fontsize=11, fontweight='bold', pad=8)
    ax_d.set_xlabel("Predicted Crossing Label", fontsize=10, fontweight='bold')
    ax_d.set_ylabel("True Ground-Truth Crossing Label", fontsize=10, fontweight='bold')

    fig_hm.savefig(os.path.join(ARTIFACT_DIR, f'fig_{dataset_name.lower()}_gating_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_hm)
    print(f"[{dataset_name}] Figures & Heatmaps rendered successfully.")


def main():
    global BASE_DIR, ARTIFACT_DIR, CKPT_DIR
    parser = argparse.ArgumentParser(description='Train and evaluate SAGE-Net on JAAD and PIE.')
    parser.add_argument('--data-root', type=Path, default=BASE_DIR, help='Directory containing JAAD and PIE tensor caches.')
    parser.add_argument('--output-dir', type=Path, default=ARTIFACT_DIR, help='Directory for generated JSON and figures.')
    parser.add_argument('--checkpoint-dir', type=Path, default=CKPT_DIR, help='Directory for generated checkpoints.')
    args = parser.parse_args()
    BASE_DIR = args.data_root.expanduser().resolve()
    ARTIFACT_DIR = args.output_dir.expanduser().resolve()
    CKPT_DIR = args.checkpoint_dir.expanduser().resolve()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("  SAGE-NET: SCENE-ADAPTIVE GATED ENCODERS NETWORK")
    print("  MULTI-DATASET TRAINING, EVALUATION & HEATMAP PIPELINE (JAAD & PIE)")
    print(f"  Device: {DEVICE} | Platform: PyTorch")
    print("=" * 80)
    sys.stdout.flush()

    # 1. Train and evaluate on JAAD
    results_jaad = train_and_eval_dataset(
        dataset_name='JAAD',
        cache_filename='jaad_sliding_cache.pt',
        manifest_filename='split_manifest_jaad.json',
        epochs=12,
        batch_size=32,
        lr=1e-4
    )

    # 2. Train and evaluate on PIE
    results_pie = train_and_eval_dataset(
        dataset_name='PIE',
        cache_filename='pie_sliding_cache.pt',
        manifest_filename='split_manifest_pie.json',
        epochs=10,
        batch_size=48,
        lr=5e-5
    )

    # Summary Table across both datasets
    print("\n" + "=" * 90)
    print("  FINAL MULTI-DATASET SAGE-NET BENCHMARK SUMMARY (JAAD & PIE)")
    print("=" * 90)
    header = f"{'Dataset':<10} | {'Variant':<35} | {'ROC-AUC':<16} | {'PR-AUC (AP)':<12} | {'Acc (tau*)':<10} | {'F1 (tau*)':<10}"
    print(header)
    print("-" * len(header))

    for dname, rdict in [('JAAD', results_jaad), ('PIE', results_pie)]:
        for vk, res in rdict.items():
            auc_str = f"{res['mean_auc']:.4f} ± {res['std_auc']:.4f}"
            print(f"{dname:<10} | {res['name']:<35} | {auc_str:<16} | {res['mean_ap']:<12.4f} | {res['mean_acc_star']*100:<9.2f}% | {res['mean_f1_star']:<10.4f}")
        print("-" * len(header))

    print("\nSAGE-Net Multi-Dataset Pipeline Completed Successfully!")


if __name__ == '__main__':
    main()
