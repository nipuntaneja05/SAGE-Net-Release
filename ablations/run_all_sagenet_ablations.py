"""
run_all_sagenet_ablations.py
Comprehensive Ablation Suite for SAGE-Net across IDD-PeD, JAAD, and PIE benchmarks:
Evaluates 13 systematic architectural, modality, and component ablation variants:
1. Full Proposed SAGE-Net (All 6 encoders + Occl + Illum + Adaptive Gate)
2. Fixed Uniform Fusion (50/50, no adaptive gate)
3. Concat MLP Fusion (Early/late concatenation head)
4. w/o Occlusion Motion Reconstruction Block (Raw Kinematics)
5. w/o Illumination Dual-Pathway Adapter Block (Raw Scene)
6. w/o Both Robustness Blocks (Raw Kinem + Raw Scene)
7. w/o Interaction Streams (No K_I, No V_I)
8. w/o Scene Dynamics (No K_S, No V_S)
9. Motion-Only Streams (K_M + V_M)
10. Kinematics Stream Alone (K_M + K_S + K_I)
11. Kinematics Motion Alone (K_M trajectory only)
12. Visual Streams Alone (V_M + V_S + V_I)
13. Visual Pedestrian Crop Alone (V_M crop only)

Generates:
- Full metrics JSON for all 3 datasets
- Publication-grade ROC-AUC curves for all ablations
- Publication-grade PR-AUC curves for all ablations
- 2D Component Contribution / Marginal Drop Heatmap
- Cross-Dataset Comparative Performance Matrix
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

BASE_DIR = Path(os.environ.get('SAGENET_DATA_DIR', ROOT_DIR / 'data'))
ARTIFACT_DIR = ROOT_DIR / 'results' / 'generated'
SCRATCH_DIR = ARTIFACT_DIR
CKPT_DIR = ROOT_DIR / 'checkpoints' / 'generated'
os.makedirs(CKPT_DIR, exist_ok=True)

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


# ==============================================================================
# ABLATION SPECIFICATION DEFINITIONS
# ==============================================================================
ABLATION_DEFS = [
    {
        'id': '01_full_sagenet',
        'name': 'Full SAGE-Net (Adaptive Fusion)',
        'category': 'Reference',
        'desc': 'All 6 encoders + Adaptive Gate + Occlusion Block + Illumination Block',
        'color': '#0284c7',
        'linestyle': '-',
        'kwargs': {}
    },
    {
        'id': '02_fixed_uniform_5050',
        'name': 'Fixed Uniform Fusion (50/50)',
        'category': 'Fusion Strategy',
        'desc': 'Static equal weighting 0.5 * L_K + 0.5 * L_V without scene gate',
        'color': '#10b981',
        'linestyle': '--',
        'kwargs': {'fixed_uniform_fusion': True}
    },
    {
        'id': '03_concat_mlp',
        'name': 'Concatenation MLP Fusion',
        'category': 'Fusion Strategy',
        'desc': 'Joint MLP on concatenated [H_K, H_V] embeddings without gating',
        'color': '#8b5cf6',
        'linestyle': ':',
        'kwargs': {'fusion_mode': 'concat'}
    },
    {
        'id': '04_no_occlusion_block',
        'name': 'w/o Occlusion Reconstruction Block',
        'category': 'Robustness Preprocessing',
        'desc': 'Raw kinematics fed directly without trajectory denoising / inpainting',
        'color': '#f59e0b',
        'linestyle': '-.',
        'kwargs': {'disable_occlusion_block': True}
    },
    {
        'id': '05_no_illumination_block',
        'name': 'w/o Illumination Adapter Block',
        'category': 'Robustness Preprocessing',
        'desc': 'Raw global scene context fed without dual daylight/night pathways',
        'color': '#ec4899',
        'linestyle': '-.',
        'kwargs': {'disable_illum_block': True}
    },
    {
        'id': '06_no_robustness_blocks',
        'name': 'w/o Both Robustness Blocks',
        'category': 'Robustness Preprocessing',
        'desc': 'Neither Occlusion nor Illumination block active (Raw features only)',
        'color': '#ef4444',
        'linestyle': '--',
        'kwargs': {'disable_occlusion_block': True, 'disable_illum_block': True}
    },
    {
        'id': '07_no_interaction_streams',
        'name': 'w/o Interaction Encoders (No K_I, No V_I)',
        'category': 'Sub-Encoder Streams',
        'desc': 'Pedestrian motion + Scene dynamics only (spatial interaction zeroed)',
        'color': '#14b8a6',
        'linestyle': '-.',
        'kwargs': {'disable_interaction': True}
    },
    {
        'id': '08_no_scene_streams',
        'name': 'w/o Scene Encoders (No K_S, No V_S)',
        'category': 'Sub-Encoder Streams',
        'desc': 'Pedestrian motion + Interaction dynamics only (ego-speed & scene zeroed)',
        'color': '#6366f1',
        'linestyle': ':',
        'kwargs': {'disable_scene_streams': True}
    },
    {
        'id': '09_motion_only_streams',
        'name': 'Motion-Only Streams (K_M + V_M)',
        'category': 'Sub-Encoder Streams',
        'desc': 'Pedestrian kinematics + Pedestrian crop appearance only (no scene/interaction)',
        'color': '#84cc16',
        'linestyle': '--',
        'kwargs': {'motion_only': True}
    },
    {
        'id': '10_kinematics_all',
        'name': 'Kinematics Stream Alone (K_M + K_S + K_I)',
        'category': 'Modality Solos',
        'desc': 'All 3 kinematic encoders alone (vision disabled)',
        'color': '#3b82f6',
        'linestyle': '-',
        'kwargs': {'disable_vision': True}
    },
    {
        'id': '11_kinematics_motion_only',
        'name': 'Kinematics Motion Alone (K_M)',
        'category': 'Modality Solos',
        'desc': 'Bounding box coordinates, velocity, and acceleration alone',
        'color': '#06b6d4',
        'linestyle': ':',
        'kwargs': {'disable_vision': True, 'kinematics_motion_only': True}
    },
    {
        'id': '12_visual_all',
        'name': 'Visual Streams Alone (V_M + V_S + V_I)',
        'category': 'Modality Solos',
        'desc': 'All 3 visual encoders alone (kinematics disabled)',
        'color': '#f97316',
        'linestyle': '-',
        'kwargs': {'disable_kinematics': True}
    },
    {
        'id': '13_visual_crop_only',
        'name': 'Visual Pedestrian Crop Alone (V_M)',
        'category': 'Modality Solos',
        'desc': 'Pedestrian appearance crop alone (no scene, no surround)',
        'color': '#d97706',
        'linestyle': ':',
        'kwargs': {'disable_kinematics': True, 'visual_motion_only': True}
    }
]


# ==============================================================================
# EVALUATION HELPER FOR A GIVEN DATASET
# ==============================================================================
def evaluate_ablations_on_dataset(dataset_name, model, val_loader, test_loader, y_val, y_test):
    print(f"\n================================================================================")
    print(f"  RUNNING COMPLETE ABLATION SUITE ON {dataset_name} ({len(y_test)} Test Windows)")
    print(f"================================================================================")
    sys.stdout.flush()

    results = {}
    test_curves = {}

    model.eval()

    for ab in ABLATION_DEFS:
        ab_id = ab['id']
        ab_name = ab['name']
        ab_kwargs = ab['kwargs']

        print(f"  --> Evaluating [{ab_id}] {ab_name}...")
        sys.stdout.flush()

        # 1. Validation evaluation for tau* calibration
        val_probs = []
        with torch.no_grad():
            for b in val_loader:
                vm = b['V_M'].to(DEVICE)
                vs = b['V_S'].to(DEVICE)
                vi = b['V_I'].to(DEVICE)
                km = b['K_M'].to(DEVICE)
                logits, _ = model(vm, vs, vi, km, **ab_kwargs)
                val_probs.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
        val_probs = np.array(val_probs)
        tau_star = calibrate_tau_on_val(y_val, val_probs)

        # 2. Test evaluation
        test_probs = []
        with torch.no_grad():
            for b in test_loader:
                vm = b['V_M'].to(DEVICE)
                vs = b['V_S'].to(DEVICE)
                vi = b['V_I'].to(DEVICE)
                km = b['K_M'].to(DEVICE)
                logits, _ = model(vm, vs, vi, km, **ab_kwargs)
                test_probs.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
        test_probs = np.array(test_probs)

        # Metrics
        m_star = eval_metrics(y_test, test_probs, tau=tau_star)
        m_05   = eval_metrics(y_test, test_probs, tau=0.50)

        # ROC & PR curve data
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        prec, rec, _ = precision_recall_curve(y_test, test_probs)

        results[ab_id] = {
            'id': ab_id,
            'name': ab_name,
            'category': ab['category'],
            'desc': ab['desc'],
            'ROC_AUC': m_star['ROC_AUC'],
            'PR_AUC': m_star['PR_AUC'],
            'tau_star': tau_star,
            'Accuracy_tau_star': m_star['Accuracy'],
            'F1_tau_star': m_star['F1_Score'],
            'Precision_tau_star': m_star['Precision'],
            'Recall_tau_star': m_star['Recall'],
            'Accuracy_tau_05': m_05['Accuracy'],
            'F1_tau_05': m_05['F1_Score']
        }

        test_curves[ab_id] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'prec': prec.tolist(),
            'rec': rec.tolist(),
            'auc': m_star['ROC_AUC'],
            'ap': m_star['PR_AUC']
        }

    # Compute delta vs Full SAGE-Net
    ref_auc = results['01_full_sagenet']['ROC_AUC']
    ref_ap  = results['01_full_sagenet']['PR_AUC']
    for ab_id in results:
        results[ab_id]['delta_auc'] = results[ab_id]['ROC_AUC'] - ref_auc
        results[ab_id]['delta_ap']  = results[ab_id]['PR_AUC'] - ref_ap

    return results, test_curves


# ==============================================================================
# PUBLICATION FIGURE GENERATORS
# ==============================================================================
def plot_dataset_ablation_figures(dataset_name, results, curves, out_dir):
    print(f"\n[{dataset_name}] Generating Publication-Grade Ablation Visualizations...")
    sys.stdout.flush()

    palette = {ab['id']: ab['color'] for ab in ABLATION_DEFS}
    linestyles = {ab['id']: ab['linestyle'] for ab in ABLATION_DEFS}

    # 1. Comprehensive ROC-AUC Curves
    fig_roc = plt.figure(figsize=(10, 8), dpi=300)
    fig_roc.patch.set_facecolor('#ffffff')
    ax_roc = fig_roc.add_subplot(1, 1, 1)

    # Plot diagonal
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Chance (AUC = 0.500)')

    # Sort curves by AUC for clean legend
    sorted_abs = sorted(ABLATION_DEFS, key=lambda a: results[a['id']]['ROC_AUC'], reverse=True)

    for ab in sorted_abs:
        ab_id = ab['id']
        c = curves[ab_id]
        fpr = np.array(c['fpr'])
        tpr = np.array(c['tpr'])
        auc_val = results[ab_id]['ROC_AUC']
        lw = 2.4 if ab_id == '01_full_sagenet' else 1.6
        label = f"{ab['name']} (AUC: {auc_val:.4f})"
        ax_roc.plot(fpr, tpr, color=palette[ab_id], linestyle=linestyles[ab_id], linewidth=lw, label=label)

    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.02])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold', labelpad=8)
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold', labelpad=8)
    ax_roc.set_title(f'SAGE-Net Full Architectural Ablation Suite — ROC Curves ({dataset_name})',
                     fontsize=13, fontweight='bold', pad=12, color='#0f172a')
    ax_roc.grid(True, linestyle=':', alpha=0.6)
    ax_roc.legend(loc='lower right', fontsize=8.2, framealpha=0.92)

    # Inset zoom on low FPR region [0, 0.25]
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_ins = inset_axes(ax_roc, width="40%", height="36%", loc='center right', borderpad=2)
    for ab in sorted_abs:
        ab_id = ab['id']
        c = curves[ab_id]
        fpr = np.array(c['fpr'])
        tpr = np.array(c['tpr'])
        lw = 2.0 if ab_id == '01_full_sagenet' else 1.2
        ax_ins.plot(fpr, tpr, color=palette[ab_id], linestyle=linestyles[ab_id], linewidth=lw)
    ax_ins.set_xlim([0.0, 0.25])
    ax_ins.set_ylim([0.40, 1.0])
    ax_ins.set_title('Zoom: FPR [0, 0.25]', fontsize=8, fontweight='bold')
    ax_ins.grid(True, linestyle=':', alpha=0.5)

    roc_path = os.path.join(out_dir, f'fig_ablations_roc_curves_{dataset_name.lower()}.png')
    fig_roc.savefig(roc_path, bbox_inches='tight', dpi=300)
    plt.close(fig_roc)
    print(f"  Saved: {roc_path}")

    # 2. Comprehensive Precision-Recall Curves
    fig_pr = plt.figure(figsize=(10, 8), dpi=300)
    fig_pr.patch.set_facecolor('#ffffff')
    ax_pr = fig_pr.add_subplot(1, 1, 1)

    for ab in sorted_abs:
        ab_id = ab['id']
        c = curves[ab_id]
        rec = np.array(c['rec'])
        prec = np.array(c['prec'])
        ap_val = results[ab_id]['PR_AUC']
        lw = 2.4 if ab_id == '01_full_sagenet' else 1.6
        label = f"{ab['name']} (AP: {ap_val:.4f})"
        ax_pr.plot(rec, prec, color=palette[ab_id], linestyle=linestyles[ab_id], linewidth=lw, label=label)

    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.02])
    ax_pr.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold', labelpad=8)
    ax_pr.set_ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold', labelpad=8)
    ax_pr.set_title(f'SAGE-Net Architectural Ablation Suite — Precision-Recall Curves ({dataset_name})',
                    fontsize=13, fontweight='bold', pad=12, color='#0f172a')
    ax_pr.grid(True, linestyle=':', alpha=0.6)
    ax_pr.legend(loc='upper right', fontsize=8.2, framealpha=0.92)

    pr_path = os.path.join(out_dir, f'fig_ablations_pr_curves_{dataset_name.lower()}.png')
    fig_pr.savefig(pr_path, bbox_inches='tight', dpi=300)
    plt.close(fig_pr)
    print(f"  Saved: {pr_path}")


def plot_cross_dataset_ablation_heatmaps(unified_results, out_dir):
    print("\nGenerating Cross-Dataset Ablation Heatmap and Marginal Contribution Chart...")
    sys.stdout.flush()

    datasets = ['IDD-PeD', 'JAAD', 'PIE']
    ab_labels = [ab['name'] for ab in ABLATION_DEFS]
    ab_ids = [ab['id'] for ab in ABLATION_DEFS]

    # Matrix 1: Absolute ROC-AUC across datasets
    auc_matrix = np.zeros((len(ab_ids), len(datasets)))
    delta_matrix = np.zeros((len(ab_ids), len(datasets)))

    for j, d in enumerate(datasets):
        d_res = unified_results[d]
        ref_auc = d_res['01_full_sagenet']['ROC_AUC']
        for i, ab_id in enumerate(ab_ids):
            val = d_res[ab_id]['ROC_AUC']
            auc_matrix[i, j] = val
            delta_matrix[i, j] = val - ref_auc

    # Heatmap 1: Absolute AUC
    fig_matrix = plt.figure(figsize=(12, 10), dpi=300)
    fig_matrix.patch.set_facecolor('#ffffff')
    ax_m = fig_matrix.add_subplot(1, 1, 1)

    sns.heatmap(
        auc_matrix,
        ax=ax_m,
        cmap='YlGnBu',
        annot=True,
        fmt='.4f',
        cbar_kws={'label': 'Test ROC-AUC'},
        xticklabels=datasets,
        yticklabels=ab_labels,
        linewidths=0.8,
        linecolor='#e2e8f0'
    )
    ax_m.set_title("SAGE-Net Cross-Dataset Architectural Ablation Matrix (ROC-AUC)\n"
                   "Performance Comparison Across IDD-PeD, JAAD, and PIE Held-Out Benchmarks",
                   fontsize=13, fontweight='bold', pad=14, color='#0f172a')
    ax_m.set_xlabel("Benchmark Dataset", fontsize=11, fontweight='bold', labelpad=8)
    ax_m.set_ylabel("Architecture / Ablation Variant", fontsize=11, fontweight='bold', labelpad=8)

    matrix_path = os.path.join(out_dir, 'fig_ablations_cross_dataset_matrix.png')
    fig_matrix.savefig(matrix_path, bbox_inches='tight', dpi=300)
    plt.close(fig_matrix)
    print(f"  Saved: {matrix_path}")

    # Heatmap 2: Marginal Performance Drop Delta-AUC
    fig_delta = plt.figure(figsize=(12, 10), dpi=300)
    fig_delta.patch.set_facecolor('#ffffff')
    ax_d = fig_delta.add_subplot(1, 1, 1)

    # Use diverging RdBu colormap (negative is red/drop, zero is neutral)
    sns.heatmap(
        delta_matrix,
        ax=ax_d,
        cmap='RdYlGn',
        annot=True,
        fmt='+.4f',
        center=0.0,
        cbar_kws={'label': 'Delta ROC-AUC vs Full SAGE-Net'},
        xticklabels=datasets,
        yticklabels=ab_labels,
        linewidths=0.8,
        linecolor='#e2e8f0'
    )
    ax_d.set_title("SAGE-Net Component Marginal Contribution Heatmap (Delta ROC-AUC)\n"
                   "Quantifying Degradation When Ablating Key Architectural Blocks and Modalities",
                   fontsize=13, fontweight='bold', pad=14, color='#0f172a')
    ax_d.set_xlabel("Benchmark Dataset", fontsize=11, fontweight='bold', labelpad=8)
    ax_d.set_ylabel("Ablation Variant", fontsize=11, fontweight='bold', labelpad=8)

    delta_path = os.path.join(out_dir, 'fig_ablations_component_drop_heatmap.png')
    fig_delta.savefig(delta_path, bbox_inches='tight', dpi=300)
    plt.close(fig_delta)
    print(f"  Saved: {delta_path}")


# ==============================================================================
# MAIN DRIVER
# ==============================================================================
def main():
    global BASE_DIR, ARTIFACT_DIR, SCRATCH_DIR, CKPT_DIR
    parser = argparse.ArgumentParser(description='Run the SAGE-Net ablation evaluation suite.')
    parser.add_argument('--data-root', type=Path, default=BASE_DIR, help='Directory containing precomputed tensor caches.')
    parser.add_argument('--output-dir', type=Path, default=ARTIFACT_DIR, help='Directory for generated JSON and figures.')
    parser.add_argument('--checkpoint-dir', type=Path, default=CKPT_DIR, help='Directory containing trained checkpoints.')
    args = parser.parse_args()
    BASE_DIR = args.data_root.expanduser().resolve()
    ARTIFACT_DIR = args.output_dir.expanduser().resolve()
    SCRATCH_DIR = ARTIFACT_DIR
    CKPT_DIR = args.checkpoint_dir.expanduser().resolve()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("  SAGE-NET DEFINITIVE COMPREHENSIVE ABLATION BENCHMARK RUNNER")
    print("  Datasets: IDD-PeD, JAAD, PIE | 13 Systematic Ablation Variants")
    print(f"  Device: {DEVICE} | Platform: PyTorch")
    print("=" * 80)
    sys.stdout.flush()

    unified_results = {}
    unified_curves = {}

    # --------------------------------------------------------------------------
    # 1. EVALUATE JAAD ABLATIONS
    # --------------------------------------------------------------------------
    print("\n>>> LOADING AND PREPARING JAAD DATASET...")
    jaad_cache = torch.load(BASE_DIR / 'jaad_sliding_cache.pt', map_location='cpu', weights_only=False)
    with open(ROOT_DIR / 'datasets' / 'splits' / 'split_manifest_jaad.json', 'r') as f:
        manifest_jaad = json.load(f)
    sinfo_jaad = manifest_jaad.get('official_split', manifest_jaad.get('frozen_unseen_split'))
    val_vids_j = set(sinfo_jaad['val_videos'])
    test_vids_j = set(sinfo_jaad['test_videos'])

    val_s_jaad = [d for d in jaad_cache if d['video_id'] in val_vids_j]
    test_s_jaad = [d for d in jaad_cache if d['video_id'] in test_vids_j]

    val_ds_j = SAGEResolutionDataset(val_s_jaad, oversample_positives=False)
    test_ds_j = SAGEResolutionDataset(test_s_jaad, oversample_positives=False)

    val_loader_j = DataLoader(val_ds_j, batch_size=32, shuffle=False)
    test_loader_j = DataLoader(test_ds_j, batch_size=32, shuffle=False)

    y_val_j = np.array([s['label'] for s in val_ds_j.processed_samples])
    y_test_j = np.array([s['label'] for s in test_ds_j.processed_samples])

    model_jaad = AdaptiveFusionMasterModel(use_concat_head=True).to(DEVICE)
    ckpt_jaad = torch.load(CKPT_DIR / 'sagenet_jaad_seed42.pth', map_location=DEVICE, weights_only=False)
    model_jaad.load_state_dict(ckpt_jaad, strict=False)

    res_jaad, curves_jaad = evaluate_ablations_on_dataset('JAAD', model_jaad, val_loader_j, test_loader_j, y_val_j, y_test_j)
    unified_results['JAAD'] = res_jaad
    unified_curves['JAAD'] = curves_jaad
    plot_dataset_ablation_figures('JAAD', res_jaad, curves_jaad, ARTIFACT_DIR)

    with open(os.path.join(ARTIFACT_DIR, 'results_sagenet_ablations_jaad.json'), 'w') as f:
        json.dump(res_jaad, f, indent=2)

    # --------------------------------------------------------------------------
    # 2. EVALUATE PIE ABLATIONS
    # --------------------------------------------------------------------------
    print("\n>>> LOADING AND PREPARING PIE DATASET...")
    pie_cache = torch.load(BASE_DIR / 'pie_sliding_cache.pt', map_location='cpu', weights_only=False)
    with open(ROOT_DIR / 'datasets' / 'splits' / 'split_manifest_pie.json', 'r') as f:
        manifest_pie = json.load(f)
    sinfo_pie = manifest_pie.get('official_split', manifest_pie.get('frozen_unseen_split'))
    val_vids_p = set(sinfo_pie['val_videos'])
    test_vids_p = set(sinfo_pie['test_videos'])

    val_s_pie = [d for d in pie_cache if d['video_id'] in val_vids_p]
    test_s_pie = [d for d in pie_cache if d['video_id'] in test_vids_p]

    val_ds_p = SAGEResolutionDataset(val_s_pie, oversample_positives=False)
    test_ds_p = SAGEResolutionDataset(test_s_pie, oversample_positives=False)

    val_loader_p = DataLoader(val_ds_p, batch_size=48, shuffle=False)
    test_loader_p = DataLoader(test_ds_p, batch_size=48, shuffle=False)

    y_val_p = np.array([s['label'] for s in val_ds_p.processed_samples])
    y_test_p = np.array([s['label'] for s in test_ds_p.processed_samples])

    model_pie = AdaptiveFusionMasterModel(use_concat_head=True).to(DEVICE)
    ckpt_pie = torch.load(CKPT_DIR / 'sagenet_pie_seed42.pth', map_location=DEVICE, weights_only=False)
    model_pie.load_state_dict(ckpt_pie, strict=False)

    res_pie, curves_pie = evaluate_ablations_on_dataset('PIE', model_pie, val_loader_p, test_loader_p, y_val_p, y_test_p)
    unified_results['PIE'] = res_pie
    unified_curves['PIE'] = curves_pie
    plot_dataset_ablation_figures('PIE', res_pie, curves_pie, ARTIFACT_DIR)

    with open(os.path.join(ARTIFACT_DIR, 'results_sagenet_ablations_pie.json'), 'w') as f:
        json.dump(res_pie, f, indent=2)

    # --------------------------------------------------------------------------
    # 3. EVALUATE IDD-PED ABLATIONS
    # --------------------------------------------------------------------------
    print("\n>>> LOADING AND PREPARING IDD-PED DATASET...")
    idd_cache = torch.load(BASE_DIR / 'iddped_anticipation_cache.pt', map_location='cpu', weights_only=False)
    with open(ROOT_DIR / 'datasets' / 'splits' / 'split_manifest_idd_ped.json', 'r') as f:
        manifest_idd = json.load(f)
    sinfo_idd = manifest_idd.get('official_split', manifest_idd.get('frozen_unseen_split'))
    val_vids_i = set(sinfo_idd['val_videos'])
    val_s_idd = [d for d in idd_cache if d['video_id'] in val_vids_i]

    val_ds_i = IDDPedAnticipationDataset(val_s_idd, oversample_positives=False)
    val_loader_i = DataLoader(val_ds_i, batch_size=64, shuffle=False)
    y_val_i = np.array([int(s['label']) for s in val_s_idd])

    test_cached = torch.load(BASE_DIR / 'iddped_official_2598_test_tensors.pt', map_location='cpu', weights_only=False)
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
    test_loader_i = DataLoader(test_ds_i, batch_size=64, shuffle=False)
    y_test_i = test_cached['targets']

    model_idd = AdaptiveFusionMasterModel(use_concat_head=True).to(DEVICE)
    ckpt_idd = torch.load(CKPT_DIR / 'adaptive_fusion_seed42.pth', map_location=DEVICE, weights_only=False)
    model_idd.load_state_dict(ckpt_idd, strict=False)

    res_idd, curves_idd = evaluate_ablations_on_dataset('IDD-PeD', model_idd, val_loader_i, test_loader_i, y_val_i, y_test_i)
    unified_results['IDD-PeD'] = res_idd
    unified_curves['IDD-PeD'] = curves_idd
    plot_dataset_ablation_figures('IDD-PeD', res_idd, curves_idd, ARTIFACT_DIR)

    with open(os.path.join(ARTIFACT_DIR, 'results_sagenet_ablations_iddped.json'), 'w') as f:
        json.dump(res_idd, f, indent=2)

    # --------------------------------------------------------------------------
    # 4. CROSS-DATASET MATRIX & DELTA HEATMAP
    # --------------------------------------------------------------------------
    plot_cross_dataset_ablation_heatmaps(unified_results, ARTIFACT_DIR)

    with open(os.path.join(ARTIFACT_DIR, 'results_sagenet_ablations_unified.json'), 'w') as f:
        json.dump(unified_results, f, indent=2)

    # --------------------------------------------------------------------------
    # 5. FINAL TERMINAL REPORT TABLE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("  FINAL CROSS-DATASET SAGE-NET ABLATION BENCHMARK SUMMARY")
    print("=" * 115)
    header = f"{'Dataset':<8} | {'Variant Name':<42} | {'ROC-AUC':<9} | {'Delta AUC':<10} | {'PR-AUC':<8} | {'Acc (tau*)':<10} | {'F1 (tau*)':<9}"
    print(header)
    print("-" * len(header))

    for d in ['IDD-PeD', 'JAAD', 'PIE']:
        d_res = unified_results[d]
        for ab in ABLATION_DEFS:
            ab_id = ab['id']
            r = d_res[ab_id]
            d_str = f"{r['delta_auc']:+.4f}" if ab_id != '01_full_sagenet' else "Ref."
            print(f"{d:<8} | {r['name']:<42} | {r['ROC_AUC']:.4f}   | {d_str:<10} | {r['PR_AUC']:.4f}   | {r['Accuracy_tau_star']*100:6.2f}%    | {r['F1_tau_star']:.4f}")
        print("-" * len(header))

    print("\nComplete Ablation Suite Executed and Visualized Successfully!")


if __name__ == '__main__':
    main()
