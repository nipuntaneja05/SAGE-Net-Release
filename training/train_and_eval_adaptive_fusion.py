"""
train_and_eval_adaptive_fusion.py
Definitive 5-Seed Training and Matched Evaluation Suite for the Adaptive Scene-Gated
Multimodal Architecture on the Official IDD-PeD 2,598 Benchmark:
- 5 Independent Seeds: [42, 101, 2024, 7, 999]
- Random Over-Sampling (ROS) on training set for 1:1 class balance
- Multi-Task Loss: Focal(L_final) + 0.3*BCE(L_K) + 0.2*BCE(L_V) + 0.1*MSE(Recon)
- Validation Threshold Calibration (tau* via Youden's J)
- Evaluation on exact official 2,598 test windows (432 pedestrians, 270 positives)
- Comparison against published PCPA baseline (0.7113 AUC)
- Cluster Bootstrap Uncertainty Analysis (B=1,000 pedestrian clusters)
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, roc_curve
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from models.adaptive_fusion_model import AdaptiveFusionMasterModel
from datasets.iddped_anticipation_dataset import IDDPedAnticipationDataset

BASE_DIR = Path(os.environ.get('SAGENET_DATA_DIR', ROOT_DIR / 'data'))
SCRATCH_DIR = ROOT_DIR / 'results' / 'generated'
CKPT_DIR = ROOT_DIR / 'checkpoints' / 'generated'
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
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


def cluster_bootstrap_test(y_true, y_prob, clusters, baseline_auc=0.7113, B=1000, seed=42):
    rng = np.random.RandomState(seed)
    unique_clusters = np.unique(clusters)
    N_clusters = len(unique_clusters)
    cluster_to_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}

    obs_auc = roc_auc_score(y_true, y_prob)
    diff_obs = obs_auc - baseline_auc

    boot_diffs = []
    for _ in range(B):
        sample_clusters = rng.choice(unique_clusters, size=N_clusters, replace=True)
        sample_indices = np.concatenate([cluster_to_indices[c] for c in sample_clusters])
        y_b = y_true[sample_indices]
        p_b = y_prob[sample_indices]
        if len(np.unique(y_b)) < 2:
            continue
        auc_b = roc_auc_score(y_b, p_b)
        boot_diffs.append(auc_b - baseline_auc)

    boot_diffs = np.array(boot_diffs)
    se = float(np.std(boot_diffs))
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    p_value = float(np.mean(boot_diffs <= 0.0))

    return {
        'obs_auc': float(obs_auc),
        'baseline_auc': float(baseline_auc),
        'delta_obs': float(diff_obs),
        'bootstrap_se': se,
        'ci_95': [ci_lower, ci_upper],
        'p_value': p_value
    }


def main():
    global BASE_DIR, SCRATCH_DIR, CKPT_DIR
    parser = argparse.ArgumentParser(description='Run the five-seed IDD-PeD experiment.')
    parser.add_argument('--data-root', type=Path, default=BASE_DIR, help='Directory containing IDD-PeD tensor caches.')
    parser.add_argument('--output-dir', type=Path, default=SCRATCH_DIR, help='Directory for generated results.')
    parser.add_argument('--checkpoint-dir', type=Path, default=CKPT_DIR, help='Directory for generated checkpoints.')
    args = parser.parse_args()
    BASE_DIR = args.data_root.expanduser().resolve()
    SCRATCH_DIR = args.output_dir.expanduser().resolve()
    CKPT_DIR = args.checkpoint_dir.expanduser().resolve()
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("  CANONICAL 5-SEED TRAINING & MATCHED BENCHMARK EVALUATION")
    print("  ARCHITECTURE: 6-Encoder Multimodal with Scene-Adaptive Fusion Gate")
    print(f"  Device: {DEVICE} | Platform: PyTorch")
    print("=" * 80)
    sys.stdout.flush()

    # 1. Load Data & Manifest
    cache_path = BASE_DIR / 'iddped_anticipation_cache.pt'
    all_data = torch.load(cache_path, map_location='cpu', weights_only=False)

    manifest_path = ROOT_DIR / 'datasets' / 'splits' / 'split_manifest_idd_ped.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    split_info = manifest.get('official_split', manifest.get('frozen_unseen_split'))
    train_vids = set(split_info['train_videos'])
    val_vids   = set(split_info['val_videos'])

    train_samples = [d for d in all_data if d['video_id'] in train_vids]
    val_samples   = [d for d in all_data if d['video_id'] in val_vids]

    # Load 2,598 Test Tensors
    test_cached = torch.load(BASE_DIR / 'iddped_official_2598_test_tensors.pt', map_location='cpu', weights_only=False)
    V_M_test = test_cached['V_M'].to(DEVICE)
    V_S_test = test_cached['V_S'].to(DEVICE)
    V_I_test = test_cached['V_I'].to(DEVICE)
    K_M_test = test_cached['K_M'].to(DEVICE)
    y_test   = test_cached['targets']
    pids_test = test_cached['pids']
    N_test   = len(y_test)

    print(f"Loaded Datasets: Train={len(train_samples)} (Pos: {sum(s['label']==1 for s in train_samples)}), "
          f"Val={len(val_samples)} (Pos: {sum(s['label']==1 for s in val_samples)}), "
          f"Test={N_test} (Pos: {(y_test==1).sum()}, Pedestrians: {len(np.unique(pids_test))})")
    sys.stdout.flush()

    seeds = [42, 101, 2024, 7, 999]
    epochs = 12
    batch_size = 64

    focal_crit = FocalLoss(alpha=0.75, gamma=2.0)
    bce_crit = nn.BCEWithLogitsLoss()

    trained_models = {}
    val_y_arr = np.array([int(s['label']) for s in val_samples])

    # ==============================================================================
    # 2. TRAINING LOOP ACROSS 5 SEEDS (WITH RANDOM OVER-SAMPLING)
    # ==============================================================================
    for seed in seeds:
        print("\n" + "-" * 70)
        print(f"  [SEED {seed}] Training Adaptive Fusion Architecture ({epochs} Epochs with ROS)...")
        print("-" * 70)
        sys.stdout.flush()
        t_start = time.time()
        seed_everything(seed)

        # Build ROS Training Dataset (1:1 balanced ratio)
        train_ds = IDDPedAnticipationDataset(train_samples, oversample_positives=True,
                                             target_ratio=1.0, seed=seed)
        val_ds   = IDDPedAnticipationDataset(val_samples, oversample_positives=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = AdaptiveFusionMasterModel().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
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

            # Validation Evaluation
            model.eval()
            val_probs = []
            with torch.no_grad():
                for b in val_loader:
                    vm = b['V_M'].to(DEVICE)
                    vs = b['V_S'].to(DEVICE)
                    vi = b['V_I'].to(DEVICE)
                    km = b['K_M'].to(DEVICE)
                    logits, _ = model(vm, vs, vi, km)
                    p = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                    val_probs.extend(p)

            val_auc = roc_auc_score(val_y_arr, val_probs) if len(np.unique(val_y_arr)) > 1 else 0.5
            mean_loss = ep_loss / len(train_loader)

            if ep % 3 == 0 or ep == epochs:
                print(f"    Ep {ep:2d}/{epochs:2d} | Train Loss: {mean_loss:.4f} | Val AUC: {val_auc:.4f}")
                sys.stdout.flush()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_ep = ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        ckpt_path = os.path.join(CKPT_DIR, f'adaptive_fusion_seed{seed}.pth')
        try:
            torch.save(best_state, ckpt_path)
            print(f"  Seed {seed} Done in {time.time()-t_start:.1f}s | Best Ep: {best_ep} | Best Val AUC: {best_val_auc:.4f} | Checkpoint Saved.")
        except Exception as e:
            print(f"  Seed {seed} Done in {time.time()-t_start:.1f}s | Best Ep: {best_ep} | Best Val AUC: {best_val_auc:.4f} (Ckpt note: {e})")
        sys.stdout.flush()
        trained_models[seed] = model

    # ==============================================================================
    # 3. MATCHED 4-VARIANT ABLATION ON 2,598 TEST WINDOWS
    # ==============================================================================
    print("\n" + "=" * 80)
    print("  EVALUATION OF 4 MATCHED ABLATION VARIANTS ON OFFICIAL 2,598 TEST WINDOWS")
    print("=" * 80)
    sys.stdout.flush()

    variant_configs = {
        '1_kinematics_only': {
            'name': 'Kinematics Only (K_M)',
            'kwargs': {'disable_vision': True, 'disable_kinematics': False, 'fixed_uniform_fusion': False},
            'desc': 'Pedestrian motion trajectory dynamics alone'
        },
        '2_visual_only': {
            'name': 'Visual Streams Only (V_M, V_S, V_I)',
            'kwargs': {'disable_vision': False, 'disable_kinematics': True, 'fixed_uniform_fusion': False},
            'desc': 'Pedestrian crop, scene context and surround alone'
        },
        '3_fixed_uniform_fusion': {
            'name': 'Fixed Uniform Fusion (50/50)',
            'kwargs': {'disable_vision': False, 'disable_kinematics': False, 'fixed_uniform_fusion': True},
            'desc': 'Equal fixed 50/50 weighting of kinematics and visual streams'
        },
        '4_full_adaptive_fusion': {
            'name': 'Full Proposed Architecture (Scene-Adaptive Gate)',
            'kwargs': {'disable_vision': False, 'disable_kinematics': False, 'fixed_uniform_fusion': False},
            'desc': 'Full 6-encoder architecture with scene-adaptive gate and occlusion/illumination blocks'
        }
    }

    all_results = {}

    for v_key, v_info in variant_configs.items():
        v_name = v_info['name']
        v_kwargs = v_info['kwargs']
        print(f"\nEvaluating Variant: {v_name}...")
        sys.stdout.flush()

        seed_probs = []
        seed_metrics_tau_star = []
        seed_metrics_tau_05 = []
        tau_stars = []

        for seed in seeds:
            model = trained_models[seed]
            model.eval()

            # 1. Calibrate tau* on validation set
            val_ds = IDDPedAnticipationDataset(val_samples, oversample_positives=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            val_p = []
            with torch.no_grad():
                for b in val_loader:
                    vm = b['V_M'].to(DEVICE)
                    vs = b['V_S'].to(DEVICE)
                    vi = b['V_I'].to(DEVICE)
                    km = b['K_M'].to(DEVICE)
                    logits, _ = model(vm, vs, vi, km, **v_kwargs)
                    val_p.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
            tau_star = calibrate_tau_on_val(val_y_arr, np.array(val_p))
            tau_stars.append(tau_star)

            # 2. Evaluate on 2,598 Test Windows in mini-batches
            test_p = []
            test_bs = 128
            with torch.no_grad():
                for i in range(0, N_test, test_bs):
                    vm_b = V_M_test[i:i+test_bs]
                    vs_b = V_S_test[i:i+test_bs]
                    vi_b = V_I_test[i:i+test_bs]
                    km_b = K_M_test[i:i+test_bs]
                    logits_b, _ = model(vm_b, vs_b, vi_b, km_b, **v_kwargs)
                    test_p.extend(torch.sigmoid(logits_b.squeeze(-1)).cpu().numpy())

            test_p = np.array(test_p)
            seed_probs.append(test_p)

            m_star = eval_metrics(y_test, test_p, tau=tau_star)
            m_05   = eval_metrics(y_test, test_p, tau=0.50)
            seed_metrics_tau_star.append(m_star)
            seed_metrics_tau_05.append(m_05)

        seed_probs = np.array(seed_probs)
        ensemble_probs = np.mean(seed_probs, axis=0)
        ens_tau_star = float(np.mean(tau_stars))
        ens_metrics_star = eval_metrics(y_test, ensemble_probs, tau=ens_tau_star)
        ens_metrics_05   = eval_metrics(y_test, ensemble_probs, tau=0.50)

        # Aggregate stats
        aucs = [m['ROC_AUC'] for m in seed_metrics_tau_star]
        aps  = [m['PR_AUC'] for m in seed_metrics_tau_star]
        f1s_star = [m['F1_Score'] for m in seed_metrics_tau_star]
        accs_star = [m['Accuracy'] for m in seed_metrics_tau_star]
        recs_star = [m['Recall'] for m in seed_metrics_tau_star]
        precs_star = [m['Precision'] for m in seed_metrics_tau_star]
        accs_05 = [m['Accuracy'] for m in seed_metrics_tau_05]

        # Bootstrap analysis for full model against published PCPA (0.7113)
        boot_res = None
        if v_key == '4_full_adaptive_fusion':
            boot_res = cluster_bootstrap_test(y_test, ensemble_probs, pids_test,
                                              baseline_auc=0.7113, B=1000, seed=42)

        all_results[v_key] = {
            'name': v_name,
            'desc': v_info['desc'],
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'mean_ap': float(np.mean(aps)),
            'std_ap': float(np.std(aps)),
            'mean_f1_star': float(np.mean(f1s_star)),
            'std_f1_star': float(np.std(f1s_star)),
            'mean_acc_star': float(np.mean(accs_star)),
            'mean_rec_star': float(np.mean(recs_star)),
            'mean_prec_star': float(np.mean(precs_star)),
            'mean_acc_05': float(np.mean(accs_05)),
            'seed_aucs': {str(s): float(a) for s, a in zip(seeds, aucs)},
            'ensemble_auc': float(ens_metrics_star['ROC_AUC']),
            'ensemble_ap': float(ens_metrics_star['PR_AUC']),
            'ensemble_f1_star': float(ens_metrics_star['F1_Score']),
            'ensemble_acc_star': float(ens_metrics_star['Accuracy']),
            'bootstrap_test_vs_pcpa': boot_res
        }

        print(f"  -> 5-Seed Mean ROC-AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} | PR-AUC: {np.mean(aps):.4f}")
        print(f"  -> Ensemble ROC-AUC:    {ens_metrics_star['ROC_AUC']:.4f} | Accuracy (tau*): {np.mean(accs_star)*100:.2f}%")
        sys.stdout.flush()

    # ==============================================================================
    # 4. MASTER RESULTS TABLE PRINT & JSON EXPORT
    # ==============================================================================
    print("\n" + "=" * 90)
    print("  FINAL PUBLICATION-GRADE BENCHMARK TABLE (2,598 Official Windows)")
    print("=" * 90)
    header = f"{'Model Variant':<45} | {'ROC-AUC':<16} | {'PR-AUC (AP)':<12} | {'F1 (tau*)':<10} | {'Acc (tau*)':<10} | {'Acc (@0.50)':<10}"
    print(header)
    print("-" * len(header))
    print(f"{'PCPA (Official Published Baseline)':<45} | {'0.7113':<16} | {'0.2442':<12} | {'0.3345':<10} | {'69.98%':<10} | {'69.98%':<10}")
    for k, res in all_results.items():
        auc_str = f"{res['mean_auc']:.4f} +/- {res['std_auc']:.4f}"
        ap_str = f"{res['mean_ap']:.4f}"
        f1_str = f"{res['mean_f1_star']:.4f}"
        acc_star_str = f"{res['mean_acc_star']*100:.2f}%"
        acc_05_str = f"{res['mean_acc_05']*100:.2f}%"
        print(f"{res['name']:<45} | {auc_str:<16} | {ap_str:<12} | {f1_str:<10} | {acc_star_str:<10} | {acc_05_str:<10}")

    full_res = all_results['4_full_adaptive_fusion']
    print("\n" + "=" * 90)
    print(f"  PROPOSED ARCHITECTURE HEADLINE SUMMARY:")
    print(f"  5-Seed Mean ROC-AUC:   {full_res['mean_auc']:.4f} +/- {full_res['std_auc']:.4f}")
    print(f"  Ensemble ROC-AUC:      {full_res['ensemble_auc']:.4f} (+{full_res['ensemble_auc'] - 0.7113:+.4f} vs Published PCPA)")
    print(f"  Accuracy at tau*:      {full_res['mean_acc_star']*100:.2f}% (+{full_res['mean_acc_star']*100 - 69.98:+.2f}% vs PCPA)")
    print(f"  Accuracy at 0.50:      {full_res['mean_acc_05']*100:.2f}% (+{full_res['mean_acc_05']*100 - 69.98:+.2f}% vs PCPA)")
    if full_res['bootstrap_test_vs_pcpa']:
        b = full_res['bootstrap_test_vs_pcpa']
        print(f"  Cluster Bootstrap SE:  {b['bootstrap_se']:.4f} | 95% CI on Delta: [{b['ci_95'][0]:+.4f}, {b['ci_95'][1]:+.4f}]")
    print("=" * 90)

    out_json = os.path.join(SCRATCH_DIR, 'results_adaptive_fusion_2598.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved complete benchmark results to: {out_json}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
