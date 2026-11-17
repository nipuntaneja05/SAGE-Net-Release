"""
iddped_anticipation_dataset.py
Dataset module for IDD-PeD crossing anticipation with:
- Random Over-Sampling (ROS) to manage the severe 1:16 class imbalance
- Strict zero-ground-zero verification assertions (TTE in [30, 60], obs_length = 16)
- Efficient PyTorch DataLoader generation
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader


class IDDPedAnticipationDataset(Dataset):
    """
    Anticipation Dataset for IDD-PeD:
    Yields strictly pre-crossing observation sequences (16 frames):
    - V_M: [16, 1792]
    - V_S: [16, 1792]
    - V_I: [16, 1792]
    - K_M: [16, 17]
    - label: float tensor (0.0 or 1.0)
    """
    def __init__(self, samples, oversample_positives=False, target_ratio=1.0, seed=42):
        self.original_samples = samples
        self.oversample_positives = oversample_positives
        self.target_ratio = target_ratio
        self.seed = seed
        self.samples = self._build_samples()

    def _build_samples(self):
        # Strict zero-ground-zero verification assertions
        for s in self.original_samples:
            assert s['V_M'].shape[0] == 16, f"Observation length must be 16, got {s['V_M'].shape[0]}"
            assert s['K_M'].shape[0] == 16, f"Kinematic length must be 16, got {s['K_M'].shape[0]}"
            assert 30 <= s['time_to_event'] <= 60, f"TTE must be in [30, 60] (pre-crossing), got {s['time_to_event']}"

        if not self.oversample_positives:
            return list(self.original_samples)

        # Separate positive and negative classes
        pos_samples = [s for s in self.original_samples if s['label'] == 1.0]
        neg_samples = [s for s in self.original_samples if s['label'] == 0.0]

        n_pos = len(pos_samples)
        n_neg = len(neg_samples)

        if n_pos == 0 or n_neg == 0:
            return list(self.original_samples)

        # Target positive count to achieve desired ratio (target_ratio * n_neg)
        target_pos_count = int(n_neg * self.target_ratio)
        rng = random.Random(self.seed)

        # Random over-sampling with replacement of minority positive class
        oversampled_pos = [rng.choice(pos_samples) for _ in range(target_pos_count)]

        combined = list(neg_samples) + oversampled_pos
        rng.shuffle(combined)
        return combined

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'V_M': s['V_M'],
            'V_S': s['V_S'],
            'V_I': s['V_I'],
            'K_M': s['K_M'],
            'label': torch.tensor(s['label'], dtype=torch.float32),
            'video_id': s.get('video_id', ''),
            'pedestrian_id': s.get('pedestrian_id', '')
        }


def get_iddped_dataloaders(all_data, train_vids, val_vids, batch_size=64,
                           use_ros=True, target_ratio=1.0, seed=42):
    """
    Creates train and validation DataLoaders with frozen video-level partition.
    """
    train_samples = [s for s in all_data if s['video_id'] in train_vids]
    val_samples   = [s for s in all_data if s['video_id'] in val_vids]

    train_ds = IDDPedAnticipationDataset(train_samples, oversample_positives=use_ros,
                                         target_ratio=target_ratio, seed=seed)
    val_ds   = IDDPedAnticipationDataset(val_samples, oversample_positives=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_samples, val_samples
