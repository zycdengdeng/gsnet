#
# Dataset over precomputed sparse->dense correspondences (.npz files produced
# by gsnet/build_correspondences.py). Each sparse SfM point is an independent
# training sample because its M-neighbor features are baked into the .npz, so we
# simply concatenate all points across scenes/sequences and sample batches.
#

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

_KEYS = [
    "center_xyz", "center_rgb", "neighbor_xyz", "neighbor_rgb",
    "gt_mu", "gt_rgb", "gt_scale", "gt_quat", "gt_opacity",
]


class CorrespondenceDataset(Dataset):
    """Loads and concatenates all correspondence .npz files under a directory."""

    def __init__(self, corr_dir, pattern="*.npz"):
        files = sorted(glob.glob(os.path.join(corr_dir, pattern)))
        if not files:
            raise FileNotFoundError(f"No correspondence files in {corr_dir}/{pattern}")
        buffers = {k: [] for k in _KEYS}
        for f in files:
            d = np.load(f)
            for k in _KEYS:
                buffers[k].append(d[k])
        self.data = {
            k: torch.from_numpy(np.concatenate(buffers[k], axis=0)).float()
            for k in _KEYS
        }
        self.n = self.data["center_xyz"].shape[0]
        self.files = files
        print(f"[dataset] {len(files)} sequences, {self.n} sparse points total")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: self.data[k][i] for k in _KEYS}
