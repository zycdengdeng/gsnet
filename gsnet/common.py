#
# Shared utilities for the GS-Net data pipeline: robust point/Gaussian readers,
# G_dense outlier filtering, and per-sequence coordinate normalization.
#
# Coordinate handling (addresses cross-sequence "coordinate unification"):
# each sequence is normalized by a similarity transform derived *only from its
# sparse SfM points* (so it is reproducible at inference time, where G_dense is
# unavailable): x' = (x - center) / scale. The same transform is applied to the
# G_dense targets when building correspondences, and inverted after inference
# before exporting the init .ply (so 3DGS optimizes in the original COLMAP frame
# that matches the cameras).
#

import os
import sys

import numpy as np
from plyfile import PlyData

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.colmap_loader import read_points3D_binary, read_points3D_text

C0 = 0.28209479177387814


def _sh_to_rgb(f_dc):
    return np.clip(f_dc * C0 + 0.5, 0.0, 1.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def read_points_any(path):
    """Read a point cloud as (xyz (N,3) float32, rgb (N,3) float32 in [0,1]).

    Supports COLMAP points3D.bin/.txt and .ply files whose color is stored as
    red/green/blue (uint8) or f_dc_* (0-order SH). Missing color -> gray 0.5.
    """
    if path.endswith(".bin"):
        xyz, rgb, _ = read_points3D_binary(path)
        return xyz.astype(np.float32), (rgb.astype(np.float32) / 255.0)
    if path.endswith(".txt"):
        xyz, rgb, _ = read_points3D_text(path)
        return xyz.astype(np.float32), (rgb.astype(np.float32) / 255.0)

    ply = PlyData.read(path)
    v = ply.elements[0]
    names = set(p.name for p in v.properties)
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    if {"red", "green", "blue"} <= names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    elif {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
        rgb = _sh_to_rgb(f_dc)
    else:
        rgb = np.full_like(xyz, 0.5)
    return xyz, rgb


def read_dense_gaussians(path):
    """Read a standard 3DGS .ply (G_dense) into rendered-space attributes."""
    ply = PlyData.read(path)
    v = ply.elements[0]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    rgb = _sh_to_rgb(f_dc)
    opacity = _sigmoid(np.asarray(v["opacity"], dtype=np.float32)).reshape(-1, 1)
    scale = np.exp(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    )
    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True).clip(1e-8)
    return dict(xyz=xyz, rgb=rgb, opacity=opacity, scale=scale, quat=quat)


# --------------------------------------------------------------------------- #
# Filtering of G_dense outliers ("far, sparse junk" + near-transparent points)
# --------------------------------------------------------------------------- #
def filter_dense(g, ref_xyz, radius_margin=1.5, radius_pct=99.0,
                 opacity_min=0.005, sor_k=0, sor_std=2.0, verbose=True):
    """Remove G_dense outliers.

    1. Drop Gaussians farther than radius_margin * percentile(ref distances) from
       the reference (sparse) centroid -> removes far floating junk.
    2. Drop Gaussians with opacity < opacity_min -> removes near-invisible junk.
    3. Optional statistical outlier removal (mean kNN distance > mean + sor_std*std).
    """
    n0 = g["xyz"].shape[0]
    center = np.median(ref_xyz, axis=0)
    ref_d = np.linalg.norm(ref_xyz - center, axis=1)
    thresh = np.percentile(ref_d, radius_pct) * radius_margin

    d = np.linalg.norm(g["xyz"] - center, axis=1)
    keep = (d <= thresh) & (g["opacity"][:, 0] >= opacity_min)

    if sor_k and sor_k > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(g["xyz"])
        dist, _ = tree.query(g["xyz"], k=sor_k + 1)
        mean_d = dist[:, 1:].mean(axis=1)
        sor_keep = mean_d <= (mean_d.mean() + sor_std * mean_d.std())
        keep = keep & sor_keep

    g = {k: v[keep] for k, v in g.items()}
    if verbose:
        print(f"[filter] G_dense {n0} -> {g['xyz'].shape[0]} "
              f"(radius<{thresh:.3f}, opacity>={opacity_min})")
    return g


# --------------------------------------------------------------------------- #
# Per-sequence normalization (similarity: translate + isotropic scale)
# --------------------------------------------------------------------------- #
def compute_normalization(xyz, scale_pct=95.0):
    """Return (center (3,), scale (float)) from sparse points only."""
    center = np.median(xyz, axis=0).astype(np.float32)
    d = np.linalg.norm(xyz - center, axis=1)
    scale = float(max(np.percentile(d, scale_pct), 1e-6))
    return center, scale
