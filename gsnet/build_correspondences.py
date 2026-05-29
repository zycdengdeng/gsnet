#
# Offline data-engine step (the part I own): build sparse-to-dense
# correspondences used as pseudo-ground-truth for GS-Net training.
#
# For each sparse SfM point p_n:
#   * find its M nearest sparse neighbors (for the geometry-aware encoder), and
#   * retrieve its K nearest dense Gaussians from G_dense (the pseudo-GT set,
#     via a KD-tree over G_dense centers, Sec. IV-C / Sec. The CARLA-NVS Dataset).
#
# Inputs (standard formats):
#   --sfm     COLMAP points3D.bin / points3D.txt  (sparse SfM point cloud)
#   --gdense  optimized dense 3DGS point_cloud.ply (G_dense)
# Output: a self-contained .npz per sequence so training can simply concatenate
# points across all scenes/sequences and sample batches.
#
# Usage (single sequence):
#   python -m gsnet.build_correspondences \
#       --sfm  SCENE/seq/sparse/0/points3D.bin \
#       --gdense SCENE/seq/gaussians/point_cloud.ply \
#       --out  CORR/scene_seq.npz --K 5 --M 3
#

import argparse
import os
import sys

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

# Make repo modules importable when run as a script.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.colmap_loader import read_points3D_binary, read_points3D_text

C0 = 0.28209479177387814


def _sh_to_rgb(f_dc):
    return np.clip(f_dc * C0 + 0.5, 0.0, 1.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def read_sparse_points(path):
    """Return (xyz (N,3), rgb (N,3) in [0,1]) from a COLMAP points3D file."""
    if path.endswith(".bin"):
        xyz, rgb, _ = read_points3D_binary(path)
    else:
        xyz, rgb, _ = read_points3D_text(path)
    return xyz.astype(np.float32), (rgb.astype(np.float32) / 255.0)


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


def build_for_sequence(sfm_path, gdense_path, out_path, K=5, M=3):
    cxyz, crgb = read_sparse_points(sfm_path)
    g = read_dense_gaussians(gdense_path)
    N = cxyz.shape[0]

    # M nearest sparse neighbors (exclude self -> query K=M+1, drop first).
    sparse_tree = cKDTree(cxyz)
    _, nn_idx = sparse_tree.query(cxyz, k=min(M + 1, N))
    nn_idx = np.atleast_2d(nn_idx)[:, 1:M + 1]
    # Pad if a (degenerate) tiny cloud has fewer than M neighbors.
    if nn_idx.shape[1] < M:
        pad = np.repeat(nn_idx[:, -1:], M - nn_idx.shape[1], axis=1)
        nn_idx = np.concatenate([nn_idx, pad], axis=1)
    neighbor_xyz = cxyz[nn_idx]          # (N, M, 3)
    neighbor_rgb = crgb[nn_idx]          # (N, M, 3)

    # K nearest dense Gaussians per sparse point (pseudo-GT set).
    dense_tree = cKDTree(g["xyz"])
    _, k_idx = dense_tree.query(cxyz, k=K)
    k_idx = np.atleast_2d(k_idx) if K > 1 else k_idx.reshape(N, 1)

    out = dict(
        center_xyz=cxyz,
        center_rgb=crgb,
        neighbor_xyz=neighbor_xyz.astype(np.float32),
        neighbor_rgb=neighbor_rgb.astype(np.float32),
        gt_mu=g["xyz"][k_idx].astype(np.float32),        # (N, K, 3)
        gt_rgb=g["rgb"][k_idx].astype(np.float32),       # (N, K, 3)
        gt_scale=g["scale"][k_idx].astype(np.float32),   # (N, K, 3)
        gt_quat=g["quat"][k_idx].astype(np.float32),     # (N, K, 4)
        gt_opacity=g["opacity"][k_idx].astype(np.float32),  # (N, K, 1)
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"[corr] {os.path.basename(out_path)}: N={N} sparse pts, "
          f"K={K}, M={M}, dense={g['xyz'].shape[0]}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Build GS-Net sparse->dense correspondences.")
    ap.add_argument("--sfm", required=True, help="COLMAP points3D.bin/.txt")
    ap.add_argument("--gdense", required=True, help="optimized dense 3DGS .ply (G_dense)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--M", type=int, default=3)
    args = ap.parse_args()
    build_for_sequence(args.sfm, args.gdense, args.out, K=args.K, M=args.M)


if __name__ == "__main__":
    main()
