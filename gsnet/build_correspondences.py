#
# Offline data-engine step (the part we own): build sparse->dense correspondences
# used as pseudo-ground-truth for GS-Net training.
#
# For each sparse SfM point p_n:
#   * find its M nearest sparse neighbors (for the geometry-aware encoder), and
#   * retrieve its K nearest dense Gaussians from (filtered) G_dense.
# Points are normalized per sequence (see common.compute_normalization) so the
# learned local mapping is scale/scene-invariant.
#
# Single sequence:
#   python -m gsnet.build_correspondences \
#       --sparse sparse_point/S01/101_sparse.ply \
#       --gdense input_output/output_101_dense/point_cloud/iteration_30000/point_cloud.ply \
#       --out CORR/train/101.npz --K 5 --M 3
#
# Whole dataset (auto-discovers training sequences 101-109,...,501-509):
#   python -m gsnet.build_correspondences --batch \
#       --io_dir /mnt/zihanw/carla/input_output \
#       --sparse_root /mnt/zihanw/carla/sparse_point \
#       --out_dir CORR/train
#

import argparse
import glob
import json
import os
import re
import time

import numpy as np
from scipy.spatial import cKDTree

from gsnet.common import (
    read_points_any, read_dense_gaussians, filter_dense, compute_normalization,
)


def build_for_sequence(sparse_path, gdense_path, out_path, K=5, M=3,
                       normalize=True, subsample=1.0, **filt_kwargs):
    t0 = time.time()
    cxyz, crgb = read_points_any(sparse_path)
    g = read_dense_gaussians(gdense_path)
    g = filter_dense(g, cxyz, **filt_kwargs)
    # Optional random subsampling of G_dense (probes sensitivity to pseudo-GT
    # density / MVS coverage; reproducible per output file).
    if subsample < 1.0:
        n0 = g["xyz"].shape[0]
        rng = np.random.default_rng(abs(hash(os.path.basename(out_path))) % (2**32))
        keep = rng.choice(n0, int(round(n0 * subsample)), replace=False)
        g = {k: v[keep] for k, v in g.items()}
        print(f"[subsample] {os.path.basename(out_path)}: {n0} -> {g['xyz'].shape[0]} "
              f"({subsample:.2f})")
    N = cxyz.shape[0]

    # Per-sequence normalization (from sparse points only -> reproducible).
    center, scale = compute_normalization(cxyz) if normalize else (
        np.zeros(3, np.float32), 1.0)
    n_cxyz = (cxyz - center) / scale
    n_gmu = (g["xyz"] - center) / scale
    n_gscale = g["scale"] / scale

    # M nearest sparse neighbors (exclude self).
    stree = cKDTree(n_cxyz)
    _, nn_idx = stree.query(n_cxyz, k=min(M + 1, N))
    nn_idx = np.atleast_2d(nn_idx)[:, 1:M + 1]
    if nn_idx.shape[1] < M:
        nn_idx = np.concatenate(
            [nn_idx, np.repeat(nn_idx[:, -1:], M - nn_idx.shape[1], axis=1)], axis=1)

    # K nearest dense Gaussians per sparse point (pseudo-GT set).
    dtree = cKDTree(n_gmu)
    _, k_idx = dtree.query(n_cxyz, k=K)
    k_idx = k_idx.reshape(N, K)

    # Clip normalized GT scales to the sigmoid-representable range (0, 1). A few
    # very large Gaussians (ground/sky planes) may exceed it after normalization.
    gt_scale = n_gscale[k_idx]
    frac_clip = float((gt_scale > 0.999).mean())
    if frac_clip > 0.01:
        print(f"[warn] {os.path.basename(out_path)}: {frac_clip*100:.1f}% of GT "
              f"scales > 1 after normalization (clipped to 0.999)")
    gt_scale = np.clip(gt_scale, 1e-6, 0.999)

    out = dict(
        center_xyz=n_cxyz.astype(np.float32),
        center_rgb=crgb.astype(np.float32),
        neighbor_xyz=n_cxyz[nn_idx].astype(np.float32),
        neighbor_rgb=crgb[nn_idx].astype(np.float32),
        gt_mu=n_gmu[k_idx].astype(np.float32),
        gt_rgb=g["rgb"][k_idx].astype(np.float32),
        gt_scale=gt_scale.astype(np.float32),
        gt_quat=g["quat"][k_idx].astype(np.float32),
        gt_opacity=g["opacity"][k_idx].astype(np.float32),
        norm_center=center.astype(np.float32),
        norm_scale=np.float32(scale),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **out)
    dt = time.time() - t0
    print(f"[corr] {os.path.basename(out_path)}: N={N}, K={K}, M={M}, "
          f"dense={g['xyz'].shape[0]}, scale={scale:.3f}, {dt:.1f}s")
    return dict(out=out_path, N=int(N), dense=int(g["xyz"].shape[0]),
                scale=float(scale), seconds=float(dt))


def discover_training_sequences(io_dir, sparse_root, iteration=30000):
    """Map each output_<id>_dense (training target) to its sparse SfM .ply."""
    seqs = []
    for p in sorted(glob.glob(os.path.join(io_dir, "output_*_dense"))):
        m = re.search(r"output_(\d+)_dense", os.path.basename(p))
        if not m:
            continue
        sid = m.group(1)
        gdense = os.path.join(p, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        scene = int(sid) // 100
        sparse = os.path.join(sparse_root, f"S{scene:02d}", f"{sid}_sparse.ply")
        if os.path.exists(gdense) and os.path.exists(sparse):
            seqs.append((sid, sparse, gdense))
        else:
            print(f"[skip] {sid}: missing "
                  f"{'gdense' if not os.path.exists(gdense) else ''} "
                  f"{'sparse' if not os.path.exists(sparse) else ''}")
    return seqs


def main():
    ap = argparse.ArgumentParser(description="Build GS-Net sparse->dense correspondences.")
    ap.add_argument("--batch", action="store_true", help="process whole dataset")
    # single-sequence
    ap.add_argument("--sparse", help="sparse SfM .ply/.bin/.txt (P_sfm)")
    ap.add_argument("--gdense", help="optimized dense 3DGS .ply (G_dense)")
    ap.add_argument("--out", help="output .npz (single)")
    # batch
    ap.add_argument("--io_dir", help="dir containing output_<id>_dense/")
    ap.add_argument("--sparse_root", help="dir containing S0X/<id>_sparse.ply")
    ap.add_argument("--out_dir", help="output dir for .npz (batch)")
    # common
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--radius_margin", type=float, default=1.5)
    ap.add_argument("--opacity_min", type=float, default=0.005)
    ap.add_argument("--sor_k", type=int, default=0, help="0 disables statistical outlier removal")
    ap.add_argument("--workers", type=int, default=1, help="parallel CPU workers (batch mode)")
    ap.add_argument("--gdense_iter", type=int, default=30000,
                    help="3DGS optimization iteration of G_dense to supervise from")
    ap.add_argument("--gdense_subsample", type=float, default=1.0,
                    help="fraction of G_dense to keep (pseudo-GT density sensitivity)")
    args = ap.parse_args()

    filt = dict(radius_margin=args.radius_margin, opacity_min=args.opacity_min,
                sor_k=args.sor_k)
    common = dict(K=args.K, M=args.M, normalize=not args.no_normalize,
                  subsample=args.gdense_subsample, **filt)

    if args.batch:
        assert args.io_dir and args.sparse_root and args.out_dir
        seqs = discover_training_sequences(args.io_dir, args.sparse_root, args.gdense_iter)
        print(f"[batch] {len(seqs)} training sequences, workers={args.workers}")
        t0 = time.time()
        tasks = [(sparse, gdense, os.path.join(args.out_dir, f"{sid}.npz"))
                 for sid, sparse, gdense in seqs]
        if args.workers and args.workers > 1:
            import concurrent.futures as cf
            stats = []
            with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(build_for_sequence, s, g, o, **common)
                        for s, g, o in tasks]
                for f in cf.as_completed(futs):
                    stats.append(f.result())
        else:
            stats = [build_for_sequence(s, g, o, **common) for s, g, o in tasks]
        total = time.time() - t0
        log = dict(total_seconds=total, sequences=stats)
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "build_times.json"), "w") as f:
            json.dump(log, f, indent=2)
        print(f"[batch] done {len(seqs)} seqs in {total:.1f}s "
              f"(log -> {args.out_dir}/build_times.json)")
    else:
        assert args.sparse and args.gdense and args.out
        build_for_sequence(args.sparse, args.gdense, args.out, **common)


if __name__ == "__main__":
    main()
