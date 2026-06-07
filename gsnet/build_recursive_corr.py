#
# Recursive-densification training data (idea B): teach GS-Net to be
# DENSITY-AGNOSTIC so it can be applied recursively (densify -> select -> densify).
#
# For each TRAIN clip we add correspondences whose INPUT is a random subsample of
# that clip's G_dense (at a few densities) and whose TARGET is the full G_dense.
# i.e. "given a partial dense cloud, predict the full dense" -- exactly the step a
# recursive pass performs. We merge these with the original SfM->G_dense corr
# (copied in) so the trained net handles BOTH sparse-SfM and denser inputs.
#
# Usage:
#   python -m gsnet.build_recursive_corr \
#       --root /mnt/zihanw/gsnet_nusc --gdense_dir runs/nusc/gdense \
#       --sfm_corr runs/nusc/corr --out_dir runs/nusc_recur/corr \
#       --test_scenes 348_clip_09 ... --T 5 --global_scale 45 --no_filter \
#       --subsamples 8000 16000 --workers 8
#
import argparse
import glob
import os
import shutil
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsnet.waymo import discover_scenes, seg_name, gdense_path as gd_path
from gsnet.common import read_dense_gaussians, filter_dense
from gsnet.build_correspondences import build_for_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gdense_dir", required=True)
    ap.add_argument("--sfm_corr", default="", help="existing SfM->G_dense corr to copy in (merge)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_scenes", nargs="+", default=[])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--T", type=int, default=5, help="K = T (loss matches 1:1)")
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--global_scale", type=float, default=0.0)
    ap.add_argument("--no_filter", action="store_true")
    ap.add_argument("--subsamples", type=int, nargs="+", default=[8000, 16000],
                    help="input densities (#points subsampled from G_dense) to train on")
    ap.add_argument("--radius_margin", type=float, default=1.5)
    ap.add_argument("--opacity_min", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) copy the SfM corr in (so training sees BOTH sparse-SfM and dense inputs)
    if args.sfm_corr:
        for f in glob.glob(os.path.join(args.sfm_corr, "*.npz")):
            dst = os.path.join(args.out_dir, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy(f, dst)
        print(f"[recur] copied SfM corr from {args.sfm_corr}")

    filt = dict(radius_margin=(1e9 if args.no_filter else args.radius_margin),
                opacity_min=(0.0 if args.no_filter else args.opacity_min), sor_k=0)
    scale_override = args.global_scale if args.global_scale > 0 else None
    rng = np.random.default_rng(0)

    def is_test(name):
        return any(t in name for t in args.test_scenes)

    train = [s for s in discover_scenes(args.root) if not is_test(seg_name(s))]
    print(f"[recur] {len(train)} train clips, subsamples={args.subsamples}")

    tasks = []
    for s in train:
        gd = gd_path(args.gdense_dir, s, args.iterations)
        if not os.path.exists(gd):
            print(f"[skip] {seg_name(s)}: no G_dense {gd}")
            continue
        g = read_dense_gaussians(gd)
        xyz_all, rgb_all = g["xyz"], g["rgb"]
        for n in args.subsamples:
            if n >= xyz_all.shape[0]:
                continue
            idx = rng.choice(xyz_all.shape[0], n, replace=False)
            inp = (xyz_all[idx].astype(np.float32), rgb_all[idx].astype(np.float32))
            out = os.path.join(args.out_dir, f"{seg_name(s)}_sub{n}.npz")
            if os.path.exists(out):
                continue
            tasks.append((inp, gd, out))

    common = dict(K=args.T, M=args.M, normalize=True, scale_override=scale_override, **filt)
    for i, (inp, gd, out) in enumerate(tasks):
        build_for_sequence(None, gd, out, input_override=inp, **common)
        if (i + 1) % 10 == 0:
            print(f"[recur] {i+1}/{len(tasks)}", flush=True)
    print(f"[recur] built {len(tasks)} recursive-corr files -> {args.out_dir}")


if __name__ == "__main__":
    main()
