#
# Build GS-Net training correspondences for the Waymo experiment.
#
# For each TRAINING scene (all discovered scenes except --test_scenes), pairs the
# sparse SfM points with the generated G_dense and writes a correspondence .npz
# (same format/normalization/filtering as the CARLA pipeline).
#
# Usage:
#   python -m gsnet.waymo_corr \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input \
#       --gdense_dir runs/waymo_gdense \
#       --out_dir CORR/waymo_train \
#       --test_scenes <segA> <segB> --workers 8
#

import argparse
import json
import os
import time

from gsnet.waymo import discover_scenes, seg_name, sparse_points, gdense_path
from gsnet.build_correspondences import build_for_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gdense_dir", default="runs/waymo_gdense")
    ap.add_argument("--out_dir", default="CORR/waymo_train")
    ap.add_argument("--test_scenes", nargs="+", default=[],
                    help="segment names to EXCLUDE from training (the test set)")
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--radius_margin", type=float, default=1.5)
    ap.add_argument("--opacity_min", type=float, default=0.005)
    ap.add_argument("--sor_k", type=int, default=0)
    ap.add_argument("--input_subsample", type=float, default=1.0,
                    help="fraction of INPUT sparse points to keep (sparse-input training)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    test = set(args.test_scenes)
    scenes = [s for s in discover_scenes(args.root) if seg_name(s) not in test]
    print(f"[waymo_corr] {len(scenes)} training scenes "
          f"(excluded {len(test)} test): {[seg_name(s) for s in scenes]}")

    common = dict(K=args.K, M=args.M, normalize=not args.no_normalize,
                  radius_margin=args.radius_margin, opacity_min=args.opacity_min,
                  sor_k=args.sor_k, input_subsample=args.input_subsample)
    tasks = []
    for s in scenes:
        gd = gdense_path(args.gdense_dir, s, args.iterations)
        sp = sparse_points(s)
        if not os.path.exists(gd):
            print(f"[skip] {seg_name(s)}: missing G_dense {gd}")
            continue
        if not os.path.exists(sp):
            print(f"[skip] {seg_name(s)}: missing sparse {sp}")
            continue
        tasks.append((sp, gd, os.path.join(args.out_dir, f"{seg_name(s)}.npz")))

    t0 = time.time()
    if args.workers > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(build_for_sequence, sp, gd, o, **common)
                    for sp, gd, o in tasks]
            stats = [f.result() for f in cf.as_completed(futs)]
    else:
        stats = [build_for_sequence(sp, gd, o, **common) for sp, gd, o in tasks]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "build_times.json"), "w") as f:
        json.dump({"total_seconds": time.time() - t0, "sequences": stats}, f, indent=2)
    print(f"[waymo_corr] done {len(tasks)} scenes in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
