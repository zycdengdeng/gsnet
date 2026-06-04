#
# Generate G_dense (training targets) for Waymo scenes: run 3DGS initialized from
# the MVS fused.ply over ALL 60 images (no train/test split), 30k iterations.
# Scenes are distributed across --gpus. Resumable (skips scenes already done).
#
# Usage:
#   python -m gsnet.waymo_gdense \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input \
#       --out_dir runs/waymo_gdense --gpus 2 3 4 5 6 7
#

import argparse
import concurrent.futures as cf
import json
import os
import queue
import subprocess
import sys
import time

from gsnet.waymo import discover_scenes, seg_name, dense_dir, fused_ply

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n[gpu{gpu}] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def gdense_path(out_dir, scene, iterations):
    return os.path.join(out_dir, seg_name(scene), "point_cloud",
                        f"iteration_{iterations}", "point_cloud.ply")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_dir", default="runs/waymo_gdense")
    ap.add_argument("--scenes", nargs="+", default=None, help="segment names (default: all)")
    ap.add_argument("--gpus", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--train_extra", default="",
                    help="extra args to train.py, e.g. \"--densify_grad_threshold 0.0004\" "
                         "to cap densification (anti-OOM on wide-baseline sparse scenes)")
    args = ap.parse_args()
    extra = args.train_extra.split()

    scenes = ([os.path.join(args.root, s) for s in args.scenes]
              if args.scenes else discover_scenes(args.root))
    print(f"[gdense] {len(scenes)} scenes")

    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)
    results = {}
    lock = __import__("threading").Lock()

    def work(scene):
        gpu = gpu_q.get()
        try:
            mp = os.path.join(args.out_dir, seg_name(scene))
            gd = gdense_path(args.out_dir, scene, args.iterations)
            if os.path.exists(gd):
                print(f"[skip] {seg_name(scene)} (G_dense exists)")
                with lock:
                    results[seg_name(scene)] = {"gdense": gd, "seconds": 0, "skipped": True}
                return
            t0 = time.time()
            it = str(args.iterations)
            run([PY, "train.py", "-s", dense_dir(scene), "-m", mp,
                 "--init_pcd", fused_ply(scene),
                 "--iterations", it, "--test_iterations", it,
                 "--save_iterations", it, "--disable_viewer", "--quiet", *extra], gpu)
            with lock:
                results[seg_name(scene)] = {"gdense": gd, "seconds": time.time() - t0}
            print(f"[done] {seg_name(scene)} -> {gd}")
        except Exception as e:                       # one scene's OOM must not abort the batch
            print(f"[FAILED] {seg_name(scene)}: {e}", flush=True)
        finally:
            gpu_q.put(gpu)

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        for f in cf.as_completed([ex.submit(work, s) for s in scenes]):
            f.result()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "gdense_times.json"), "w") as f:
        json.dump({"total_seconds": time.time() - t0, "scenes": results}, f, indent=2)
    print(f"\n[gdense] done {len(scenes)} scenes in {(time.time()-t0)/60:.1f} min "
          f"-> {args.out_dir}/gdense_times.json")


if __name__ == "__main__":
    main()
