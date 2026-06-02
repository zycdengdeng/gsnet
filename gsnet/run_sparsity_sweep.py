#
# Sparsity sweep (Waymo SSE): vary how many frames-per-camera are held out as
# test (--n_holdouts). Larger holdout => fewer TRAINING views => sparser regime.
# At each level runs BOTH baseline (sparse-SfM init) and GS-Net+3DGS, then
# tabulates the gap (gsnet - baseline). Tests the regime hypothesis: GS-Net
# should approach/overtake baseline as training views get sparse (less for plain
# SfM init to exploit), while it trails when views are dense (SfM init already
# good).
#
# NOTE: this varies the number of TRAINING VIEWS. The init point cloud (and the
# GS-Net input) still come from the full-scene SfM; only the supervision gets
# sparser. (For genuinely sparse INPUT geometry you'd rebuild SfM from the train
# frames -- a separate, heavier experiment.)
#
# Scheduler: all (level x scene x method) jobs are flattened into ONE pool and
# packed across --gpus with --jobs_per_gpu slots each (oversubscription), so 6
# free GPUs stay busy. Each (level,scene) gets its OWN isolated sparse copy +
# test.txt, so nothing collides and the whole run is resumable.
#
# Usage (5-cam, 20 frames/camera, GPUs 0-5):
#   python -m gsnet.run_sparsity_sweep \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam \
#       --test_scenes <segA> <segB> \
#       --ckpt runs/waymo5_gsnet/gsnet_latest.pt \
#       --out_dir runs/waymo5_sparsity --n_holdouts 4 12 16 18 \
#       --frames_per_cam 20 --gpus 0 1 2 3 4 5 --jobs_per_gpu 2
#

import argparse
import concurrent.futures as cf
import json
import os
import queue
import subprocess
import sys
import threading
import time

from gsnet.waymo import resolve_scene, seg_name, scene_source, sparse_points
from gsnet.make_cam_split import write_cam_split

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n[gpu{gpu}] $ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)
    return time.time() - t0


def read_results(mp):
    with open(os.path.join(mp, "results.json")) as f:
        d = json.load(f)
    m = sorted(d.keys())[-1]
    return {k: d[m][k] for k in ("PSNR", "SSIM", "LPIPS")}


def optimize_eval(source, mp, iterations, extra, gpu):
    it = str(iterations)
    s = run([PY, "train.py", "-s", source, "-m", mp, "--eval",
             "--iterations", it, "--test_iterations", it, "--save_iterations", it,
             "--disable_viewer", "--quiet", *extra], gpu)
    run([PY, "render.py", "-m", mp, "--skip_train", "--quiet"], gpu)
    run([PY, "metrics.py", "-m", mp], gpu)
    return {**read_results(mp), "optim_seconds": s}


def aggregate(data, args):
    """data[nh][scene][cfg] = metrics -> sorted rows + md table with the gap."""
    rows = []
    for nh in sorted(data):
        row = {"n_holdout": nh}
        if args.frames_per_cam:
            row["train_per_cam"] = args.frames_per_cam - nh
        for cfg in ("baseline", "gsnet"):
            vals = [s[cfg] for s in data[nh].values() if cfg in s]
            if vals:
                row[cfg] = {k: sum(v[k] for v in vals) / len(vals)
                            for k in ("PSNR", "SSIM", "LPIPS")}
                row[cfg]["n_scenes"] = len(vals)
        rows.append(row)
    json.dump({"per_scene": data, "rows": rows},
              open(os.path.join(args.out_dir, "sparsity_sweep.json"), "w"), indent=2)

    has_train = any("train_per_cam" in r for r in rows)
    head = "| n_holdout |" + (" train/cam |" if has_train else "") + \
           " baseline PSNR | gsnet PSNR | Δ(gsnet-base) |"
    sep = "|" + "---|" * (head.count("|") - 1)
    lines = ["", f"## Sparsity sweep (Waymo SSE, @{args.iterations} iters) "
                 f"-- larger n_holdout = sparser training", head, sep]
    for r in rows:
        b = r.get("baseline", {}).get("PSNR")
        g = r.get("gsnet", {}).get("PSNR")
        d = (g - b) if (b is not None and g is not None) else None
        cells = [str(r["n_holdout"])]
        if has_train:
            cells.append(str(r.get("train_per_cam", "?")))
        cells.append(f"{b:.2f}" if b is not None else "-")
        cells.append(f"{g:.2f}" if g is not None else "-")
        cells.append(f"{d:+.2f}" if d is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    open(os.path.join(args.out_dir, "sparsity_sweep.md"), "w").write(table + "\n")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--test_scenes", nargs="+", required=True)
    ap.add_argument("--ckpt", default="runs/waymo5_gsnet/gsnet_latest.pt")
    ap.add_argument("--out_dir", default="runs/waymo5_sparsity")
    ap.add_argument("--n_holdouts", type=int, nargs="+", default=[4, 12, 16, 18])
    ap.add_argument("--frames_per_cam", type=int, default=0,
                    help="if set, also report train frames/camera (= this - n_holdout)")
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--jobs_per_gpu", type=int, default=2,
                    help="concurrent jobs packed per GPU (oversubscription)")
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--skip_gsnet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_tag = args.out_dir.strip("/").replace("/", "_")
    scenes = [resolve_scene(args.root, s) for s in args.test_scenes]

    # Write every (level, scene) split up front into its OWN isolated copy, so no
    # two jobs ever share a test.txt. tag carries the holdout level.
    src_by = {}
    for nh in args.n_holdouts:
        for sc in scenes:
            src = scene_source(sc, f"{out_tag}_nh{nh}")
            write_cam_split(src, nh)
            src_by[(nh, seg_name(sc))] = src

    # Flatten all jobs; seed results from any prior run (resume).
    cfgs = ([] if args.skip_baseline else ["baseline"]) + \
           ([] if args.skip_gsnet else ["gsnet"])
    data = {nh: {} for nh in args.n_holdouts}
    jobs = [(nh, sc, cfg) for nh in args.n_holdouts for sc in scenes for cfg in cfgs]

    gpu_q = queue.Queue()
    for _ in range(args.jobs_per_gpu):
        for g in args.gpus:
            gpu_q.put(g)
    lock = threading.Lock()

    def worker(nh, scene, cfg):
        gpu = gpu_q.get()
        try:
            name = seg_name(scene)
            source = src_by[(nh, name)]
            mp = os.path.join(args.out_dir, f"nh{nh}", name, cfg)
            if os.path.exists(os.path.join(mp, "results.json")):
                metrics = read_results(mp)  # resume: already done
            elif cfg == "baseline":
                metrics = optimize_eval(source, mp, args.iterations, [], gpu)
            else:
                os.makedirs(mp, exist_ok=True)
                init_ply = os.path.join(mp, "gsnet_init.ply")
                run([PY, "-m", "gsnet.infer", "--ckpt", args.ckpt,
                     "--sparse", sparse_points(scene), "--out", init_ply], gpu)
                metrics = optimize_eval(source, mp, args.iterations,
                                        ["--gsnet_init", init_ply], gpu)
            with lock:
                data[nh].setdefault(name, {})[cfg] = metrics
                table = aggregate(data, args)
            print(f"\n[done] nh{nh}/{name}/{cfg} gpu{gpu} PSNR={metrics['PSNR']:.2f}\n{table}",
                  flush=True)
        except Exception as e:
            print(f"[FAILED] nh{nh}/{seg_name(scene)}/{cfg}: {e}", flush=True)
        finally:
            gpu_q.put(gpu)

    with cf.ThreadPoolExecutor(max_workers=len(args.gpus) * args.jobs_per_gpu) as ex:
        for f in cf.as_completed([ex.submit(worker, *j) for j in jobs]):
            f.result()

    print("\n" + aggregate(data, args))
    print(f"\nResults -> {args.out_dir}/sparsity_sweep.{{json,md}}")


if __name__ == "__main__":
    main()
