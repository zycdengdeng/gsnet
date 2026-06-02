#
# Leave-One-Scene-Out (LOSO) CROSS-SCENE CARLA SSE.
#
# Why: the default CARLA protocol trains GS-Net on segments 101-109 and tests on
# 110 -- i.e. the SAME scene/drive as 9 of its training segments (within-scene,
# near in-distribution). Waymo, by contrast, tests on entirely held-out scenes
# (cross-scene). That mismatch can by itself explain GS-Net winning on CARLA but
# not Waymo. This driver re-runs CARLA the FAIR way: to test segment {s}10, it
# trains GS-Net with ALL of scene s excluded (101..109 AND 110), so the test
# scene is genuinely unseen -- matching Waymo. Compare the resulting gap to the
# original within-scene CARLA win to see how much of it was scene familiarity.
#
# Cheap because correspondences are per-sequence (.npz): a fold just trains on
# the subset of existing .npz whose scene != held-out scene (no corr rebuild).
#
# Phase 1: train one GS-Net per held-out scene (folds run across --gpus).
# Phase 2: eval baseline + gsnet(fold model) per test segment (SSE frame holdout)
#          packed across --gpus x --jobs_per_gpu.
#
# Usage:
#   python -m gsnet.run_carla_loso \
#       --corr_dir CORR/train \
#       --io_dir /mnt/zihanw/carla/input_output \
#       --sparse_root /mnt/zihanw/carla/sparse_point \
#       --out_dir runs/carla_loso --test_ids 110 210 310 410 510 \
#       --T 5 --M 3 --gpus 0 1 2 3 4 5 --jobs_per_gpu 2
#

import argparse
import concurrent.futures as cf
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time

from gsnet.make_sse_split import write_split

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
    return read_results(mp)


def scene_of(sid):
    return int(sid) // 100


def build_fold_corr(corr_dir, out_dir, held_scene):
    """Symlink every CORR/<sid>.npz whose scene != held_scene into a fold dir."""
    fold = os.path.join(out_dir, f"fold_s{held_scene}", "corr")
    os.makedirs(fold, exist_ok=True)
    kept = 0
    for f in sorted(glob.glob(os.path.join(corr_dir, "*.npz"))):
        sid = os.path.splitext(os.path.basename(f))[0]
        if not sid.isdigit() or scene_of(sid) == held_scene:
            continue
        link = os.path.join(fold, os.path.basename(f))
        if not os.path.exists(link):
            os.symlink(os.path.abspath(f), link)
        kept += 1
    if kept == 0:
        raise RuntimeError(f"fold s{held_scene}: no training corr left "
                           f"(check --corr_dir {corr_dir})")
    print(f"[fold s{held_scene}] {kept} training sequences (scene {held_scene} excluded)")
    return fold


def train_fold(args, held_scene, gpu):
    model = os.path.join(args.out_dir, f"fold_s{held_scene}", "model")
    ckpt = os.path.join(model, "gsnet_latest.pt")
    if os.path.exists(ckpt):
        return ckpt
    fold_corr = build_fold_corr(args.corr_dir, args.out_dir, held_scene)
    run([PY, "-m", "gsnet.train_gsnet", "--corr_dir", fold_corr, "--out_dir", model,
         "--encoder_type", "geom", "--color_activation", "tanh",
         "--w_rot", "0.1", "--w_pos", "10", "--T", str(args.T), "--M", str(args.M),
         "--in_memory", "1", "--epochs", str(args.epochs)], gpu)
    return ckpt


def aggregate(data, args):
    """data[tid] = {'baseline':m, 'gsnet':m} -> md comparison table."""
    rows = sorted(data.items())
    def avg(cfg, k):
        v = [m[cfg][k] for _, m in rows if cfg in m]
        return sum(v) / len(v) if v else None
    summary = {"per_seq": data,
               "averages": {c: {k: avg(c, k) for k in ("PSNR", "SSIM", "LPIPS")}
                            for c in ("baseline", "gsnet")}}
    json.dump(summary, open(os.path.join(args.out_dir, "loso_results.json"), "w"), indent=2)

    lines = ["", "## CROSS-SCENE CARLA SSE (leave-one-scene-out) "
                 f"-- GS-Net never saw the test scene  @{args.iterations} iters",
             "| Seq | baseline | gsnet (cross-scene) | Δ(gsnet-base) |",
             "|-----|----------|---------------------|---------------|"]
    for tid, m in rows:
        b = m.get("baseline", {}).get("PSNR")
        g = m.get("gsnet", {}).get("PSNR")
        d = (g - b) if (b is not None and g is not None) else None
        lines.append(f"| {tid} | {b:.2f} | {g:.2f} | {d:+.2f} |"
                     if d is not None else
                     f"| {tid} | {b if b is None else f'{b:.2f}'} | "
                     f"{g if g is None else f'{g:.2f}'} | - |")
    ab, ag = avg("baseline", "PSNR"), avg("gsnet", "PSNR")
    if ab is not None and ag is not None:
        lines.append(f"| **Avg** | **{ab:.2f}** | **{ag:.2f}** | **{ag-ab:+.2f}** |")
    table = "\n".join(lines)
    open(os.path.join(args.out_dir, "loso_results.md"), "w").write(table + "\n")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_dir", default="CORR/train",
                    help="per-sequence correspondence .npz (built with K=T)")
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--sparse_root", required=True)
    ap.add_argument("--out_dir", default="runs/carla_loso")
    ap.add_argument("--test_ids", nargs="+", default=["110", "210", "310", "410", "510"])
    ap.add_argument("--T", type=int, default=5, help="must match K used to build --corr_dir")
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--num_images", type=int, default=60)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--holdout", type=int, nargs="+", default=[4, 9])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--jobs_per_gpu", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # SSE splits up-front (one per test seq dir; baseline & gsnet share it).
    for tid in args.test_ids:
        base = os.path.join(args.io_dir, f"{tid}_base")
        assert os.path.isdir(base), f"missing test seq dir {base}"
        write_split(base, num_images=args.num_images, block=args.block,
                    holdout=tuple(args.holdout))

    gpu_q = queue.Queue()
    for _ in range(args.jobs_per_gpu):
        for g in args.gpus:
            gpu_q.put(g)

    # Phase 1: train one LOSO model per held-out scene (dedup scenes).
    held_scenes = sorted({scene_of(t) for t in args.test_ids})
    ckpts = {}
    cl = threading.Lock()

    def train_worker(s):
        gpu = gpu_q.get()
        try:
            ckpts[s] = train_fold(args, s, gpu)
            print(f"[trained] fold s{s} -> {ckpts[s]}", flush=True)
        finally:
            gpu_q.put(gpu)

    with cf.ThreadPoolExecutor(max_workers=len(args.gpus) * args.jobs_per_gpu) as ex:
        for f in cf.as_completed([ex.submit(train_worker, s) for s in held_scenes]):
            f.result()

    # Phase 2: eval baseline + gsnet(fold) for every test segment.
    data = {}
    rp = os.path.join(args.out_dir, "loso_results.json")
    if os.path.exists(rp):
        data = json.load(open(rp)).get("per_seq", {})
    lock = threading.Lock()
    jobs = [(tid, cfg) for tid in args.test_ids for cfg in ("baseline", "gsnet")]

    def eval_worker(tid, cfg):
        gpu = gpu_q.get()
        try:
            base = os.path.join(args.io_dir, f"{tid}_base")
            mp = os.path.join(args.out_dir, tid, cfg)
            if os.path.exists(os.path.join(mp, "results.json")):
                metrics = read_results(mp)
            elif cfg == "baseline":
                metrics = optimize_eval(base, mp, args.iterations, [], gpu)
            else:
                os.makedirs(mp, exist_ok=True)
                scene = scene_of(tid)
                sparse_ply = os.path.join(args.sparse_root, f"S{scene:02d}",
                                          f"{tid}_sparse.ply")
                init_ply = os.path.join(mp, "gsnet_init.ply")
                run([PY, "-m", "gsnet.infer", "--ckpt", ckpts[scene],
                     "--sparse", sparse_ply, "--out", init_ply], gpu)
                metrics = optimize_eval(base, mp, args.iterations,
                                        ["--gsnet_init", init_ply], gpu)
            with lock:
                data.setdefault(tid, {})[cfg] = metrics
                table = aggregate(data, args)
            print(f"\n[done] {tid}/{cfg} gpu{gpu} PSNR={metrics['PSNR']:.2f}\n{table}",
                  flush=True)
        except Exception as e:
            print(f"[FAILED] {tid}/{cfg}: {e}", flush=True)
        finally:
            gpu_q.put(gpu)

    with cf.ThreadPoolExecutor(max_workers=len(args.gpus) * args.jobs_per_gpu) as ex:
        for f in cf.as_completed([ex.submit(eval_worker, *j) for j in jobs]):
            f.result()

    print("\n" + aggregate(data, args))
    print(f"\nResults -> {args.out_dir}/loso_results.{{json,md}}")


if __name__ == "__main__":
    main()
