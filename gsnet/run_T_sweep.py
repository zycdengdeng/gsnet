#
# Empirical T (densification factor) sweep -- do NOT trust the paper's T=5 claim
# blindly (single-run, CARLA-specific, under high variance). For each T rebuilds
# correspondences (K=T nearest G_dense), trains GS-Net (recipe
# geom:tanh:w_rot0.1:w_pos10), evaluates, reports PSNR vs T. T values run in
# parallel (one per GPU); correspondences built once up-front (CPU).
#
# NOTE: each T is a single (stochastic) training -> read the TREND, not <1dB diffs.
#
# CARLA (SSE on subset):
#   python -m gsnet.run_T_sweep --dataset carla \
#       --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point \
#       --out_dir runs/Tsweep_carla --Ts 3 5 8 12 16 --gpus 0 1 2 3 4 --eval_ids 110 310 510
#
# Waymo (5-cam; SSE or cross-sensor via --target_cams):
#   python -m gsnet.run_T_sweep --dataset waymo \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam \
#       --gdense_dir runs/waymo5_gdense \
#       --test_scenes <segA> <segB> [--target_cams cam3 cam4] \
#       --out_dir runs/Tsweep_waymo --Ts 3 5 8 12 16 --gpus 0 1 2 3 4
#

import argparse
import concurrent.futures as cf
import json
import os
import queue
import subprocess
import sys
import threading

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("\n" + (f"[gpu{gpu}] " if gpu is not None else "") + "$ "
          + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def corr_dir(args, T):
    return os.path.join(args.out_dir, "corr", f"T{T}")


def build_all_corr(args):
    for T in args.Ts:
        cd = corr_dir(args, T)
        if os.path.exists(os.path.join(cd, "build_times.json")):
            continue
        if args.dataset == "carla":
            run([PY, "-m", "gsnet.build_correspondences", "--batch",
                 "--io_dir", args.io_dir, "--sparse_root", args.sparse_root,
                 "--out_dir", cd, "--K", str(T), "--M", str(args.M),
                 "--workers", str(args.workers)])
        else:
            run([PY, "-m", "gsnet.waymo_corr", "--root", args.root,
                 "--gdense_dir", args.gdense_dir, "--out_dir", cd,
                 "--K", str(T), "--M", str(args.M), "--workers", str(args.workers),
                 "--test_scenes", *args.test_scenes])


def one_T(T, args, gpu):
    model = os.path.join(args.out_dir, "model", f"T{T}")
    sse = os.path.join(args.out_dir, "sse", f"T{T}")
    ckpt = os.path.join(model, "gsnet_latest.pt")
    if not os.path.exists(ckpt):
        run([PY, "-m", "gsnet.train_gsnet", "--corr_dir", corr_dir(args, T),
             "--out_dir", model, "--encoder_type", "geom", "--color_activation", "tanh",
             "--w_rot", "0.1", "--w_pos", "10", "--T", str(T), "--M", str(args.M),
             "--in_memory", "1", "--epochs", str(args.epochs)], gpu)
    if args.dataset == "carla":
        run([PY, "-m", "gsnet.run_sse", "--io_dir", args.io_dir,
             "--sparse_root", args.sparse_root, "--ckpt", ckpt, "--out_dir", sse,
             "--skip_baseline", "--iterations", str(args.iterations),
             "--test_ids", *args.eval_ids, "--gpus", str(gpu)], gpu)
        results = os.path.join(sse, "sse_results.json")
    else:
        cmd = [PY, "-m", "gsnet.waymo_sse", "--root", args.root,
               "--test_scenes", *args.test_scenes, "--ckpt", ckpt, "--out_dir", sse,
               "--skip_baseline", "--iterations", str(args.iterations), "--gpus", str(gpu)]
        if args.target_cams:
            cmd += ["--target_cams", *args.target_cams]
        if args.source_cams:
            cmd += ["--source_cams", *args.source_cams]
        run(cmd, gpu)
        results = os.path.join(sse, "sse_results.json")
    g = json.load(open(results))["averages"]["gsnet"]
    return {"T": T, **{k: g[k] for k in ("PSNR", "SSIM", "LPIPS")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["carla", "waymo"], default="carla")
    # carla
    ap.add_argument("--io_dir")
    ap.add_argument("--sparse_root")
    ap.add_argument("--eval_ids", nargs="+", default=["110", "310", "510"])
    # waymo
    ap.add_argument("--root")
    ap.add_argument("--gdense_dir")
    ap.add_argument("--test_scenes", nargs="+", default=[])
    ap.add_argument("--target_cams", nargs="+", default=[])
    ap.add_argument("--source_cams", nargs="+", default=[])
    # common
    ap.add_argument("--out_dir", default="runs/Tsweep")
    ap.add_argument("--Ts", type=int, nargs="+", default=[3, 5, 8, 12, 16])
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    build_all_corr(args)

    rows = []
    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)
    lock = threading.Lock()
    os.makedirs(args.out_dir, exist_ok=True)

    def worker(T):
        gpu = gpu_q.get()
        try:
            r = one_T(T, args, gpu)
            with lock:
                rows.append(r)
                rows.sort(key=lambda x: x["T"])
                json.dump(rows, open(os.path.join(args.out_dir, "Tsweep.json"), "w"), indent=2)
            print(f"[done] T={T} PSNR={r['PSNR']:.2f}", flush=True)
        except Exception as e:
            print(f"[FAILED] T={T}: {e}", flush=True)  # don't abort the whole sweep
        finally:
            gpu_q.put(gpu)

    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        for f in cf.as_completed([ex.submit(worker, T) for T in args.Ts]):
            f.result()

    mode = f"target_cams={args.target_cams}" if args.target_cams else "SSE"
    lines = ["", f"## T sweep ({args.dataset}, {mode}, geom:tanh:0.1:10, M={args.M}, @{args.iterations} iters)",
             "| T | PSNR | SSIM | LPIPS |", "|---|------|------|-------|"]
    for r in rows:
        lines.append(f"| {r['T']} | {r['PSNR']:.2f} | {r['SSIM']:.3f} | {r['LPIPS']:.3f} |")
    table = "\n".join(lines)
    open(os.path.join(args.out_dir, "Tsweep.md"), "w").write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
