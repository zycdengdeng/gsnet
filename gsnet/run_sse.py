#
# Same-Sensor Evaluation (SSE) driver, multi-GPU.
#
# For each test sequence (default 110, 210, 310, 410, 510) runs two configs:
#   * baseline 3DGS  : standard sparse-SfM init (create_from_pcd)
#   * GS-Net + 3DGS  : init from GS-Net predicted dense Gaussians (--gsnet_init)
# Each config: 3DGS optimization -> render test set -> PSNR/SSIM/LPIPS.
#
# The (sequence x config) jobs are independent and are distributed across the
# given GPUs (one 3DGS run per GPU). All wall-clock times (GS-Net inference +
# 3DGS optimization) and metrics are recorded to <out_dir>/sse_results.json and
# printed as a Markdown table; the summary is refreshed as each job finishes.
#
# Assumed layout:
#   sparse SfM   : <sparse_root>/S0<scene>/<id>_sparse.ply
#   test seq dir : <io_dir>/<id>_base   (images/ + sparse/0/)
#
# Usage (4 A100s on cards 4,5,6,7):
#   python -m gsnet.run_sse \
#       --io_dir /mnt/zihanw/carla/input_output \
#       --sparse_root /mnt/zihanw/carla/sparse_point \
#       --ckpt runs/gsnet/gsnet_latest.pt \
#       --out_dir runs/sse --gpus 4 5 6 7
#

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tag = f"[gpu{gpu}] " if gpu is not None else ""
    print(f"\n{tag}$ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)
    return time.time() - t0


def read_results(model_path):
    with open(os.path.join(model_path, "results.json")) as f:
        d = json.load(f)
    method = sorted(d.keys())[-1]  # e.g. ours_30000
    return {k: d[method][k] for k in ("PSNR", "SSIM", "LPIPS")}


def optimize_and_eval(source, model_path, iterations, extra_args, gpu):
    it = str(iterations)
    optim_s = run([PY, "train.py", "-s", source, "-m", model_path, "--eval",
                   "--iterations", it, "--test_iterations", it,
                   "--save_iterations", it, "--disable_viewer", "--quiet",
                   *extra_args], gpu=gpu)
    run([PY, "render.py", "-m", model_path, "--skip_train", "--quiet"], gpu=gpu)
    run([PY, "metrics.py", "-m", model_path], gpu=gpu)
    return read_results(model_path), optim_s


def job_baseline(sid, args, gpu):
    base = os.path.join(args.io_dir, f"{sid}_base")
    mp = os.path.join(args.out_dir, sid, "baseline")
    metrics, optim_s = optimize_and_eval(base, mp, args.iterations, [], gpu)
    return {**metrics, "optim_seconds": optim_s, "total_seconds": optim_s}


def job_gsnet(sid, args, gpu):
    scene = int(sid) // 100
    base = os.path.join(args.io_dir, f"{sid}_base")
    sparse_ply = os.path.join(args.sparse_root, f"S{scene:02d}", f"{sid}_sparse.ply")
    mp = os.path.join(args.out_dir, sid, "gsnet")
    os.makedirs(mp, exist_ok=True)
    init_ply = os.path.join(mp, "gsnet_init.ply")
    infer_s = run([PY, "-m", "gsnet.infer", "--ckpt", args.ckpt,
                   "--sparse", sparse_ply, "--out", init_ply], gpu=gpu)
    metrics, optim_s = optimize_and_eval(
        base, mp, args.iterations, ["--gsnet_init", init_ply], gpu)
    return {**metrics, "infer_seconds": infer_s, "optim_seconds": optim_s,
            "total_seconds": infer_s + optim_s}


def summarize(records, out_dir):
    def avg(cfg, key):
        vals = [r[cfg][key] for r in records if cfg in r]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {"per_sequence": records, "averages": {}}
    for cfg in ("baseline", "gsnet"):
        if any(cfg in r for r in records):
            summary["averages"][cfg] = {
                k: avg(cfg, k) for k in ("PSNR", "SSIM", "LPIPS",
                                         "optim_seconds", "total_seconds")}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sse_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["", "| Seq | Method | PSNR | SSIM | LPIPS | Optim(min) | Total(min) |",
             "|-----|--------|------|------|-------|------------|------------|"]
    for r in sorted(records, key=lambda x: x["id"]):
        for cfg in ("baseline", "gsnet"):
            if cfg in r:
                m = r[cfg]
                lines.append(
                    f"| {r['id']} | {cfg} | {m['PSNR']:.2f} | {m['SSIM']:.3f} | "
                    f"{m['LPIPS']:.3f} | {m['optim_seconds']/60:.1f} | "
                    f"{m['total_seconds']/60:.1f} |")
    for cfg in ("baseline", "gsnet"):
        if cfg in summary["averages"]:
            a = summary["averages"][cfg]
            lines.append(
                f"| **Avg** | **{cfg}** | **{a['PSNR']:.2f}** | **{a['SSIM']:.3f}** | "
                f"**{a['LPIPS']:.3f}** | {a['optim_seconds']/60:.1f} | "
                f"{a['total_seconds']/60:.1f} |")
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "sse_results.md"), "w") as f:
        f.write(table + "\n")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--sparse_root", required=True)
    ap.add_argument("--ckpt", default="runs/gsnet/gsnet_latest.pt")
    ap.add_argument("--out_dir", default="runs/sse")
    ap.add_argument("--test_ids", nargs="+", default=["110", "210", "310", "410", "510"])
    ap.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7],
                    help="GPU ids to distribute jobs across")
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--num_images", type=int, default=60)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--holdout", type=int, nargs="+", default=[4, 9])
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--skip_gsnet", action="store_true")
    args = ap.parse_args()

    # Validate inputs and write SSE splits up-front (avoids concurrent writes).
    from gsnet.make_sse_split import write_split
    for sid in args.test_ids:
        base = os.path.join(args.io_dir, f"{sid}_base")
        assert os.path.isdir(base), f"missing test seq dir {base}"
        write_split(base, num_images=args.num_images, block=args.block,
                    holdout=tuple(args.holdout))

    # Build the job list.
    jobs = []
    for sid in args.test_ids:
        if not args.skip_baseline:
            jobs.append((sid, "baseline"))
        if not args.skip_gsnet:
            jobs.append((sid, "gsnet"))

    records = {}        # sid -> {"id","scene", "baseline":..., "gsnet":...}
    lock = threading.Lock()
    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)

    def worker(sid, cfg):
        gpu = gpu_q.get()
        try:
            t0 = time.time()
            res = (job_baseline if cfg == "baseline" else job_gsnet)(sid, args, gpu)
            with lock:
                rec = records.setdefault(sid, {"id": sid, "scene": int(sid) // 100})
                rec[cfg] = res
                table = summarize(list(records.values()), args.out_dir)
            print(f"\n[done] {sid}/{cfg} on gpu{gpu} in {time.time()-t0:.1f}s :: "
                  f"PSNR={res['PSNR']:.2f}\n{table}", flush=True)
        finally:
            gpu_q.put(gpu)

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        futs = [ex.submit(worker, sid, cfg) for sid, cfg in jobs]
        for f in cf.as_completed(futs):
            f.result()  # surface exceptions

    print("\n" + summarize(list(records.values()), args.out_dir))
    print(f"\nResults -> {args.out_dir}/sse_results.json , sse_results.md")


if __name__ == "__main__":
    main()
