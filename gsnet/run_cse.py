#
# Cross-Sensor Evaluation (CSE) driver. On each test sequence's CSE scene
# (built by make_cse_scene.py: 60 odd train + 60 even test), compares:
#   * baseline 3DGS  : sparse-SfM init
#   * GS-Net + 3DGS  : GS-Net init from the odd sparse points
# Reconstruction uses all 60 odd cameras; evaluation is on the 60 even cameras
# (positions absent during reconstruction). Multi-GPU, merge/resume.
#
# Usage:
#   python -m gsnet.run_cse \
#       --scenes_dir runs/cse_scenes --sparse_root /mnt/zihanw/carla/sparse_point \
#       --ckpt runs/encoder_ablation/concat/gsnet_latest.pt \
#       --out_dir runs/cse --gpus 2 3 4 5 6 7
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
    return read_results(mp), s


def summarize(records, out_dir):
    def avg(cfg, k):
        v = [r[cfg][k] for r in records if cfg in r]
        return sum(v) / len(v) if v else float("nan")
    summary = {"per_sequence": records, "averages": {}}
    for cfg in ("baseline", "gsnet"):
        if any(cfg in r for r in records):
            summary["averages"][cfg] = {k: avg(cfg, k) for k in
                                        ("PSNR", "SSIM", "LPIPS", "optim_seconds")}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(out_dir, "cse_results.json"), "w"), indent=2)
    lines = ["", "| Seq | Method | PSNR | SSIM | LPIPS | Optim(min) |",
             "|-----|--------|------|------|-------|------------|"]
    for r in sorted(records, key=lambda x: x["id"]):
        for cfg in ("baseline", "gsnet"):
            if cfg in r:
                m = r[cfg]
                lines.append(f"| {r['id']} | {cfg} | {m['PSNR']:.2f} | {m['SSIM']:.3f} "
                             f"| {m['LPIPS']:.3f} | {m['optim_seconds']/60:.1f} |")
    for cfg in ("baseline", "gsnet"):
        if cfg in summary["averages"]:
            a = summary["averages"][cfg]
            lines.append(f"| **Avg** | **{cfg}** | **{a['PSNR']:.2f}** | "
                         f"**{a['SSIM']:.3f}** | **{a['LPIPS']:.3f}** | "
                         f"{a['optim_seconds']/60:.1f} |")
    table = "\n".join(lines)
    open(os.path.join(out_dir, "cse_results.md"), "w").write(table + "\n")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_dir", default="runs/cse_scenes")
    ap.add_argument("--sparse_root", required=True)
    ap.add_argument("--ckpt", default="runs/encoder_ablation/concat/gsnet_latest.pt")
    ap.add_argument("--out_dir", default="runs/cse")
    ap.add_argument("--ids", nargs="+", default=["110", "210", "310", "410", "510"])
    ap.add_argument("--gpus", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--skip_gsnet", action="store_true")
    args = ap.parse_args()

    jobs = []
    for sid in args.ids:
        if not args.skip_baseline:
            jobs.append((sid, "baseline"))
        if not args.skip_gsnet:
            jobs.append((sid, "gsnet"))

    records = {}
    rp = os.path.join(args.out_dir, "cse_results.json")
    if os.path.exists(rp):
        for r in json.load(open(rp)).get("per_sequence", []):
            records[r["id"]] = r
    lock = threading.Lock()
    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)

    def worker(sid, cfg):
        gpu = gpu_q.get()
        try:
            source = os.path.join(args.scenes_dir, sid)
            scene = int(sid) // 100
            sparse_ply = os.path.join(args.sparse_root, f"S{scene:02d}", f"{sid}_sparse.ply")
            if cfg == "baseline":
                mp = os.path.join(args.out_dir, sid, "baseline")
                metrics, s = optimize_eval(source, mp, args.iterations, [], gpu)
                res = {**metrics, "optim_seconds": s, "total_seconds": s}
            else:
                mp = os.path.join(args.out_dir, sid, "gsnet")
                os.makedirs(mp, exist_ok=True)
                init_ply = os.path.join(mp, "gsnet_init.ply")
                infer_s = run([PY, "-m", "gsnet.infer", "--ckpt", args.ckpt,
                               "--sparse", sparse_ply, "--out", init_ply], gpu)
                metrics, s = optimize_eval(source, mp, args.iterations,
                                           ["--gsnet_init", init_ply], gpu)
                res = {**metrics, "infer_seconds": infer_s, "optim_seconds": s,
                       "total_seconds": infer_s + s}
            with lock:
                records.setdefault(sid, {"id": sid})[cfg] = res
                table = summarize(list(records.values()), args.out_dir)
            print(f"\n[done] {sid}/{cfg} gpu{gpu} :: PSNR={res['PSNR']:.2f}\n{table}",
                  flush=True)
        finally:
            gpu_q.put(gpu)

    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        for f in cf.as_completed([ex.submit(worker, sid, cfg) for sid, cfg in jobs]):
            f.result()
    print("\n" + summarize(list(records.values()), args.out_dir))
    print(f"\nResults -> {args.out_dir}/cse_results.{{json,md}}")


if __name__ == "__main__":
    main()
