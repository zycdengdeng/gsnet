#
# Same-Sensor Evaluation (SSE) driver.
#
# For each test sequence (default 110, 210, 310, 410, 510) runs:
#   * baseline 3DGS  : standard sparse-SfM init (create_from_pcd)
#   * GS-Net + 3DGS  : init from GS-Net predicted dense Gaussians (--gsnet_init)
# Each config: 3DGS optimization -> render test set -> compute PSNR/SSIM/LPIPS.
# All wall-clock times (GS-Net inference + 3DGS optimization) and metrics are
# recorded to <out_dir>/sse_results.json and printed as a Markdown table.
#
# Assumed layout:
#   sparse SfM   : <sparse_root>/S0<scene>/<id>_sparse.ply
#   test seq dir : <io_dir>/<id>_base   (images/ + sparse/0/)
#
# Usage:
#   python -m gsnet.run_sse \
#       --io_dir /mnt/zihanw/carla/input_output \
#       --sparse_root /mnt/zihanw/carla/sparse_point \
#       --ckpt runs/gsnet/gsnet_latest.pt \
#       --out_dir runs/sse
#

import argparse
import json
import os
import subprocess
import sys
import time

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, **kw):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, **kw)
    return time.time() - t0


def read_results(model_path):
    """Read PSNR/SSIM/LPIPS from a model's results.json (method ours_*)."""
    rp = os.path.join(model_path, "results.json")
    with open(rp) as f:
        d = json.load(f)
    method = sorted(d.keys())[-1]  # e.g. ours_30000
    return {k: d[method][k] for k in ("PSNR", "SSIM", "LPIPS")}


def optimize_and_eval(source, model_path, iterations, extra_args):
    """Run 3DGS optimization + render + metrics; return (metrics, optim_seconds)."""
    it = str(iterations)
    optim_s = run([PY, "train.py", "-s", source, "-m", model_path, "--eval",
                   "--iterations", it, "--test_iterations", it,
                   "--save_iterations", it, "--disable_viewer", "--quiet",
                   *extra_args])
    run([PY, "render.py", "-m", model_path, "--skip_train", "--quiet"])
    run([PY, "metrics.py", "-m", model_path])
    return read_results(model_path), optim_s


def process_sequence(sid, args):
    scene = int(sid) // 100
    base = os.path.join(args.io_dir, f"{sid}_base")
    sparse_ply = os.path.join(args.sparse_root, f"S{scene:02d}", f"{sid}_sparse.ply")
    assert os.path.isdir(base), f"missing test seq dir {base}"
    assert os.path.exists(sparse_ply), f"missing sparse SfM {sparse_ply}"

    # SSE split (idempotent).
    from gsnet.make_sse_split import write_split
    write_split(base, num_images=args.num_images, block=args.block,
                holdout=tuple(args.holdout))

    rec = {"id": sid, "scene": scene}

    if not args.skip_baseline:
        mp = os.path.join(args.out_dir, sid, "baseline")
        metrics, optim_s = optimize_and_eval(base, mp, args.iterations, [])
        rec["baseline"] = {**metrics, "optim_seconds": optim_s,
                           "total_seconds": optim_s}
        print(f"[{sid}] baseline: {metrics}  optim={optim_s:.1f}s")

    if not args.skip_gsnet:
        mp = os.path.join(args.out_dir, sid, "gsnet")
        os.makedirs(mp, exist_ok=True)
        init_ply = os.path.join(mp, "gsnet_init.ply")
        infer_s = run([PY, "-m", "gsnet.infer", "--ckpt", args.ckpt,
                       "--sparse", sparse_ply, "--out", init_ply])
        metrics, optim_s = optimize_and_eval(
            base, mp, args.iterations, ["--gsnet_init", init_ply])
        rec["gsnet"] = {**metrics, "infer_seconds": infer_s,
                        "optim_seconds": optim_s,
                        "total_seconds": infer_s + optim_s}
        print(f"[{sid}] gsnet: {metrics}  infer={infer_s:.1f}s optim={optim_s:.1f}s")

    return rec


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

    # Markdown table.
    lines = ["", "| Seq | Method | PSNR | SSIM | LPIPS | Optim(min) | Total(min) |",
             "|-----|--------|------|------|-------|------------|------------|"]
    for r in records:
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
    print(table)
    print(f"\nResults -> {out_dir}/sse_results.json , sse_results.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--sparse_root", required=True)
    ap.add_argument("--ckpt", default="runs/gsnet/gsnet_latest.pt")
    ap.add_argument("--out_dir", default="runs/sse")
    ap.add_argument("--test_ids", nargs="+", default=["110", "210", "310", "410", "510"])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--num_images", type=int, default=60)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--holdout", type=int, nargs="+", default=[4, 9])
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--skip_gsnet", action="store_true")
    args = ap.parse_args()

    records = []
    for sid in args.test_ids:
        records.append(process_sequence(sid, args))
        summarize(records, args.out_dir)  # checkpoint after each sequence


if __name__ == "__main__":
    main()
