#
# Encoder-design ablation driver (rebuttal).
#
# For each geometry-aware encoder variant -- mlp_only (a), concat/Ours (b),
# edgeconv (c), attention (d), geom (e) -- this:
#   Phase 1: trains GS-Net (variants distributed across --gpus, one per GPU),
#   Phase 2: runs the full SSE evaluation (GS-Net+3DGS on all 5 test sequences),
#   Phase 3: aggregates PSNR/SSIM/LPIPS, #params, train time, and per-scene
#            GS-Net inference time into a single comparison table.
#
# Everything except the encoder is held fixed (expansion head, losses, T, M,
# normalization, training schedule, SSE protocol).
#
# Usage:
#   python -m gsnet.run_encoder_ablation \
#       --corr_dir CORR/train \
#       --io_dir /mnt/zihanw/carla/input_output \
#       --sparse_root /mnt/zihanw/carla/sparse_point \
#       --out_dir runs/encoder_ablation --gpus 4 5 6 7
#

import argparse
import concurrent.futures as cf
import json
import os
import queue
import subprocess
import sys
import time

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_ENCODERS = ["mlp_only", "concat", "edgeconv", "attention", "geom"]
PRETTY = {"mlp_only": "(a) MLP-only", "concat": "(b) Concat-MLP (Ours)",
          "edgeconv": "(c) EdgeConv/DGCNN", "attention": "(d) Self-Attention",
          "geom": "(e) Explicit-Geometry"}


def run(cmd, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n[gpu{gpu}] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def count_params(enc):
    import torch  # noqa
    from gsnet.model import GSNet, GSNetConfig
    return sum(p.numel() for p in GSNet(GSNetConfig(encoder_type=enc)).parameters())


def train_all(args):
    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)

    def train_one(enc):
        gpu = gpu_q.get()
        try:
            out = os.path.join(args.out_dir, enc)
            run([PY, "-m", "gsnet.train_gsnet", "--corr_dir", args.corr_dir,
                 "--out_dir", out, "--encoder_type", enc,
                 "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
                 "--w_rot", str(args.w_rot)], gpu=gpu)
        finally:
            gpu_q.put(gpu)

    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        for f in cf.as_completed([ex.submit(train_one, e) for e in args.encoders]):
            f.result()


def eval_all(args):
    # Each run_sse saturates all GPUs internally; run variants sequentially.
    for enc in args.encoders:
        ckpt = os.path.join(args.out_dir, enc, "gsnet_latest.pt")
        out = os.path.join(args.out_dir, enc, "sse")
        run([PY, "-m", "gsnet.run_sse", "--io_dir", args.io_dir,
             "--sparse_root", args.sparse_root, "--ckpt", ckpt,
             "--out_dir", out, "--gpus", *[str(g) for g in args.gpus],
             "--skip_baseline", "--iterations", str(args.iterations)])


def aggregate(args):
    rows = []
    for enc in args.encoders:
        out = os.path.join(args.out_dir, enc)
        row = {"encoder": enc, "params_K": count_params(enc) / 1e3}
        tt = os.path.join(out, "train_times.json")
        if os.path.exists(tt):
            row["train_min"] = json.load(open(tt))["total_seconds"] / 60
        sj = os.path.join(out, "sse", "sse_results.json")
        if os.path.exists(sj):
            d = json.load(open(sj))
            g = d.get("averages", {}).get("gsnet", {})
            row.update({k: g.get(k) for k in ("PSNR", "SSIM", "LPIPS")})
            infers = [r["gsnet"]["infer_seconds"] for r in d.get("per_sequence", [])
                      if "gsnet" in r]
            row["infer_s"] = sum(infers) / len(infers) if infers else None
        rows.append(row)

    lines = ["", "| Encoder | #Params(K) | PSNR | SSIM | LPIPS | Infer(s) | Train(min) |",
             "|---------|-----------|------|------|-------|----------|------------|"]
    for r in rows:
        def f(k, fmt):
            v = r.get(k)
            return (fmt.format(v) if v is not None else "-")
        lines.append(
            f"| {PRETTY.get(r['encoder'], r['encoder'])} | {r['params_K']:.1f} | "
            f"{f('PSNR','{:.2f}')} | {f('SSIM','{:.3f}')} | {f('LPIPS','{:.3f}')} | "
            f"{f('infer_s','{:.1f}')} | {f('train_min','{:.1f}')} |")
    table = "\n".join(lines)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "encoder_ablation.md"), "w") as fp:
        fp.write(table + "\n")
    with open(os.path.join(args.out_dir, "encoder_ablation.json"), "w") as fp:
        json.dump(rows, fp, indent=2)
    print(table)
    print(f"\n-> {args.out_dir}/encoder_ablation.{{md,json}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_dir", required=True)
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--sparse_root", required=True)
    ap.add_argument("--out_dir", default="runs/encoder_ablation")
    ap.add_argument("--encoders", nargs="+", default=ALL_ENCODERS)
    ap.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--w_rot", type=float, default=1.0)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_train:
        train_all(args)
    if not args.skip_eval:
        eval_all(args)
    aggregate(args)
    print(f"\nEncoder ablation done in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
