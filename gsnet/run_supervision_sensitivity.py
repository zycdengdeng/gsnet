#
# Supervision-quality sensitivity study (rebuttal): how sensitive is GS-Net to
# the quality of the pseudo-GT dense Gaussians it is trained on?
#
#   Axis A (optimized-Gaussian quality): supervise from G_dense extracted at
#           different 3DGS optimization iterations (--iters).
#   Axis B (pseudo-GT density / MVS coverage): supervise from randomly
#           subsampled G_dense (--subsamples).
#
# For each level: build correspondences -> train GS-Net -> SSE eval (GS-Net+3DGS
# on the 5 test sequences) -> record average PSNR/SSIM/LPIPS. Baselines are the
# same as the main SSE table (unaffected), so we only run the GS-Net config.
#
# Usage:
#   python -m gsnet.run_supervision_sensitivity \
#       --io_dir /mnt/zihanw/carla/input_output \
#       --sparse_root /mnt/zihanw/carla/sparse_point \
#       --out_dir runs/sup_sens --gpus 2 3 4 5 6 7 \
#       --iters 5000 10000 20000 30000 --subsamples 1.0 0.5 0.25 0.1
#

import argparse
import json
import os
import subprocess
import sys
import time

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def one_level(label, gdense_iter, subsample, args):
    corr = os.path.join(args.out_dir, "corr", label)
    model = os.path.join(args.out_dir, "model", label)
    sse = os.path.join(args.out_dir, "sse", label)
    t0 = time.time()

    if not os.path.exists(os.path.join(corr, "build_times.json")):
        run([PY, "-m", "gsnet.build_correspondences", "--batch",
             "--io_dir", args.io_dir, "--sparse_root", args.sparse_root,
             "--out_dir", corr, "--workers", str(args.workers),
             "--gdense_iter", str(gdense_iter),
             "--gdense_subsample", str(subsample)])
    if not os.path.exists(os.path.join(model, "gsnet_latest.pt")):
        run([PY, "-m", "gsnet.train_gsnet", "--corr_dir", corr,
             "--out_dir", model, "--epochs", str(args.epochs),
             "--encoder_type", args.encoder_type,
             "--color_activation", args.color_activation,
             "--w_rot", str(args.w_rot), "--w_pos", str(args.w_pos)], gpu=args.gpus[0])
    sse_cmd = [PY, "-m", "gsnet.run_sse", "--io_dir", args.io_dir,
               "--sparse_root", args.sparse_root,
               "--ckpt", os.path.join(model, "gsnet_latest.pt"),
               "--out_dir", sse, "--gpus", *[str(g) for g in args.gpus]]
    if not args.eval_baseline:                  # baseline is constant across levels
        sse_cmd.append("--skip_baseline")
    run(sse_cmd)

    g = json.load(open(os.path.join(sse, "sse_results.json")))["averages"]["gsnet"]
    return {"label": label, "gdense_iter": gdense_iter, "subsample": subsample,
            "PSNR": g["PSNR"], "SSIM": g["SSIM"], "LPIPS": g["LPIPS"],
            "minutes": (time.time() - t0) / 60}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--sparse_root", required=True)
    ap.add_argument("--out_dir", default="runs/sup_sens")
    ap.add_argument("--gpus", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--iters", type=int, nargs="+", default=[5000, 10000, 20000, 30000])
    ap.add_argument("--subsamples", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_baseline", action="store_true",
                    help="also evaluate the SfM-init baseline (fills ΔPSNR; baseline is "
                         "constant across levels, so running it on one level is enough)")
    # Recipe knobs: defaults reproduce the original sup_sens runs (default recipe:
    # concat, sigmoid, equal weights). For consistency with the final-recipe main
    # table, pass --color_activation tanh --w_rot 0.1 --w_pos 10.
    ap.add_argument("--encoder_type", default="concat")
    ap.add_argument("--color_activation", default="sigmoid", choices=["sigmoid", "tanh"])
    ap.add_argument("--w_rot", type=float, default=1.0)
    ap.add_argument("--w_pos", type=float, default=1.0)
    args = ap.parse_args()

    levels = ([(f"iter{it}", it, 1.0) for it in args.iters]
              + [(f"sub{ss:.2f}", 30000, ss) for ss in args.subsamples])
    rows = []
    for label, it, ss in levels:
        rows.append(one_level(label, it, ss, args))
        # checkpoint table after each level
        os.makedirs(args.out_dir, exist_ok=True)
        json.dump(rows, open(os.path.join(args.out_dir, "sensitivity.json"), "w"), indent=2)

    lines = ["", "## Axis A: supervision from 3DGS @ iteration",
             "| Supervision iter | PSNR | SSIM | LPIPS |",
             "|------------------|------|------|-------|"]
    for r in rows:
        if r["subsample"] == 1.0 and r["label"].startswith("iter"):
            lines.append(f"| {r['gdense_iter']} | {r['PSNR']:.2f} | {r['SSIM']:.3f} | {r['LPIPS']:.3f} |")
    lines += ["", "## Axis B: pseudo-GT density (subsample of G_dense @ 30k)",
              "| Keep fraction | PSNR | SSIM | LPIPS |",
              "|---------------|------|------|-------|"]
    for r in rows:
        if r["label"].startswith("sub"):
            lines.append(f"| {r['subsample']:.2f} | {r['PSNR']:.2f} | {r['SSIM']:.3f} | {r['LPIPS']:.3f} |")
    table = "\n".join(lines)
    open(os.path.join(args.out_dir, "sensitivity.md"), "w").write(table + "\n")
    print(table)
    print(f"\n-> {args.out_dir}/sensitivity.{{md,json}}")


if __name__ == "__main__":
    main()
