#
# Train a GS-Net model on a Waymo corr dir, then auto-evaluate it on Waymo SSE.
# Convenience wrapper so train + eval run unattended as one command. Resume-safe
# (skips training if the ckpt already exists). Image-conditioning is detected
# automatically by train_gsnet from the corr; pass --image_feats so the SSE
# inference step computes per-point image features for the test scenes too.
#
# Usage (image-conditioned):
#   python -m gsnet.run_waymo_train_sse \
#       --corr_dir CORR/waymo5_img \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam \
#       --test_scenes 10275144660749673822_5755_561 15868625208244306149_4340_000 \
#       --out_dir runs/waymo5_imgcond --image_feats --gpus 0 1 2 3
#

import argparse
import json
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, gpus):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_dir", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--test_scenes", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    # training hyperparams (default = final recipe)
    ap.add_argument("--encoder_type", default="geom")
    ap.add_argument("--color_activation", default="tanh")
    ap.add_argument("--w_rot", default="0.1")
    ap.add_argument("--w_pos", default="10")
    ap.add_argument("--w_scale", default="1")
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None)
    # eval
    ap.add_argument("--image_feats", action="store_true",
                    help="image-conditioned model: pass --image_feats to SSE infer")
    ap.add_argument("--n_holdout", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--train_extra", default="",
                    help="passthrough to waymo_sse (e.g. \"--densify_until_iter 0\")")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Step 1: train (skip if ckpt exists) ---
    model = os.path.join(args.out_dir, "model")
    ckpt = os.path.join(model, "gsnet_latest.pt")
    if os.path.exists(ckpt):
        print(f"[skip train] {ckpt} exists")
    else:
        cmd = [PY, "-m", "gsnet.train_gsnet", "--corr_dir", args.corr_dir,
               "--out_dir", model, "--encoder_type", args.encoder_type,
               "--color_activation", args.color_activation,
               "--w_rot", args.w_rot, "--w_pos", args.w_pos, "--w_scale", args.w_scale,
               "--T", str(args.T), "--M", str(args.M), "--in_memory", "1",
               "--epochs", str(args.epochs)]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        sh(cmd, args.gpus[:1])                    # training uses one GPU

    # --- Step 2: SSE eval (baseline + gsnet) ---
    sse = os.path.join(args.out_dir, "sse")
    cmd = [PY, "-m", "gsnet.waymo_sse", "--root", args.root,
           "--test_scenes", *args.test_scenes, "--ckpt", ckpt, "--out_dir", sse,
           "--n_holdout", str(args.n_holdout), "--iterations", str(args.iterations),
           "--gpus", *[str(g) for g in args.gpus]]
    if args.image_feats:
        cmd.append("--image_feats")
    if args.train_extra:
        cmd += ["--train_extra", args.train_extra]
    sh(cmd, args.gpus)

    md = os.path.join(sse, "sse_results.md")
    if os.path.exists(md):
        print("\n===== RESULT =====\n" + open(md).read())
    print(f"\nResults -> {sse}/sse_results.{{json,md}}")


if __name__ == "__main__":
    main()
