#
# One-command Waymo pipeline: G_dense -> correspondences -> train GS-Net -> SSE.
# For the sparse-wide-baseline multi-scene experiment (and any new Waymo root).
# Each step is resumable (gdense skips done scenes; training skips if ckpt
# exists). Test scenes are held out of the corr (cross-scene).
#
# Usage:
#   python -m gsnet.run_waymo_pipeline \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam_wide20 \
#       --test_scenes 10275144660749673822_5755_561 15868625208244306149_4340_000 \
#       --out_root runs/wide20 --n_holdout 2 --gpus 0 1 2 3 4 5 6 7
#

import argparse
import json
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, gpus=None):
    env = os.environ.copy()
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--test_scenes", nargs="+", required=True)
    ap.add_argument("--out_root", required=True, help="dir for gdense/corr/model/sse")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--n_holdout", type=int, default=2, help="SSE frames held out per camera")
    ap.add_argument("--gdense_iter", type=int, default=30000)
    ap.add_argument("--gdense_train_extra", default="",
                    help="passthrough to gdense's train.py, e.g. "
                         "\"--densify_grad_threshold 0.0004\" (anti-OOM on sparse scenes)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--encoder_type", default="concat",
                    help="GS-Net encoder (paper uses concat)")
    ap.add_argument("--holdout_frames", type=int, nargs="+", default=None,
                    help="explicit per-camera SSE test frame indices (e.g. 4 9 = "
                         "CARLA-SSE scheme); overrides --n_holdout")
    args = ap.parse_args()
    o = args.out_root
    os.makedirs(o, exist_ok=True)
    gdense, corr, model, sse = (os.path.join(o, x) for x in ("gdense", "corr", "model", "sse"))

    # 1) G_dense (pseudo-GT) for ALL scenes (3DGS from MVS fused.ply, resumable)
    gd_cmd = [PY, "-m", "gsnet.waymo_gdense", "--root", args.root, "--out_dir", gdense,
              "--iterations", str(args.gdense_iter), "--gpus", *[str(g) for g in args.gpus]]
    if args.gdense_train_extra:
        gd_cmd += ["--train_extra", args.gdense_train_extra]
    sh(gd_cmd)

    # 2) sparse->dense correspondences for TRAINING scenes (test held out)
    sh([PY, "-m", "gsnet.waymo_corr", "--root", args.root, "--gdense_dir", gdense,
        "--out_dir", corr, "--test_scenes", *args.test_scenes,
        "--iterations", str(args.gdense_iter), "--workers", "8"])

    # 3) train GS-Net (final recipe)
    ckpt = os.path.join(model, "gsnet_latest.pt")
    if os.path.exists(ckpt):
        print(f"[skip train] {ckpt}")
    else:
        sh([PY, "-m", "gsnet.train_gsnet", "--corr_dir", corr, "--out_dir", model,
            "--encoder_type", args.encoder_type, "--color_activation", "tanh",
            "--w_rot", "0.1", "--w_pos", "10", "--T", "5", "--M", "3",
            "--in_memory", "1", "--epochs", str(args.epochs)], args.gpus[:1])

    # 4) SSE on the held-out test scenes — TWO variants so we come back to the
    #    full picture: (a) standard densify-on; (b) densify-off (preserves the
    #    GS-Net init from being washed out -> the variant most likely to show a
    #    positive in this sparse regime, per the CARLA-CSE finding).
    sse_off = sse + "_doff"
    hf = ["--holdout_frames", *[str(f) for f in args.holdout_frames]] if args.holdout_frames else []
    for out, ex in ((sse, []), (sse_off, ["--train_extra", "--densify_until_iter 0"])):
        sh([PY, "-m", "gsnet.waymo_sse", "--root", args.root, "--test_scenes", *args.test_scenes,
            "--ckpt", ckpt, "--out_dir", out, "--n_holdout", str(args.n_holdout),
            "--iterations", str(args.iterations), "--gpus", *[str(g) for g in args.gpus], *hf, *ex])

    print("\n=================== FINAL Waymo SSE (sparse wide-baseline) ===================")
    for label, out in (("densify-ON ", sse), ("densify-OFF", sse_off)):
        md = os.path.join(out, "sse_results.md")
        if os.path.exists(md):
            print(f"\n----- {label} ({out}) -----\n" + open(md).read())
    print(f"\nDone. Results -> {sse}/sse_results.md  and  {sse_off}/sse_results.md")


if __name__ == "__main__":
    main()
