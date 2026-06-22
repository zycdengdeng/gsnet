#
# Direct MVS-quality sensitivity study (rebuttal, Reviewer C).
#
# Axis A (supervision_sensitivity) varies the OPTIMIZED-Gaussian convergence;
# Axis B varies pseudo-GT DENSITY by subsampling G_dense. Neither directly
# corrupts the MVS *reconstruction* that initializes G_dense. This driver closes
# that gap: it degrades the MVS dense point cloud itself, RE-OPTIMIZES G_dense
# from the degraded MVS (full 30k, so the convergence axis is held fixed), then
# rebuilds correspondences -> trains GS-Net -> SSE eval.
#
# A "degradation level" = (keep fraction, xyz noise as a fraction of scene scale):
#   clean       keep=1.00 noise=0.000   (reference; reuses the original G_dense)
#   drop50      keep=0.50 noise=0.000   (half the MVS points)
#   noise01     keep=1.00 noise=0.010   (1% scene-scale Gaussian jitter)
#   drop25_n02  keep=0.25 noise=0.020   (combined heavy degradation)
#
# Pipeline per level (resumable, GPU-fault-tolerant for the heavy G_dense step):
#   1) degrade MVS ply  -> mvs/<level>/<id>.ply
#   2) re-optimize G_dense (3DGS 30k, --init_pcd degraded) -> gdense/<level>/<id>
#   3) build corr from degraded G_dense -> corr/<level>/<id>.npz
#   4) train GS-Net -> model/<level>
#   5) SSE eval (baseline + gsnet, 5 test seqs) -> sse/<level>
#
# Read results with the shared re-aggregator (drops degenerate scene 310):
#   python -m gsnet.reaggregate --root runs/mvs_degrade --exclude 310
# -> one row per level (gsnet PSNR/SSIM/LPIPS + baseline), saved to a .md.
#
# Usage:
#   nohup python -m gsnet.run_mvs_degradation \
#       --io_dir /mnt/zihanw/carla/input_output \
#       --sparse_root /mnt/zihanw/carla/sparse_point \
#       --out_dir runs/mvs_degrade --gpus 0 1 2 3 4 5 6 7 \
#       > runs/mvs_degrade.log 2>&1 &
#
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsnet.common import read_points_any, compute_normalization
from gsnet.io import save_init_pcd_ply
from gsnet.build_correspondences import build_for_sequence, discover_training_sequences
from gsnet.gpu_pool import run_jobs

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (label, keep_fraction, noise_fraction_of_scene_scale)
LEVELS = [("clean", 1.0, 0.0), ("drop50", 0.5, 0.0),
          ("noise01", 1.0, 0.01), ("drop25_n02", 0.25, 0.02)]


def on_gpu(cmd, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[gpu{gpu}] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def sh(cmd, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def degrade_mvs(mvs_ply, out_ply, keep, noise, sid):
    """Read MVS dense points, randomly drop to `keep`, add Gaussian xyz jitter of
    `noise` x scene-scale, write a fetchPly-compatible ply. Reproducible per id."""
    xyz, rgb = read_points_any(mvs_ply)
    rng = np.random.default_rng(abs(hash(f"{sid}|{keep}|{noise}")) % (2**32))
    n0 = xyz.shape[0]
    if keep < 1.0:
        idx = rng.choice(n0, int(round(n0 * keep)), replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    if noise > 0.0:
        _, scale = compute_normalization(xyz)
        xyz = xyz + rng.normal(0.0, noise * scale, size=xyz.shape).astype(np.float32)
    save_init_pcd_ply(out_ply, xyz, rgb)
    return n0, xyz.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--sparse_root", required=True)
    ap.add_argument("--out_dir", default="runs/mvs_degrade")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--levels", nargs="+", default=[lab for lab, _, _ in LEVELS],
                    help="which levels to run (subset of clean/drop50/noise01/drop25_n02)")
    ap.add_argument("--iterations", type=int, default=30000,
                    help="G_dense 3DGS iterations (keep 30k to hold the convergence axis fixed)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--min_free_mb", type=int, default=12000)
    ap.add_argument("--max_retries", type=int, default=5)
    args = ap.parse_args()

    levmap = {lab: (k, n) for lab, k, n in LEVELS}
    levels = [(lab, *levmap[lab]) for lab in args.levels]
    o = args.out_dir
    os.makedirs(o, exist_ok=True)
    pool = dict(min_free_mb=args.min_free_mb, max_retries=args.max_retries)
    lock = threading.Lock()

    seqs = discover_training_sequences(args.io_dir, args.sparse_root, args.iterations)
    print(f"[mvs] {len(seqs)} training sequences; levels={[l[0] for l in levels]}", flush=True)

    def orig_gdense(sid):
        return os.path.join(args.io_dir, f"output_{sid}_dense", "point_cloud",
                            f"iteration_{args.iterations}", "point_cloud.ply")

    def lvl_gdense(label, sid):
        return os.path.join(o, "gdense", label, sid, "point_cloud",
                            f"iteration_{args.iterations}", "point_cloud.ply")

    for label, keep, noise in levels:
        print(f"\n===== level {label} (keep={keep}, noise={noise}) =====", flush=True)

        # ---- 1+2) degrade MVS -> re-optimize G_dense (skipped for clean) ----
        if label != "clean":
            def gd_fn(item, gpu):
                sid, sparse, _ = item
                if os.path.exists(lvl_gdense(label, sid)):
                    return
                mvs = os.path.join(args.io_dir, f"{sid}_dense", "sparse", "0", "points3D.ply")
                scene = os.path.join(args.io_dir, f"{sid}_dense")
                assert os.path.exists(mvs), f"missing MVS ply {mvs}"
                dply = os.path.join(o, "mvs", label, f"{sid}.ply")
                n0, n1 = degrade_mvs(mvs, dply, keep, noise, sid)
                print(f"[degrade {label}] {sid}: {n0} -> {n1}", flush=True)
                mp = os.path.dirname(os.path.dirname(os.path.dirname(lvl_gdense(label, sid))))
                on_gpu([PY, "train.py", "-s", scene, "-m", mp, "--init_pcd", dply,
                        "--iterations", str(args.iterations),
                        "--save_iterations", str(args.iterations),
                        "--test_iterations", str(args.iterations),
                        "--disable_viewer", "--quiet"], gpu)
            todo = [s for s in seqs if not os.path.exists(lvl_gdense(label, s[0]))]
            print(f"[mvs] {label} gdense: {len(todo)}/{len(seqs)} to (re)build", flush=True)
            if todo:
                _, f = run_jobs(todo, args.gpus, gd_fn, label=f"gdense-{label}", **pool)
                if f:
                    print(f"[mvs] WARN {label} gdense failed: {[x[0] for x in f]}", flush=True)

        # ---- 3) correspondences from this level's G_dense ----
        corr = os.path.join(o, "corr", label)
        if not os.path.exists(os.path.join(corr, "build_times.json")):
            os.makedirs(corr, exist_ok=True)
            stats = []
            for sid, sparse, _ in seqs:
                gd = orig_gdense(sid) if label == "clean" else lvl_gdense(label, sid)
                if not os.path.exists(gd):
                    print(f"[corr {label}] skip {sid}: missing G_dense", flush=True)
                    continue
                stats.append(build_for_sequence(sparse, gd,
                             os.path.join(corr, f"{sid}.npz"), K=5, M=3))
            json.dump({"sequences": stats}, open(os.path.join(corr, "build_times.json"), "w"),
                      indent=2)
        else:
            print(f"[skip corr] {corr}", flush=True)

        # ---- 4) train GS-Net (final recipe) ----
        model = os.path.join(o, "model", label)
        ckpt = os.path.join(model, "gsnet_latest.pt")
        if not os.path.exists(ckpt):
            def tr_fn(_, gpu):
                on_gpu([PY, "-m", "gsnet.train_gsnet", "--corr_dir", corr, "--out_dir", model,
                        "--encoder_type", "concat", "--color_activation", "tanh",
                        "--w_rot", "0.1", "--w_pos", "10", "--M", "3",
                        "--in_memory", "1", "--epochs", str(args.epochs)], gpu)
            run_jobs(["train"], args.gpus, tr_fn, label=f"train-{label}", **pool)
            assert os.path.exists(ckpt), f"training produced no ckpt for {label}"
        else:
            print(f"[skip train] {ckpt}", flush=True)

        # ---- 5) SSE eval (baseline + gsnet) ----
        sse = os.path.join(o, "sse", label)
        if not os.path.exists(os.path.join(sse, "sse_results.json")):
            sh([PY, "-m", "gsnet.run_sse", "--io_dir", args.io_dir,
                "--sparse_root", args.sparse_root, "--ckpt", ckpt,
                "--out_dir", sse, "--gpus", *[str(g) for g in args.gpus]])
        else:
            print(f"[skip sse] {sse}", flush=True)

        try:
            g = json.load(open(os.path.join(sse, "sse_results.json")))["averages"]["gsnet"]
            print(f"[mvs] {label} DONE: gsnet PSNR={g['PSNR']:.2f} (incl. 310; "
                  f"use reaggregate --exclude 310 for the clean number)", flush=True)
        except Exception:
            pass

    print(f"\n[mvs] all levels done. Aggregate with:\n"
          f"  python -m gsnet.reaggregate --root {o} --exclude 310\n", flush=True)


if __name__ == "__main__":
    main()
