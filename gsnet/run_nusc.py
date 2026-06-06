#
# nuScenes SSE — full hands-off pipeline with GPU-fault-tolerant scheduling.
#
# Flow:
#   1) G_dense  : 3DGS (30k, from MVS fused.ply) on every TRAIN clip -> GS-Net's
#                 supervision target (raw, --no_filter if asked). resumable.
#   2) corr     : sparse SfM points <-> G_dense, test clips excluded,
#                 --global_scale (metric) , --no_filter.
#   3) train    : GS-Net (encoder=concat, paper).
#   4) eval     : on each TEST clip, 3 inits x densify-sweep, CARLA-SSE frame
#                 holdout [4,9]:
#                   inits   = {sfm (baseline), mvs (fused.ply), gsnet}
#                             -> shows whether MVS-input > SfM, and where GS-Net lands
#                   densify = {on(default), d2000, off}
#                             -> shows whether full densification washes out the
#                                init advantage (CARLA-CSE sweet-spot question)
#   5) aggregate-> nusc_eval.md (per-clip + averages, per init x densify).
#
# Every multi-GPU step uses gpu_pool: prefers free cards, squeezes onto low-use
# ones, and on OOM/failure retries the job on another free card instead of
# aborting. Fully resumable (re-run to continue).
#
# Usage:
#   nohup python -m gsnet.run_nusc \
#       --root /mnt/zihanw/gsnet_nusc \
#       --test_scenes 348_clip_09 332_clip_09 331_clip_09 299_clip_09 325_clip_09 \
#       --out_root runs/nusc --gpus 0 1 2 3 4 5 6 7 \
#       --global_scale 45 --no_filter --min_free_mb 12000 \
#       > runs/nusc.log 2>&1 &
#
import argparse
import json
import os
import subprocess
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsnet.waymo import (discover_scenes, seg_name, dense_dir, fused_ply,
                         sparse_points, scene_source, resolve_scene)
from gsnet.make_cam_split import write_cam_split
from gsnet.gpu_pool import run_jobs

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INITS = ["sfm", "mvs", "gsnet"]
DENSIFY = [("on", []),
           ("d2000", ["--densify_until_iter", "2000"]),
           ("off", ["--densify_until_iter", "0"])]
HOLDOUT = [4, 9]                      # CARLA-SSE per-camera test frames


def on_gpu(cmd, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[gpu{gpu}] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env,
                   stdout=subprocess.DEVNULL if "--quiet" in cmd else None)


def sh(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO)


def read_psnr(mp):
    d = json.load(open(os.path.join(mp, "results.json")))
    return d[sorted(d)[-1]]["PSNR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--test_scenes", nargs="+", required=True)
    ap.add_argument("--out_root", default="runs/nusc")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--encoder_type", default="concat")
    ap.add_argument("--global_scale", type=float, default=45.0)
    ap.add_argument("--no_filter", action="store_true")
    ap.add_argument("--min_free_mb", type=int, default=12000,
                    help="only schedule a job on a card with >= this much free VRAM")
    ap.add_argument("--max_retries", type=int, default=5)
    args = ap.parse_args()

    o = args.out_root
    gdense, corr, model, ev = (os.path.join(o, x) for x in ("gdense", "corr", "model", "eval"))
    os.makedirs(o, exist_ok=True)
    ckpt = os.path.join(model, "gsnet_latest.pt")
    it = str(args.iterations)
    pool = dict(min_free_mb=args.min_free_mb, max_retries=args.max_retries)

    all_clips = discover_scenes(args.root)
    test_set = {seg_name(c) for c in all_clips
                if any(t in seg_name(c) for t in args.test_scenes)}
    train_clips = [c for c in all_clips if seg_name(c) not in test_set]
    test_clips = [resolve_scene(args.root, t) for t in args.test_scenes]
    print(f"[nusc] {len(all_clips)} clips: {len(train_clips)} train, {len(test_clips)} test={sorted(test_set)}")

    # ---- 1) G_dense on TRAIN clips (skip test; resumable) ----
    def gd_path(c):
        return os.path.join(gdense, seg_name(c), "point_cloud", f"iteration_{it}", "point_cloud.ply")

    def gd_fn(c, gpu):
        if os.path.exists(gd_path(c)):
            return
        on_gpu([PY, "train.py", "-s", dense_dir(c), "-m", os.path.join(gdense, seg_name(c)),
                "--init_pcd", fused_ply(c), "--iterations", it, "--test_iterations", it,
                "--save_iterations", it, "--disable_viewer", "--quiet"], gpu)

    todo = [c for c in train_clips if not os.path.exists(gd_path(c))]
    print(f"[nusc] gdense: {len(todo)}/{len(train_clips)} to run")
    if todo:
        d, f = run_jobs(todo, args.gpus, gd_fn, label="gdense", **pool)
        if f:
            print(f"[nusc] WARNING gdense failed for {len(f)} clips: {[seg_name(x) for x in f]}")

    # ---- 2) correspondences (test excluded) ----
    if not os.path.exists(os.path.join(corr, "build_times.json")):
        cmd = [PY, "-m", "gsnet.waymo_corr", "--root", args.root, "--gdense_dir", gdense,
               "--out_dir", corr, "--test_scenes", *args.test_scenes,
               "--iterations", it, "--workers", "8",
               "--global_scale", str(args.global_scale)]
        if args.no_filter:
            cmd.append("--no_filter")
        sh(cmd)
    else:
        print(f"[skip corr] {corr}")

    # ---- 3) train GS-Net (concat) ----
    if os.path.exists(ckpt):
        print(f"[skip train] {ckpt}")
    else:
        def tr_fn(_, gpu):
            on_gpu([PY, "-m", "gsnet.train_gsnet", "--corr_dir", corr, "--out_dir", model,
                    "--encoder_type", args.encoder_type, "--color_activation", "tanh",
                    "--w_rot", "0.1", "--w_pos", "10", "--T", "5", "--M", "3",
                    "--in_memory", "1", "--epochs", str(args.epochs)], gpu)
        run_jobs(["train"], args.gpus, tr_fn, label="train", **pool)
        assert os.path.exists(ckpt), "training did not produce a ckpt"

    # ---- 4) eval: sources+splits, gsnet inits, then 3-init x densify sweep ----
    for c in test_clips:                                  # CPU: source + frame split
        src = scene_source(c, "nusc_eval")
        write_cam_split(src, frames=HOLDOUT)

    def init_ply(c):
        return os.path.join(ev, seg_name(c), "gsnet_init.ply")

    def infer_fn(c, gpu):
        if os.path.exists(init_ply(c)):
            return
        os.makedirs(os.path.dirname(init_ply(c)), exist_ok=True)
        on_gpu([PY, "-m", "gsnet.infer", "--ckpt", ckpt, "--sparse", sparse_points(c),
                "--out", init_ply(c), "--norm_scale", str(args.global_scale)], gpu)

    need = [c for c in test_clips if not os.path.exists(init_ply(c))]
    if need:
        run_jobs(need, args.gpus, infer_fn, label="infer", **pool)

    results = {}                                           # (clip,init,dtag)->psnr
    rp = os.path.join(o, "nusc_eval.json")
    if os.path.exists(rp):
        results = {tuple(k.split("|")): v for k, v in json.load(open(rp)).items()}
    rlock = threading.Lock()

    def mp_of(c, init, dtag):
        return os.path.join(ev, seg_name(c), f"{init}_{dtag}")

    def ev_fn(job, gpu):
        c, init, dtag, dextra = job
        mp = mp_of(c, init, dtag)
        if os.path.exists(os.path.join(mp, "results.json")):
            psnr = read_psnr(mp)
        else:
            src = scene_source(c, "nusc_eval")
            ia = ([] if init == "sfm" else
                  ["--init_pcd", fused_ply(c)] if init == "mvs" else
                  ["--gsnet_init", init_ply(c)])
            on_gpu([PY, "train.py", "-s", src, "-m", mp, "--eval", "--iterations", it,
                    "--test_iterations", it, "--save_iterations", it,
                    "--disable_viewer", "--quiet", *ia, *dextra], gpu)
            on_gpu([PY, "render.py", "-m", mp, "--skip_train", "--quiet"], gpu)
            on_gpu([PY, "metrics.py", "-m", mp], gpu)
            psnr = read_psnr(mp)
        with rlock:
            results[(seg_name(c), init, dtag)] = psnr
            json.dump({"|".join(k): v for k, v in results.items()}, open(rp, "w"), indent=2)
        print(f"[eval] {seg_name(c)} {init}/{dtag} PSNR={psnr:.2f}", flush=True)

    jobs = [(c, init, dtag, dextra) for c in test_clips
            for init in INITS for dtag, dextra in DENSIFY
            if (seg_name(c), init, dtag) not in results]
    print(f"[nusc] eval: {len(jobs)} jobs to run")
    if jobs:
        d, f = run_jobs(jobs, args.gpus, ev_fn, label="eval", **pool)
        if f:
            print(f"[nusc] WARNING eval failed for {len(f)} jobs")

    aggregate(results, test_set, o)


def aggregate(results, test_set, out_root):
    clips = sorted(test_set)
    lines = ["", "# nuScenes SSE — init-spectrum x densification (CARLA-SSE holdout [4,9])", ""]
    lines += ["| densify | SfM | MVS | GS-Net | MVS-SfM | GSNet-SfM |",
              "|---|---|---|---|---|---|"]

    def avg(init, dtag):
        v = [results[(c, init, dtag)] for c in clips if (c, init, dtag) in results]
        return sum(v) / len(v) if v else float("nan")

    for dtag, _ in DENSIFY:
        s, m, g = avg("sfm", dtag), avg("mvs", dtag), avg("gsnet", dtag)
        lines.append(f"| {dtag} | {s:.2f} | {m:.2f} | {g:.2f} | {m-s:+.2f} | {g-s:+.2f} |")

    lines += ["", "## per-clip PSNR", "",
              "| clip | densify | SfM | MVS | GS-Net |", "|---|---|---|---|---|"]
    for c in clips:
        for dtag, _ in DENSIFY:
            def cell(init):
                return f"{results[(c,init,dtag)]:.2f}" if (c, init, dtag) in results else "-"
            lines.append(f"| {c} | {dtag} | {cell('sfm')} | {cell('mvs')} | {cell('gsnet')} |")

    lines += ["", "Read: SfM=baseline (number to beat); MVS-SfM=densification ceiling "
              "from a dense input; GSNet-SfM=our gain. If GS-Net wins at `on` but not "
              "`off`/`d2000` (or vice-versa) that pins the densification-washout effect."]
    table = "\n".join(lines)
    open(os.path.join(out_root, "nusc_eval.md"), "w").write(table + "\n")
    print("\n" + table)
    print(f"\n-> {out_root}/nusc_eval.md")


if __name__ == "__main__":
    main()
