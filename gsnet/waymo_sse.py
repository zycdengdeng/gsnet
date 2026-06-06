#
# Waymo Same-Sensor Evaluation (SSE): on each test scene, hold out 4 frames per
# camera and compare baseline 3DGS (sparse-SfM init) vs GS-Net+3DGS. Jobs are
# distributed across --gpus; results merge/resume into sse_results.json.
#
# Usage:
#   python -m gsnet.waymo_sse \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input \
#       --test_scenes <segA> <segB> \
#       --ckpt runs/waymo_gsnet/gsnet_latest.pt \
#       --out_dir runs/waymo_sse --gpus 1 7
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

from gsnet.waymo import (resolve_scene, seg_name, scene_source, sparse_points,
                        build_subset_source, images_dir)
from gsnet.make_cam_split import write_cam_split, write_camera_split

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
    summary = {"per_scene": records, "averages": {}}
    for cfg in ("baseline", "gsnet"):
        if any(cfg in r for r in records):
            summary["averages"][cfg] = {k: avg(cfg, k) for k in
                                        ("PSNR", "SSIM", "LPIPS", "optim_seconds")}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sse_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    lines = ["", "| Scene | Method | PSNR | SSIM | LPIPS | Optim(min) |",
             "|-------|--------|------|------|-------|------------|"]
    for r in records:
        for cfg in ("baseline", "gsnet"):
            if cfg in r:
                m = r[cfg]
                lines.append(f"| {r['scene'][:24]} | {cfg} | {m['PSNR']:.2f} | "
                             f"{m['SSIM']:.3f} | {m['LPIPS']:.3f} | "
                             f"{m['optim_seconds']/60:.1f} |")
    for cfg in ("baseline", "gsnet"):
        if cfg in summary["averages"]:
            a = summary["averages"][cfg]
            lines.append(f"| **Avg** | **{cfg}** | **{a['PSNR']:.2f}** | "
                         f"**{a['SSIM']:.3f}** | **{a['LPIPS']:.3f}** | "
                         f"{a['optim_seconds']/60:.1f} |")
    table = "\n".join(lines)
    with open(os.path.join(out_dir, "sse_results.md"), "w") as f:
        f.write(table + "\n")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--test_scenes", nargs="+", required=True)
    ap.add_argument("--ckpt", default="runs/waymo_gsnet/gsnet_latest.pt")
    ap.add_argument("--out_dir", default="runs/waymo_sse")
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 7])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--n_holdout", type=int, default=4)
    ap.add_argument("--holdout_mode", choices=["interp", "extrap"], default="interp",
                    help="interp=interior frames (no holes); extrap=last-N frames "
                         "per camera (forward extrapolation, real coverage holes)")
    ap.add_argument("--holdout_frames", type=int, nargs="+", default=None,
                    help="explicit per-camera test frame indices (overrides n_holdout/"
                         "mode); e.g. 4 9 = CARLA-SSE scheme")
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--skip_gsnet", action="store_true")
    ap.add_argument("--target_cams", nargs="+", default=[],
                    help="cross-sensor: hold out these entire cameras as test "
                         "(e.g. cam3 cam4 = side); default = per-camera frame holdout")
    ap.add_argument("--source_cams", nargs="+", default=[],
                    help="cross-sensor: ONLY these cameras are training/source; "
                         "all cameras except source+target are dropped "
                         "(e.g. --source_cams cam1 cam2 --target_cams cam0 = FL+FR->FRONT)")
    ap.add_argument("--train_extra", default="",
                    help="extra args to train.py for BOTH cfgs, e.g. "
                         "\"--densify_until_iter 0\" (preserve GS-Net's view-consistent "
                         "init from source-view densification washout)")
    ap.add_argument("--image_feats", action="store_true",
                    help="pass --image_feats <images_dir> to infer (for image-conditioned models)")
    ap.add_argument("--anchor_weight", type=float, default=0.0,
                    help=">0: also anchor 3DGS optimization to the GS-Net prediction "
                         "(persistent prior, not just init)")
    args = ap.parse_args()
    args.train_extra = args.train_extra.split()

    # Per-experiment tag isolates the sparse copy + test.txt so concurrent runs
    # with different splits don't overwrite each other's test.txt.
    tag = args.out_dir.strip("/").replace("/", "_")  # globally unique per out_dir
    args._tag = tag
    # Write the split up-front (camera-holdout for cross-sensor --target_cams,
    # else per-camera frame holdout for SSE), into each experiment's own copy.
    scenes = [resolve_scene(args.root, s) for s in args.test_scenes]
    src_by_scene = {}
    for sc in scenes:
        if args.source_cams:
            # keep only source+target cameras (drop the rest), test = target
            src = build_subset_source(sc, list(args.source_cams) + list(args.target_cams), tag)
            write_camera_split(src, args.target_cams)
        elif args.target_cams:
            src = scene_source(sc, tag)
            write_camera_split(src, args.target_cams)
        else:
            src = scene_source(sc, tag)
            write_cam_split(src, args.n_holdout, args.holdout_mode, args.holdout_frames)
        src_by_scene[sc] = src
    args._src_by_scene = src_by_scene

    jobs = []
    for sc in scenes:
        if not args.skip_baseline:
            jobs.append((sc, "baseline"))
        if not args.skip_gsnet:
            jobs.append((sc, "gsnet"))

    records = {}
    rp = os.path.join(args.out_dir, "sse_results.json")
    if os.path.exists(rp):
        for r in json.load(open(rp)).get("per_scene", []):
            records[r["scene"]] = r
    lock = threading.Lock()
    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)

    def worker(scene, cfg):
        gpu = gpu_q.get()
        try:
            name = seg_name(scene)
            source = args._src_by_scene[scene]
            if cfg == "baseline":
                mp = os.path.join(args.out_dir, name, "baseline")
                metrics, s = optimize_eval(source, mp, args.iterations,
                                           list(args.train_extra), gpu)
                res = {**metrics, "optim_seconds": s, "total_seconds": s}
            else:
                mp = os.path.join(args.out_dir, name, "gsnet")
                os.makedirs(mp, exist_ok=True)
                init_ply = os.path.join(mp, "gsnet_init.ply")
                infer_cmd = [PY, "-m", "gsnet.infer", "--ckpt", args.ckpt,
                             "--sparse", sparse_points(scene), "--out", init_ply]
                if args.image_feats:
                    infer_cmd += ["--image_feats", images_dir(scene)]
                infer_s = run(infer_cmd, gpu)
                extra = ["--gsnet_init", init_ply, *args.train_extra]
                if args.anchor_weight > 0:   # GS-Net as persistent prior (init + anchor)
                    extra += ["--gsnet_anchor", init_ply, "--anchor_weight", str(args.anchor_weight)]
                metrics, s = optimize_eval(source, mp, args.iterations, extra, gpu)
                res = {**metrics, "infer_seconds": infer_s, "optim_seconds": s,
                       "total_seconds": infer_s + s}
            with lock:
                records.setdefault(name, {"scene": name})[cfg] = res
                table = summarize(list(records.values()), args.out_dir)
            print(f"\n[done] {name}/{cfg} on gpu{gpu} :: PSNR={res['PSNR']:.2f}\n{table}",
                  flush=True)
        finally:
            gpu_q.put(gpu)

    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        for f in cf.as_completed([ex.submit(worker, sc, cfg) for sc, cfg in jobs]):
            f.result()
    print("\n" + summarize(list(records.values()), args.out_dir))
    print(f"\nResults -> {args.out_dir}/sse_results.{{json,md}}")


if __name__ == "__main__":
    main()
