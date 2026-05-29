#
# Diagnostic: on a single Waymo scene, compare two 3DGS reconstructions with the
# SAME per-camera train/test split (4 frames/camera held out):
#   * sfm_3dgs : init from sparse SfM points (standard 3DGS)
#   * mvs_3dgs : init from MVS fused.ply (--init_pcd)
# Reports PSNR/SSIM/LPIPS for both, to judge whether 60 images (2s, 3 cams) give
# acceptable reconstruction quality and how much MVS densification helps.
#
# Usage:
#   python -m gsnet.waymo_diag \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input \
#       --scene segment-3425716115468765803_977_756_997_756_with_camera_labels \
#       --out_dir runs/waymo_diag --gpus 2 3
#

import argparse
import concurrent.futures as cf
import json
import os
import queue
import subprocess
import sys
import time

from gsnet.waymo import resolve_scene, seg_name, dense_dir, fused_ply
from gsnet.make_cam_split import write_cam_split

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene", required=True, help="segment name or full path")
    ap.add_argument("--out_dir", default="runs/waymo_diag")
    ap.add_argument("--gpus", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--n_holdout", type=int, default=4)
    args = ap.parse_args()

    scene = resolve_scene(args.root, args.scene)
    source = dense_dir(scene)
    write_cam_split(source, args.n_holdout)   # shared split for both configs

    configs = {
        "sfm_3dgs": [],
        "mvs_3dgs": ["--init_pcd", fused_ply(scene)],
    }
    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)
    out = {"scene": seg_name(scene)}
    lock = __import__("threading").Lock()

    def work(name, extra):
        gpu = gpu_q.get()
        try:
            mp = os.path.join(args.out_dir, seg_name(scene), name)
            metrics, s = optimize_eval(source, mp, args.iterations, extra, gpu)
            with lock:
                out[name] = {**metrics, "optim_seconds": s}
            print(f"[{name}] {metrics}  optim={s/60:.1f}min")
        finally:
            gpu_q.put(gpu)

    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        for f in cf.as_completed([ex.submit(work, n, e) for n, e in configs.items()]):
            f.result()

    os.makedirs(os.path.join(args.out_dir, seg_name(scene)), exist_ok=True)
    with open(os.path.join(args.out_dir, seg_name(scene), "diag.json"), "w") as f:
        json.dump(out, f, indent=2)

    lines = ["", f"### {seg_name(scene)}  (60 imgs, {args.n_holdout}/cam held out)",
             "| Init | PSNR | SSIM | LPIPS | Optim(min) |",
             "|------|------|------|-------|------------|"]
    for name in ("sfm_3dgs", "mvs_3dgs"):
        if name in out:
            m = out[name]
            lines.append(f"| {name} | {m['PSNR']:.2f} | {m['SSIM']:.3f} | "
                         f"{m['LPIPS']:.3f} | {m['optim_seconds']/60:.1f} |")
    table = "\n".join(lines)
    with open(os.path.join(args.out_dir, seg_name(scene), "diag.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
