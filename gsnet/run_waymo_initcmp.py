#
# Waymo INIT-SPECTRUM diagnostic: on each test scene, run 3DGS SSE from three
# different initializations and compare — to settle WHY GS-Net doesn't help
# Waymo SSE:
#   sfm   : sparse SfM points          (the baseline / floor)
#   mvs   : dense MVS fused.ply         (good dense init / near-ceiling)
#   gsnet : GS-Net predicted Gaussians  (where our method lands)
#
# Key question (user, 2026-06-03): §0.4 hinted MVS-init > SfM-init by +1.15 on
# Waymo, i.e. densification DOES have room. If MVS >> SfM but GS-Net << MVS, then
# the problem is GS-Net's PREDICTION QUALITY (fixable, novelty-preserving), NOT
# "the regime has no room". This places GS-Net on the SfM..MVS spectrum.
#
# Usage:
#   python -m gsnet.run_waymo_initcmp \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam \
#       --test_scenes 10275144660749673822_5755_561 15868625208244306149_4340_000 \
#       --ckpt runs/waymo5_gsnet/gsnet_latest.pt --out_dir runs/waymo5_initcmp \
#       --gpus 0 1 2 3
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsnet.waymo import resolve_scene, seg_name, scene_source, sparse_points, fused_ply
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


def read_psnr(mp):
    d = json.load(open(os.path.join(mp, "results.json")))
    m = sorted(d.keys())[-1]
    return d[m]["PSNR"]


def optimize_eval(source, mp, iters, extra, gpu):
    run([PY, "train.py", "-s", source, "-m", mp, "--eval", "--iterations", str(iters),
         "--test_iterations", str(iters), "--save_iterations", str(iters),
         "--disable_viewer", "--quiet", *extra], gpu)
    run([PY, "render.py", "-m", mp, "--skip_train", "--quiet"], gpu)
    run([PY, "metrics.py", "-m", mp], gpu)
    return read_psnr(mp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--test_scenes", nargs="+", required=True)
    ap.add_argument("--ckpt", default="runs/waymo5_gsnet/gsnet_latest.pt")
    ap.add_argument("--out_dir", default="runs/waymo5_initcmp")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n_holdout", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--image_feats", action="store_true")
    ap.add_argument("--train_extra", default="", help="e.g. \"--densify_until_iter 0\"")
    args = ap.parse_args()
    extra_common = args.train_extra.split()
    os.makedirs(args.out_dir, exist_ok=True)

    scenes = [resolve_scene(args.root, s) for s in args.test_scenes]
    tag = args.out_dir.strip("/").replace("/", "_")
    src_by = {}
    for sc in scenes:
        src = scene_source(sc, tag)
        write_cam_split(src, args.n_holdout)
        src_by[sc] = src

    jobs = [(sc, cfg) for sc in scenes for cfg in ("sfm", "mvs", "gsnet")]
    rows = {}
    rp = os.path.join(args.out_dir, "initcmp.json")
    if os.path.exists(rp):
        rows = json.load(open(rp))
    lock = threading.Lock()
    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)

    def worker(sc, cfg):
        name = seg_name(sc)
        if rows.get(name, {}).get(cfg) is not None:
            return
        gpu = gpu_q.get()
        try:
            src = src_by[sc]
            mp = os.path.join(args.out_dir, name, cfg)
            if cfg == "sfm":
                psnr = optimize_eval(src, mp, args.iterations, extra_common, gpu)
            elif cfg == "mvs":
                psnr = optimize_eval(src, mp, args.iterations,
                                     ["--init_pcd", fused_ply(sc), *extra_common], gpu)
            else:  # gsnet
                os.makedirs(mp, exist_ok=True)
                init_ply = os.path.join(mp, "gsnet_init.ply")
                icmd = [PY, "-m", "gsnet.infer", "--ckpt", args.ckpt,
                        "--sparse", sparse_points(sc), "--out", init_ply]
                if args.image_feats:
                    from gsnet.waymo import images_dir
                    icmd += ["--image_feats", images_dir(sc)]
                run(icmd, gpu)
                psnr = optimize_eval(src, mp, args.iterations,
                                     ["--gsnet_init", init_ply, *extra_common], gpu)
            with lock:
                rows.setdefault(name, {})[cfg] = psnr
                json.dump(rows, open(rp, "w"), indent=2)
                print(f"[done] {name}/{cfg} PSNR={psnr:.2f}", flush=True)
        finally:
            gpu_q.put(gpu)

    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        for f in cf.as_completed([ex.submit(worker, sc, cfg) for sc, cfg in jobs]):
            f.result()

    # table
    lines = ["", "## Waymo init spectrum (SSE PSNR @%d, n_holdout=%d)" % (args.iterations, args.n_holdout),
             "| scene | sfm | mvs | gsnet | mvs-sfm | gsnet-sfm | gsnet-mvs |",
             "|---|---|---|---|---|---|---|"]
    agg = {k: [] for k in ("sfm", "mvs", "gsnet")}
    for name, r in sorted(rows.items()):
        s, m, g = r.get("sfm"), r.get("mvs"), r.get("gsnet")
        for k, v in (("sfm", s), ("mvs", m), ("gsnet", g)):
            if v is not None:
                agg[k].append(v)
        def d(a, b): return f"{a-b:+.2f}" if (a is not None and b is not None) else "-"
        def f(x): return f"{x:.2f}" if x is not None else "-"
        lines.append(f"| {name[:22]} | {f(s)} | {f(m)} | {f(g)} | {d(m,s)} | {d(g,s)} | {d(g,m)} |")
    if agg["sfm"]:
        av = {k: sum(v)/len(v) for k, v in agg.items() if v}
        lines.append(f"| **Avg** | {av.get('sfm',0):.2f} | {av.get('mvs',0):.2f} | "
                     f"{av.get('gsnet',0):.2f} | {av.get('mvs',0)-av.get('sfm',0):+.2f} | "
                     f"{av.get('gsnet',0)-av.get('sfm',0):+.2f} | {av.get('gsnet',0)-av.get('mvs',0):+.2f} |")
    table = "\n".join(lines)
    open(os.path.join(args.out_dir, "initcmp.md"), "w").write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
