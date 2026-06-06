#
# nuScenes GS-Net hyper-parameter sweep (epochs / T), REUSING the main run's
# corr (runs/nusc/corr) and the test clips. For each config it trains GS-Net and
# evaluates gsnet-init vs the sfm baseline on the test clips at ONE densify
# setting (default off, which preserves the init advantage so configs are
# comparable). Records PSNR + train time + infer time.
#
# Run AFTER run_nusc (which builds gdense+corr and finds the densify sweet spot).
#
# Usage:
#   nohup python -m gsnet.run_nusc_hpsweep \
#       --root /mnt/zihanw/gsnet_nusc --corr runs/nusc/corr \
#       --test_scenes 348_clip_09 332_clip_09 331_clip_09 299_clip_09 325_clip_09 \
#       --out_root runs/nusc_hp --epochs 100 200 400 --T 5 \
#       --densify_until 0 --global_scale 45 --gpus 0 1 2 3 4 5 6 7 \
#       > runs/nusc_hp.log 2>&1 &
#
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsnet.waymo import seg_name, fused_ply, sparse_points, scene_source, resolve_scene
from gsnet.make_cam_split import write_cam_split
from gsnet.gpu_pool import run_jobs

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT = [4, 9]


def on_gpu(cmd, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[gpu{gpu}] $ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)
    return time.time() - t0


def read_psnr(mp):
    d = json.load(open(os.path.join(mp, "results.json")))
    return d[sorted(d)[-1]]["PSNR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--corr", default="runs/nusc/corr", help="reuse main run's corr")
    ap.add_argument("--test_scenes", nargs="+", required=True)
    ap.add_argument("--out_root", default="runs/nusc_hp")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--epochs", type=int, nargs="+", default=[100, 200, 400])
    ap.add_argument("--T", type=int, nargs="+", default=[5])
    ap.add_argument("--encoder_type", default="concat")
    ap.add_argument("--global_scale", type=float, default=45.0)
    ap.add_argument("--densify_until", type=int, default=0,
                    help="densify_until_iter to evaluate every config at (0=off)")
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--min_free_mb", type=int, default=12000)
    ap.add_argument("--max_retries", type=int, default=5)
    args = ap.parse_args()

    o = args.out_root
    os.makedirs(o, exist_ok=True)
    it = str(args.iterations)
    pool = dict(min_free_mb=args.min_free_mb, max_retries=args.max_retries)
    lock = threading.Lock()
    dextra = [] if args.densify_until == 15000 else ["--densify_until_iter", str(args.densify_until)]
    configs = [(ep, t) for ep in args.epochs for t in args.T]
    tag = lambda ep, t: f"e{ep}_T{t}"
    test_clips = [resolve_scene(args.root, s) for s in args.test_scenes]
    for c in test_clips:
        write_cam_split(scene_source(c, "nusc_hp"), frames=HOLDOUT)

    # ---- 1) train each config ----
    def model_dir(ep, t):
        return os.path.join(o, tag(ep, t), "model")

    def tr_fn(cfg, gpu):
        ep, t = cfg
        if os.path.exists(os.path.join(model_dir(ep, t), "gsnet_latest.pt")):
            return
        on_gpu([PY, "-m", "gsnet.train_gsnet", "--corr_dir", args.corr,
                "--out_dir", model_dir(ep, t), "--encoder_type", args.encoder_type,
                "--color_activation", "tanh", "--w_rot", "0.1", "--w_pos", "10",
                "--T", str(t), "--M", "3", "--in_memory", "1", "--epochs", str(ep)], gpu)

    need = [c for c in configs if not os.path.exists(os.path.join(model_dir(*c), "gsnet_latest.pt"))]
    if need:
        run_jobs(need, args.gpus, tr_fn, label="hp-train", **pool)

    # ---- 2) eval: sfm baseline (once) + gsnet per config, all at densify_until ----
    results = {}
    rp = os.path.join(o, "hpsweep.json")
    if os.path.exists(rp):
        results = json.load(open(rp))

    def ev_fn(job, gpu):
        kind, cfg, c = job
        name = seg_name(c)
        src = scene_source(c, "nusc_hp")
        if kind == "sfm":
            mp = os.path.join(o, "_sfm", name)
            k = f"sfm|{name}"
            if not os.path.exists(os.path.join(mp, "results.json")):
                on_gpu([PY, "train.py", "-s", src, "-m", mp, "--eval", "--iterations", it,
                        "--test_iterations", it, "--save_iterations", it,
                        "--disable_viewer", "--quiet", *dextra], gpu)
                on_gpu([PY, "render.py", "-m", mp, "--skip_train", "--quiet"], gpu)
                on_gpu([PY, "metrics.py", "-m", mp], gpu)
            psnr = read_psnr(mp)
        else:
            ep, t = cfg
            mp = os.path.join(o, tag(ep, t), name)
            k = f"{tag(ep,t)}|{name}"
            init = os.path.join(mp, "gsnet_init.ply")
            if not os.path.exists(os.path.join(mp, "results.json")):
                os.makedirs(mp, exist_ok=True)
                on_gpu([PY, "-m", "gsnet.infer", "--ckpt",
                        os.path.join(model_dir(ep, t), "gsnet_latest.pt"),
                        "--sparse", sparse_points(c), "--out", init,
                        "--norm_scale", str(args.global_scale)], gpu)
                on_gpu([PY, "train.py", "-s", src, "-m", mp, "--eval", "--iterations", it,
                        "--test_iterations", it, "--save_iterations", it,
                        "--disable_viewer", "--quiet", "--gsnet_init", init, *dextra], gpu)
                on_gpu([PY, "render.py", "-m", mp, "--skip_train", "--quiet"], gpu)
                on_gpu([PY, "metrics.py", "-m", mp], gpu)
            psnr = read_psnr(mp)
        with lock:
            results[k] = psnr
            json.dump(results, open(rp, "w"), indent=2)
        print(f"[hp-eval] {k} PSNR={psnr:.2f}", flush=True)

    jobs = [("sfm", None, c) for c in test_clips if f"sfm|{seg_name(c)}" not in results]
    jobs += [("gsnet", cfg, c) for cfg in configs for c in test_clips
             if f"{tag(*cfg)}|{seg_name(c)}" not in results]
    if jobs:
        run_jobs(jobs, args.gpus, ev_fn, label="hp-eval", **pool)

    # ---- 3) aggregate ----
    clips = [seg_name(c) for c in test_clips]
    def avg(pref):
        v = [results[f"{pref}|{c}"] for c in clips if f"{pref}|{c}" in results]
        return sum(v) / len(v) if v else float("nan")
    sfm = avg("sfm")
    L = ["", f"# nuScenes GS-Net hp-sweep (densify_until={args.densify_until}, "
         f"baseline SfM={sfm:.2f})", "",
         "| config | epochs | T | gsnet PSNR | vs SfM | train(min) |",
         "|---|---|---|---|---|---|"]
    for ep, t in configs:
        g = avg(tag(ep, t))
        cfgp = os.path.join(model_dir(ep, t), "config.json")
        tr = json.load(open(cfgp)).get("total_seconds", 0) / 60 if os.path.exists(cfgp) else 0
        L.append(f"| {tag(ep,t)} | {ep} | {t} | {g:.2f} | {g-sfm:+.2f} | {tr:.1f} |")
    table = "\n".join(L)
    open(os.path.join(o, "hpsweep.md"), "w").write(table + "\n")
    print("\n" + table + f"\n\n-> {o}/hpsweep.md")


if __name__ == "__main__":
    main()
