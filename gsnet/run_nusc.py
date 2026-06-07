#
# nuScenes SSE — full hands-off pipeline with GPU-fault-tolerant scheduling.
#
# Flow:
#   1) G_dense : 3DGS (30k, from MVS fused.ply) on every TRAIN clip -> GS-Net's
#                supervision target. resumable.
#   2) corr    : sparse SfM points <-> G_dense, test clips excluded,
#                --global_scale (metric), --no_filter.
#   3) train   : GS-Net (encoder=concat, paper). training time recorded.
#   4) eval    : on each TEST clip, 3 inits x a DENSIFICATION SWEEP, CARLA-SSE
#                frame holdout [4,9]. Each run is checkpointed at --eval_iters so
#                we get the PSNR-vs-iteration CURVE for free (does the baseline
#                catch up by 30k?). inits = {sfm, mvs(fused.ply), gsnet}:
#                MVS-input vs SfM = densification ceiling; where GS-Net lands.
#   5) aggregate-> nusc_eval.md: final PSNR/SSIM/LPIPS, convergence curve,
#                  TIMING (infer/train/optim) + #Gaussians (efficiency).
#
# gpu_pool: free-card-first, squeezes onto low-use cards, OOM/failure -> retry on
# another free card, never aborts. Fully resumable (re-run to continue).
#
# Usage:
#   nohup python -m gsnet.run_nusc \
#       --root /mnt/zihanw/gsnet_nusc \
#       --test_scenes 348_clip_09 332_clip_09 331_clip_09 299_clip_09 325_clip_09 \
#       --out_root runs/nusc --gpus 0 1 2 3 4 5 6 7 \
#       --global_scale 45 --no_filter --epochs 200 \
#       --densify_iters 0 2000 5000 15000 --eval_iters 7000 15000 30000 \
#       --min_free_mb 12000 > runs/nusc.log 2>&1 &
#
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsnet.waymo import (discover_scenes, seg_name, dense_dir, fused_ply,
                         sparse_points, scene_source, resolve_scene)
from gsnet.make_cam_split import write_cam_split
from gsnet.gpu_pool import run_jobs

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INITS = ["sfm", "mvs", "gsnet"]
HOLDOUT = [4, 9]                      # CARLA-SSE per-camera test frames
MET = ("PSNR", "SSIM", "LPIPS")


def on_gpu(cmd, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[gpu{gpu}] $ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)
    return time.time() - t0


def sh(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO)


def ply_count(path):
    try:
        with open(path, "rb") as f:
            for line in f:
                if line.startswith(b"element vertex"):
                    return int(line.split()[-1])
                if line.strip() == b"end_header":
                    break
    except Exception:
        pass
    return -1


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--test_scenes", nargs="+", required=True)
    ap.add_argument("--out_root", default="runs/nusc")
    ap.add_argument("--gdense_dir", default="",
                    help="reuse an existing G_dense dir (e.g. runs/nusc/gdense) across "
                         "variants so gdense isn't re-run; default = <out_root>/gdense")
    ap.add_argument("--corr_dir", default="",
                    help="reuse an existing corr dir (e.g. runs/nusc/corr) so corr isn't "
                         "rebuilt -- e.g. swapping ONLY the encoder; default = <out_root>/corr")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--encoder_type", default="concat")
    ap.add_argument("--T", type=int, default=5,
                    help="expansion factor = #Gaussians predicted per sparse point. "
                         "Also sets corr K (loss matches T heads to K nearest G_dense "
                         "1:1, so T==K). Bigger = denser init (T=15 -> ~3x denser).")
    ap.add_argument("--global_scale", type=float, default=45.0)
    ap.add_argument("--no_filter", action="store_true")
    ap.add_argument("--ckpt", default="",
                    help="reuse an external trained ckpt (skips training); e.g. "
                         "runs/nusc/model/gsnet_latest.pt for an inference-only variant")
    ap.add_argument("--iter_passes", type=int, default=1,
                    help=">1: iterative/recursive densification init (idea A) -- run "
                         "GS-Net N passes, feeding high-confidence output back in")
    ap.add_argument("--conf_thresh", type=float, default=0.5,
                    help="opacity threshold for high-confidence point selection between passes")
    ap.add_argument("--densify_iters", type=int, nargs="+", default=[0, 2000, 5000, 15000],
                    help="densify_until_iter values to sweep (find nuScenes' OWN sweet "
                         "spot). 0=off, 15000=3DGS default. tag=d<n>.")
    ap.add_argument("--eval_iters", type=int, nargs="+", default=[7000, 15000, 30000],
                    help="checkpoints to score -> PSNR-vs-iteration convergence curve. "
                         "Last value is the total optimisation length.")
    ap.add_argument("--gdense_save_iters", type=int, nargs="+", default=[15000, 30000],
                    help="iterations at which to SAVE G_dense (the supervision target). "
                         "Saving 15k too (≈free) preserves the option of a cleaner / less "
                         "view-overfit target than the fully-converged 30k, for a later "
                         "target-design sweep (build corr from a different iter).")
    ap.add_argument("--gdense_iter", type=int, default=30000,
                    help="which saved G_dense iteration to USE as the training target")
    ap.add_argument("--min_free_mb", type=int, default=15000)
    ap.add_argument("--max_retries", type=int, default=5)
    args = ap.parse_args()

    eval_iters = sorted(set(args.eval_iters) | {args.iterations})   # always score the final
    final = args.iterations
    gd_save = sorted(set(args.gdense_save_iters) | {final})          # G_dense checkpoints to save
    assert args.gdense_iter in gd_save, f"--gdense_iter {args.gdense_iter} not in saved {gd_save}"
    densify = [(f"d{n}", ([] if n == 15000 else ["--densify_until_iter", str(n)]))
               for n in args.densify_iters]

    o = args.out_root
    model, ev = (os.path.join(o, x) for x in ("model", "eval"))
    gdense = args.gdense_dir or os.path.join(o, "gdense")   # reuse external G_dense if given
    corr = args.corr_dir or os.path.join(o, "corr")         # reuse external corr if given
    os.makedirs(o, exist_ok=True)
    ckpt = args.ckpt or os.path.join(model, "gsnet_latest.pt")   # reuse external ckpt if given
    pool = dict(min_free_mb=args.min_free_mb, max_retries=args.max_retries)
    lock = threading.Lock()

    all_clips = discover_scenes(args.root)
    test_set = {seg_name(c) for c in all_clips
                if any(t in seg_name(c) for t in args.test_scenes)}
    train_clips = [c for c in all_clips if seg_name(c) not in test_set]
    test_clips = [resolve_scene(args.root, t) for t in args.test_scenes]
    assert len(test_set) == len(args.test_scenes), \
        f"test match mismatch: {sorted(test_set)} vs {args.test_scenes}"
    print(f"[nusc] {len(all_clips)} clips: {len(train_clips)} train, "
          f"{len(test_clips)} test={sorted(test_set)}", flush=True)

    # ---- 1) G_dense on TRAIN clips (resumable, timed) ----
    gd_times = {}
    gtp = os.path.join(o, "gdense_times.json")
    if os.path.exists(gtp):
        gd_times = json.load(open(gtp))

    def gd_path(c):
        return os.path.join(gdense, seg_name(c), "point_cloud",
                            f"iteration_{final}", "point_cloud.ply")

    def gd_fn(c, gpu):
        if os.path.exists(gd_path(c)):
            return
        si = ["--save_iterations", *[str(i) for i in gd_save]]
        s = on_gpu([PY, "train.py", "-s", dense_dir(c), "-m", os.path.join(gdense, seg_name(c)),
                    "--init_pcd", fused_ply(c), "--iterations", str(final),
                    "--test_iterations", str(final), *si,
                    "--disable_viewer", "--quiet"], gpu)
        with lock:
            gd_times[seg_name(c)] = s
            jdump(gd_times, gtp)

    todo = [c for c in train_clips if not os.path.exists(gd_path(c))]
    print(f"[nusc] gdense: {len(todo)}/{len(train_clips)} to run", flush=True)
    if todo:
        _, f = run_jobs(todo, args.gpus, gd_fn, label="gdense", **pool)
        if f:
            print(f"[nusc] WARN gdense failed: {[seg_name(x) for x in f]}", flush=True)

    # ---- 2) correspondences (test excluded) ----
    if not os.path.exists(os.path.join(corr, "build_times.json")):
        cmd = [PY, "-m", "gsnet.waymo_corr", "--root", args.root, "--gdense_dir", gdense,
               "--out_dir", corr, "--test_scenes", *args.test_scenes,
               "--iterations", str(args.gdense_iter), "--workers", "8",
               "--K", str(args.T), "--global_scale", str(args.global_scale)]
        if args.no_filter:
            cmd.append("--no_filter")
        sh(cmd)
    else:
        print(f"[skip corr] {corr}", flush=True)

    # ---- 3) train GS-Net (concat) ----
    if os.path.exists(ckpt):
        print(f"[skip train] {ckpt}", flush=True)
    else:
        def tr_fn(_, gpu):
            on_gpu([PY, "-m", "gsnet.train_gsnet", "--corr_dir", corr, "--out_dir", model,
                    "--encoder_type", args.encoder_type, "--color_activation", "tanh",
                    "--w_rot", "0.1", "--w_pos", "10", "--T", str(args.T), "--M", "3",
                    "--in_memory", "1", "--epochs", str(args.epochs)], gpu)
        run_jobs(["train"], args.gpus, tr_fn, label="train", **pool)
        assert os.path.exists(ckpt), "training produced no ckpt"

    # ---- 4) eval ----
    for c in test_clips:                                  # CPU: source + frame split (upfront)
        write_cam_split(scene_source(c, "nusc_eval"), frames=HOLDOUT)

    infer_times = {}
    itp = os.path.join(o, "infer_times.json")
    if os.path.exists(itp):
        infer_times = json.load(open(itp))

    def init_ply(c):
        return os.path.join(ev, seg_name(c), "gsnet_init.ply")

    def infer_fn(c, gpu):
        if os.path.exists(init_ply(c)):
            return
        os.makedirs(os.path.dirname(init_ply(c)), exist_ok=True)
        if args.iter_passes > 1:                      # idea A: recursive densification init
            cmd = [PY, "-m", "gsnet.infer_iterative", "--ckpt", ckpt,
                   "--sparse", sparse_points(c), "--out", init_ply(c),
                   "--passes", str(args.iter_passes), "--conf_thresh", str(args.conf_thresh),
                   "--norm_scale", str(args.global_scale)]
        else:
            cmd = [PY, "-m", "gsnet.infer", "--ckpt", ckpt, "--sparse", sparse_points(c),
                   "--out", init_ply(c), "--norm_scale", str(args.global_scale)]
        s = on_gpu(cmd, gpu)
        with lock:
            infer_times[seg_name(c)] = s
            jdump(infer_times, itp)

    need = [c for c in test_clips if not os.path.exists(init_ply(c))]
    if need:
        run_jobs(need, args.gpus, infer_fn, label="infer", **pool)

    results = {}
    rp = os.path.join(o, "nusc_eval.json")
    if os.path.exists(rp):
        results = json.load(open(rp))

    def key(c, init, dtag):
        return f"{seg_name(c)}|{init}|{dtag}"

    def ck_ply(mp, i):
        return os.path.join(mp, "point_cloud", f"iteration_{i}", "point_cloud.ply")

    def ev_fn(job, gpu):
        c, init, dtag, dextra = job
        mp = os.path.join(ev, seg_name(c), f"{init}_{dtag}")
        rj = os.path.join(mp, "results.json")
        s = -1.0
        if not os.path.exists(rj):
            src = scene_source(c, "nusc_eval")
            ia = ([] if init == "sfm" else
                  ["--init_pcd", fused_ply(c)] if init == "mvs" else
                  ["--gsnet_init", init_ply(c)])
            si = ["--save_iterations", *[str(i) for i in eval_iters]]
            ti = ["--test_iterations", *[str(i) for i in eval_iters]]
            if not all(os.path.exists(ck_ply(mp, i)) for i in eval_iters):  # retrain if any ckpt missing
                s = on_gpu([PY, "train.py", "-s", src, "-m", mp, "--eval",
                            "--iterations", str(final), *ti, *si,
                            "--disable_viewer", "--quiet", *ia, *dextra], gpu)
            for i in eval_iters:                          # render every checkpoint that exists
                if os.path.exists(ck_ply(mp, i)):
                    on_gpu([PY, "render.py", "-m", mp, "--iteration", str(i),
                        "--skip_train", "--quiet"], gpu)
            on_gpu([PY, "metrics.py", "-m", mp], gpu)
        d = json.load(open(rj))
        iters_res = {}
        for i in eval_iters:
            k = f"ours_{i}"
            if k in d:
                iters_res[str(i)] = {**{m: d[k][m] for m in MET},
                                     "ngauss": ply_count(ck_ply(mp, i))}
        with lock:
            results[key(c, init, dtag)] = {"optim_s": s, "iters": iters_res}
            jdump(results, rp)
        fin = iters_res.get(str(final), {})
        print(f"[eval] {seg_name(c)} {init}/{dtag} PSNR@{final}={fin.get('PSNR', float('nan')):.2f} "
              f"optim={s/60:.1f}min ngauss={fin.get('ngauss', -1)}", flush=True)

    jobs = [(c, init, dtag, dextra) for c in test_clips
            for init in INITS for dtag, dextra in densify
            if key(c, init, dtag) not in results]
    print(f"[nusc] eval: {len(jobs)} jobs to run", flush=True)
    if jobs:
        _, f = run_jobs(jobs, args.gpus, ev_fn, label="eval", **pool)
        if f:
            print(f"[nusc] WARN eval failed for {len(f)} jobs", flush=True)

    aggregate(results, sorted(test_set), [d for d, _ in densify], eval_iters,
              gd_times, infer_times, model, o)


def aggregate(results, clips, dtags, eval_iters, gd_times, infer_times, model_dir, out_root):
    final = eval_iters[-1]

    def avg(init, dtag, it, field):
        v = []
        for c in clips:
            r = results.get(f"{c}|{init}|{dtag}", {}).get("iters", {}).get(str(it))
            if r and field in r:
                v.append(r[field])
        return sum(v) / len(v) if v else float("nan")

    def avg_optim(init, dtag):
        v = [results[f"{c}|{init}|{dtag}"]["optim_s"] for c in clips
             if f"{c}|{init}|{dtag}" in results and results[f"{c}|{init}|{dtag}"]["optim_s"] > 0]
        return sum(v) / len(v) / 60 if v else float("nan")

    L = ["", "# nuScenes SSE — init-spectrum x densification x convergence (holdout [4,9])", ""]
    L += [f"## Final PSNR @ {final}  (GSNet-SfM = our gain; MVS-SfM = ceiling)",
          "| densify | SfM | MVS | GS-Net | MVS-SfM | GSNet-SfM |", "|---|---|---|---|---|---|"]
    for d in dtags:
        s, m, g = (avg("sfm", d, final, "PSNR"), avg("mvs", d, final, "PSNR"),
                   avg("gsnet", d, final, "PSNR"))
        L.append(f"| {d} | {s:.2f} | {m:.2f} | {g:.2f} | {m-s:+.2f} | {g-s:+.2f} |")

    L += ["", "## Convergence: GSNet-SfM (PSNR gain) at each iteration",
          "| densify | " + " | ".join(f"@{i}" for i in eval_iters) + " |",
          "|---|" + "---|" * len(eval_iters)]
    for d in dtags:
        cells = [f"{avg('gsnet',d,i,'PSNR')-avg('sfm',d,i,'PSNR'):+.2f}" for i in eval_iters]
        L.append(f"| {d} | " + " | ".join(cells) + " |")

    L += ["", f"## Final LPIPS @ {final} (lower better)",
          "| densify | SfM | MVS | GS-Net |", "|---|---|---|---|"]
    for d in dtags:
        L.append(f"| {d} | {avg('sfm',d,final,'LPIPS'):.3f} | {avg('mvs',d,final,'LPIPS'):.3f} "
                 f"| {avg('gsnet',d,final,'LPIPS'):.3f} |")

    tr, floss = "?", {}
    cfgp = os.path.join(model_dir, "train_times.json")
    if os.path.exists(cfgp):
        cj = json.load(open(cfgp))
        tr = f"{cj.get('total_seconds', 0)/60:.1f} min"
        floss = cj.get("final_loss", {})
    inf = (sum(infer_times.values()) / len(infer_times)) if infer_times else float("nan")
    gdm = (sum(gd_times.values()) / len(gd_times) / 60) if gd_times else float("nan")
    L += ["", "## Timing / efficiency",
          f"- GS-Net **train** (one-off): {tr}" + (f"  final_loss={floss}" if floss else ""),
          f"- GS-Net **infer** (one forward pass, per clip): **{inf:.2f}s avg**",
          f"- G_dense build (3DGS {final} per train clip): {gdm:.1f} min avg",
          "", f"| densify | init | optim(min) | #Gaussians@{final} |", "|---|---|---|---|"]
    for d in dtags:
        for init in INITS:
            L.append(f"| {d} | {init} | {avg_optim(init,d):.1f} | {avg(init,d,final,'ngauss'):.0f} |")

    L += ["", f"## per-clip PSNR @ {final}",
          "| clip | densify | SfM | MVS | GS-Net |", "|---|---|---|---|---|"]
    for c in clips:
        for d in dtags:
            def cell(init):
                r = results.get(f"{c}|{init}|{d}", {}).get("iters", {}).get(str(final))
                return f"{r['PSNR']:.2f}" if r else "-"
            L.append(f"| {c} | {d} | {cell('sfm')} | {cell('mvs')} | {cell('gsnet')} |")

    L += ["", "Read: scan the Final-PSNR rows for the densify value maximising "
          "GSNet-SfM (nuScenes' own sweet spot). The Convergence table shows whether "
          "GS-Net's lead shrinks from @7k to @30k (washout). MVS-SfM is the ceiling. "
          "Efficiency: GS-Net infer is seconds (vs MVS patch-match offline minutes); "
          "fewer #Gaussians / less optim at equal quality is also an advantage."]
    table = "\n".join(L)
    open(os.path.join(out_root, "nusc_eval.md"), "w").write(table + "\n")
    print("\n" + table + f"\n\n-> {out_root}/nusc_eval.md", flush=True)


if __name__ == "__main__":
    main()
