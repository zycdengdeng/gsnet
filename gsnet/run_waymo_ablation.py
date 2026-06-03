#
# Waymo attribute-ablation study: which GS-Net factor (if any) transfers to real
# data? Hypothesis (user, 2026-06-03): on real scenes opacity may not be a
# learnable/transferable factor, and "accurate densification" (positions only)
# might be what helps -- while predicting opacity/scale could hurt.
#
# Trains GS-Net variants by disabling heads (they fall back to fixed defaults and
# drop from the loss) and SSE-evaluates each on the Waymo test scenes. Baseline
# (sparse-SfM-init 3DGS) is deterministic -> evaluated once and shared.
#
# Variants:
#   full          all heads (current model)
#   no_opacity    fixed opacity 0.1                    (tests "is opacity bad?")
#   no_color      use input rgb                        (tests "does color help?")
#   no_scale_rot  fixed isotropic scale, no rotation   (tests "is covariance good?")
#   dens_only     positions only (no color/opacity/scale-rot)  ("pure densification")
#
# Usage:
#   python -m gsnet.run_waymo_ablation \
#       --corr_dir CORR/waymo5 \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam \
#       --test_scenes 10275144660749673822_5755_561 15868625208244306149_4340_000 \
#       --out_dir runs/waymo5_ablation --gpus 0 1 2 3 4
#

import argparse
import concurrent.futures as cf
import json
import os
import queue
import subprocess
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VARIANTS = {
    "full":         [],
    "no_opacity":   ["--no_opacity"],
    "no_color":     ["--no_color"],
    "no_scale_rot": ["--no_scale_rot"],
    "xyz_rgb":      ["--no_opacity", "--no_scale_rot"],   # predict position + color only
    "dens_only":    ["--no_color", "--no_opacity", "--no_scale_rot"],
}


def sh(cmd, gpus):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def train_variant(name, flags, args, gpus):
    model = os.path.join(args.out_dir, name, "model")
    ckpt = os.path.join(model, "gsnet_latest.pt")
    if os.path.exists(ckpt):
        print(f"[skip train] {ckpt}")
        return ckpt
    sh([PY, "-m", "gsnet.train_gsnet", "--corr_dir", args.corr_dir, "--out_dir", model,
        "--encoder_type", "geom", "--color_activation", "tanh",
        "--w_rot", "0.1", "--w_pos", "10", "--T", str(args.T), "--M", str(args.M),
        "--in_memory", "1", "--epochs", str(args.epochs), *flags], gpus)
    return ckpt


def run_baseline(args, gpus):
    """Deterministic baseline (sparse-SfM init), shared by all variants."""
    out = os.path.join(args.out_dir, "_baseline")
    sh([PY, "-m", "gsnet.waymo_sse", "--root", args.root,
        "--test_scenes", *args.test_scenes, "--ckpt", "none", "--out_dir", out,
        "--skip_gsnet", "--n_holdout", str(args.n_holdout),
        "--iterations", str(args.iterations), "--gpus", *[str(g) for g in gpus]], gpus)
    return json.load(open(os.path.join(out, "sse_results.json")))["averages"]["baseline"]["PSNR"]


def sse_gsnet(name, ckpt, args, gpus):
    """GS-Net-only SSE for one variant (baseline computed separately)."""
    out = os.path.join(args.out_dir, name, "sse")
    sh([PY, "-m", "gsnet.waymo_sse", "--root", args.root,
        "--test_scenes", *args.test_scenes, "--ckpt", ckpt, "--out_dir", out,
        "--skip_baseline", "--n_holdout", str(args.n_holdout),
        "--iterations", str(args.iterations), "--gpus", *[str(g) for g in gpus]], gpus)
    return json.load(open(os.path.join(out, "sse_results.json")))["averages"]["gsnet"]["PSNR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_dir", default="CORR/waymo5")
    ap.add_argument("--root", required=True)
    ap.add_argument("--test_scenes", nargs="+", required=True)
    ap.add_argument("--out_dir", default="runs/waymo5_ablation")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    help=f"subset of {list(VARIANTS)}")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--n_holdout", type=int, default=4)
    args = ap.parse_args()
    for v in args.variants:
        assert v in VARIANTS, f"unknown variant {v}"
    os.makedirs(args.out_dir, exist_ok=True)

    # Resume: preload completed variants + baseline (reuse, don't re-run).
    rows = {}
    for name in args.variants:
        rp = os.path.join(args.out_dir, name, "sse", "sse_results.json")
        if os.path.exists(rp):
            av = json.load(open(rp)).get("averages", {})
            if "gsnet" in av:
                rows[name] = av["gsnet"]["PSNR"]
    baseline = None
    bp = os.path.join(args.out_dir, "_baseline", "sse_results.json")
    if os.path.exists(bp):
        baseline = json.load(open(bp))["averages"]["baseline"]["PSNR"]
    else:
        for name in args.variants:  # older serial runs stored baseline in a variant
            rp = os.path.join(args.out_dir, name, "sse", "sse_results.json")
            if os.path.exists(rp):
                av = json.load(open(rp)).get("averages", {})
                if "baseline" in av:
                    baseline = av["baseline"]["PSNR"]
                    break
    missing = [v for v in args.variants if v not in rows]
    print(f"[resume] done={list(rows)}  missing={missing}  baseline={baseline}", flush=True)

    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)

    # Phase 1: train missing variants IN PARALLEL (one GPU each).
    ckpts = {}
    def train_worker(name):
        g = gpu_q.get()
        try:
            ckpts[name] = train_variant(name, VARIANTS[name], args, [g])
        finally:
            gpu_q.put(g)
    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        for f in cf.as_completed([ex.submit(train_worker, n) for n in missing]):
            f.result()

    # Baseline once (deterministic, shared) — uses all GPUs (parallel scenes).
    if baseline is None:
        baseline = run_baseline(args, args.gpus)

    # Phase 2: GS-Net SSE for missing variants IN PARALLEL (one GPU each).
    lock = threading.Lock()
    def sse_worker(name):
        g = gpu_q.get()
        try:
            psnr = sse_gsnet(name, ckpts[name], args, [g])
            with lock:
                rows[name] = psnr
                write_table(rows, baseline, args.out_dir)
        finally:
            gpu_q.put(g)
    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        for f in cf.as_completed([ex.submit(sse_worker, n) for n in missing]):
            f.result()

    write_table(rows, baseline, args.out_dir)
    print("\n" + open(os.path.join(args.out_dir, "ablation_results.md")).read())


def write_table(rows, baseline, out_dir):
    lines = ["", "## Waymo attribute ablation (SSE, geom:tanh:0.1:10, M=3)",
             f"baseline (sparse-SfM init) = {baseline:.2f}" if baseline else "baseline = (pending)",
             "", "| variant | gsnet PSNR | Δ vs baseline |", "|---|---|---|"]
    for name, psnr in rows.items():
        d = f"{psnr - baseline:+.2f}" if baseline else "-"
        lines.append(f"| {name} | {psnr:.2f} | {d} |")
    json.dump({"baseline": baseline, "gsnet": rows},
              open(os.path.join(out_dir, "ablation_results.json"), "w"), indent=2)
    open(os.path.join(out_dir, "ablation_results.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
