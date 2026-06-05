#
# Firm up the Waymo wide20 positive (+0.35 on 2 scenes) into a main-table number:
#   (1) EXPAND the test set: 5 test scenes / 15 train (was 2/18) -> a less-noisy
#       cross-scene average. Retrains GS-Net on the 15.
#   (2) MULTISEED: 3DGS is non-deterministic across runs even with seed=0 (CUDA
#       atomics in the rasterizer backward), so we get a valid variance estimate
#       by simply RE-RUNNING the SSE. densify-on (where the +0.35 lives) is
#       repeated R_ON times; densify-off once.
# Aggregates baseline/gsnet/Delta as mean+-std across (runs x scenes).
#
# Reuses the existing G_dense (runs/wide20/gdense) -- no 30k gdense re-run.
# Fully resumable: corr/train skip if present, each SSE run resumes its own json.
#
# Usage (unattended, free cards 4-7 to avoid the colleague on 0-3):
#   nohup python -m gsnet.run_wide20_firm \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam_wide20 \
#       --gpus 4 5 6 7 > runs/wide20_firm.log 2>&1 &
#

import argparse
import json
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 5 test scenes = the original 2 (continuity / directly comparable) + 3 more
# spread across the 20 (varied size/scale). Substrings (unique numeric ids).
TEST_SCENES = [
    "10275144660749673822_5755_561",
    "15868625208244306149_4340_000",
    "13238419657658219864",
    "9385013624094020582",
    "15270638100874320175",
]


def sh(cmd, gpus=None):
    env = os.environ.copy()
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def sse_done(out_dir):
    """True if this SSE run already has both cfgs for all 5 test scenes."""
    rp = os.path.join(out_dir, "sse_results.json")
    if not os.path.exists(rp):
        return False
    per = json.load(open(rp)).get("per_scene", [])
    return (len(per) >= len(TEST_SCENES)
            and all("baseline" in r and "gsnet" in r for r in per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--gdense_dir", default="runs/wide20/gdense",
                    help="reuse the already-built G_dense (no 30k re-run)")
    ap.add_argument("--out_root", default="runs/wide20_5test")
    ap.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--n_holdout", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--r_on", type=int, default=3, help="densify-on repeats (multiseed)")
    ap.add_argument("--r_off", type=int, default=1, help="densify-off repeats")
    args = ap.parse_args()
    o = args.out_root
    os.makedirs(o, exist_ok=True)
    corr, model = os.path.join(o, "corr"), os.path.join(o, "model")
    ckpt = os.path.join(model, "gsnet_latest.pt")

    # 1) correspondences on the 15 TRAIN scenes (5 test held out), reusing G_dense
    sh([PY, "-m", "gsnet.waymo_corr", "--root", args.root, "--gdense_dir", args.gdense_dir,
        "--out_dir", corr, "--test_scenes", *TEST_SCENES,
        "--iterations", str(args.iterations), "--workers", "8"])

    # 2) train GS-Net on the 15 (final recipe), skip if already trained
    if os.path.exists(ckpt):
        print(f"[skip train] {ckpt}")
    else:
        sh([PY, "-m", "gsnet.train_gsnet", "--corr_dir", corr, "--out_dir", model,
            "--encoder_type", "geom", "--color_activation", "tanh",
            "--w_rot", "0.1", "--w_pos", "10", "--T", "5", "--M", "3",
            "--in_memory", "1", "--epochs", str(args.epochs)], args.gpus[:1])

    # 3) SSE on the 5 test scenes -- multiseed (re-runs differ via CUDA atomics)
    runs = []  # (out_dir, variant)
    for r in range(args.r_on):
        runs.append((os.path.join(o, f"sse_on_r{r}"), []))
    for r in range(args.r_off):
        runs.append((os.path.join(o, f"sse_off_r{r}"), ["--train_extra", "--densify_until_iter 0"]))
    for out_dir, ex in runs:
        if sse_done(out_dir):
            print(f"[skip sse] {out_dir} (complete)")
            continue
        sh([PY, "-m", "gsnet.waymo_sse", "--root", args.root, "--test_scenes", *TEST_SCENES,
            "--ckpt", ckpt, "--out_dir", out_dir, "--n_holdout", str(args.n_holdout),
            "--iterations", str(args.iterations), "--gpus", *[str(g) for g in args.gpus], *ex])

    aggregate(o, args.r_on, args.r_off)


def _stats(vals):
    import statistics
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def aggregate(out_root, r_on, r_off):
    """mean+-std of baseline/gsnet/Delta across (runs x scenes), per variant."""
    lines = ["", "# Waymo wide20 FIRMING (5 test / 15 train, multiseed)", ""]
    for variant, R, pat in (("densify-ON", r_on, "sse_on_r"),
                            ("densify-OFF", r_off, "sse_off_r")):
        base_all, gs_all, delta_runs = [], [], []
        per_run = []
        for r in range(R):
            rp = os.path.join(out_root, f"{pat}{r}", "sse_results.json")
            if not os.path.exists(rp):
                continue
            per = json.load(open(rp)).get("per_scene", [])
            b = [x["baseline"]["PSNR"] for x in per if "baseline" in x]
            g = [x["gsnet"]["PSNR"] for x in per if "gsnet" in x]
            d = [x["gsnet"]["PSNR"] - x["baseline"]["PSNR"]
                 for x in per if "baseline" in x and "gsnet" in x]
            base_all += b
            gs_all += g
            if d:
                delta_runs.append(sum(d) / len(d))   # this run's mean Delta over scenes
            per_run.append((r, sum(b)/len(b) if b else float("nan"),
                            sum(g)/len(g) if g else float("nan"),
                            sum(d)/len(d) if d else float("nan")))
        bm, bs = _stats(base_all)
        gm, gs_ = _stats(gs_all)
        dm, ds = _stats(delta_runs)   # std over per-run mean Deltas = the headline error bar
        lines.append(f"## {variant}  (runs={len(per_run)}, scenes/run=5)")
        lines.append(f"- baseline PSNR = **{bm:.2f} ± {bs:.2f}**")
        lines.append(f"- gsnet    PSNR = **{gm:.2f} ± {gs_:.2f}**")
        lines.append(f"- **Delta (gsnet - baseline) = {dm:+.2f} ± {ds:.2f}**  "
                     f"(± = std of per-run mean Delta)")
        lines.append("")
        lines.append("| run | baseline | gsnet | Delta |")
        lines.append("|---|---|---|---|")
        for r, b, g, d in per_run:
            lines.append(f"| {r} | {b:.2f} | {g:.2f} | {d:+.2f} |")
        lines.append("")
    table = "\n".join(lines)
    with open(os.path.join(out_root, "firm_summary.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nDone. Summary -> {out_root}/firm_summary.md")


if __name__ == "__main__":
    main()
