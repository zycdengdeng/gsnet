#
# Sparsity sweep (Waymo SSE): vary how many frames-per-camera are held out as
# test (--n_holdouts). Larger holdout => fewer TRAINING views => sparser regime.
# At each level runs BOTH baseline (sparse-SfM init) and GS-Net+3DGS, then
# tabulates the gap (gsnet - baseline). The point is to test the regime
# hypothesis: GS-Net should approach/overtake baseline as training views get
# sparse (less for plain SfM init to work with), while it trails when views are
# dense (baseline init already good).
#
# Each level runs in its own out_dir (isolated sparse copy + test.txt), so the
# whole thing is resumable and never collides on test.txt.
#
# Usage (5-cam, 20 frames/camera):
#   python -m gsnet.run_sparsity_sweep \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam \
#       --test_scenes <segA> <segB> \
#       --ckpt runs/waymo5_gsnet/gsnet_latest.pt \
#       --out_dir runs/waymo5_sparsity --n_holdouts 4 12 16 \
#       --frames_per_cam 20 --gpus 0 1 2 3 4
#

import argparse
import json
import os
import subprocess
import sys

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO)


def level_dir(args, nh):
    return os.path.join(args.out_dir, f"nh{nh}")


def one_level(nh, args):
    out = level_dir(args, nh)
    rp = os.path.join(out, "sse_results.json")
    # Resume: only (re)run if baseline+gsnet averages aren't both present yet.
    need = True
    if os.path.exists(rp):
        avg = json.load(open(rp)).get("averages", {})
        need = not ("baseline" in avg and "gsnet" in avg)
    if need:
        run([PY, "-m", "gsnet.waymo_sse", "--root", args.root,
             "--test_scenes", *args.test_scenes, "--ckpt", args.ckpt,
             "--out_dir", out, "--n_holdout", str(nh),
             "--iterations", str(args.iterations), "--gpus", *map(str, args.gpus)])
    avg = json.load(open(rp))["averages"]
    row = {"n_holdout": nh}
    if args.frames_per_cam:
        row["train_per_cam"] = args.frames_per_cam - nh
    for cfg in ("baseline", "gsnet"):
        if cfg in avg:
            row[cfg] = {k: avg[cfg][k] for k in ("PSNR", "SSIM", "LPIPS")}
    return row


def summarize(rows, args):
    rows = sorted(rows, key=lambda r: r["n_holdout"])
    json.dump(rows, open(os.path.join(args.out_dir, "sparsity_sweep.json"), "w"), indent=2)
    has_train = any("train_per_cam" in r for r in rows)
    head = "| n_holdout |" + (" train/cam |" if has_train else "") + \
           " baseline PSNR | gsnet PSNR | Δ(gsnet-base) |"
    sep = "|" + "---|" * (head.count("|") - 1)
    lines = ["", f"## Sparsity sweep (Waymo SSE, @{args.iterations} iters) "
                 f"-- larger n_holdout = sparser training", head, sep]
    for r in rows:
        b = r.get("baseline", {}).get("PSNR")
        g = r.get("gsnet", {}).get("PSNR")
        d = (g - b) if (b is not None and g is not None) else None
        cells = [str(r["n_holdout"])]
        if has_train:
            cells.append(str(r.get("train_per_cam", "?")))
        cells.append(f"{b:.2f}" if b is not None else "-")
        cells.append(f"{g:.2f}" if g is not None else "-")
        cells.append(f"{d:+.2f}" if d is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    open(os.path.join(args.out_dir, "sparsity_sweep.md"), "w").write(table + "\n")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--test_scenes", nargs="+", required=True)
    ap.add_argument("--ckpt", default="runs/waymo5_gsnet/gsnet_latest.pt")
    ap.add_argument("--out_dir", default="runs/waymo5_sparsity")
    ap.add_argument("--n_holdouts", type=int, nargs="+", default=[4, 12, 16])
    ap.add_argument("--frames_per_cam", type=int, default=0,
                    help="if set, also report train frames/camera (= this - n_holdout)")
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    # Levels run sequentially; each waymo_sse already fans scenes x methods
    # across all --gpus. Keep going if one level fails so the rest still run.
    for nh in args.n_holdouts:
        try:
            rows.append(one_level(nh, args))
        except Exception as e:
            print(f"[FAILED] n_holdout={nh}: {e}", flush=True)
        if rows:
            print(summarize(rows, args), flush=True)

    print("\n" + summarize(rows, args))
    print(f"\nResults -> {args.out_dir}/sparsity_sweep.{{json,md}}")


if __name__ == "__main__":
    main()
