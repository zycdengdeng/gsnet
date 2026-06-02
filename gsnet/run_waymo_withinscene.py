#
# Waymo WITHIN-SCENE evaluation -- the Waymo-side mirror of run_carla_loso.
#
# Question (the 2x2's bottom-left cell): on Waymo, does GS-Net help on a scene's
# *back* segment BECAUSE it saw that scene's *front* during training (exact-
# scene identity / memorization), or not at all (pure regime limit)?
#
# We already proved (scene_similarity) the test scenes are in-distribution, so a
# within-scene win cannot be attributed to "finally a similar scene" -- it would
# specifically mean "saw THIS scene." Design:
#   SEEN model   = waymo5_gsnet (trained WITH these scenes' fronts)
#   UNSEEN model = retrain on CORR/waymo5 with these scenes EXCLUDED
# Both evaluated on the SAME back segments (frames 020-039, independent SfM).
# Baseline 3DGS is deterministic -> computed once and shared.
#
# Δ_seen  = gsnet(seen)   - baseline   on the backs
# Δ_unseen= gsnet(unseen) - baseline   on the backs
# If Δ_seen >> Δ_unseen  -> exact-scene identity is what helps (the CARLA story).
# If Δ_seen ≈ Δ_unseen ≈ (the cross-scene −Δ) -> pure regime; scene familiarity
# does not rescue Waymo.
#
# Usage:
#   python -m gsnet.run_waymo_withinscene \
#       --back_root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam_next20 \
#       --front_corr CORR/waymo5 \
#       --within_ids 12879640240483815315 14004546003548947884 3988957004231180266 \
#       --seen_ckpt runs/waymo5_gsnet/gsnet_latest.pt \
#       --out_dir runs/waymo5_withinscene --gpus 0 1 2 3 4 5 6 7
#
# (Minor confound, same as LOSO: the UNSEEN model trains on fewer scenes
#  (8-3=5) so loses some data; --retrain_seen also retrains SEEN on the full 8
#  here for a hyperparameter-matched comparison if you want it airtight.)
#

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsnet.waymo import discover_scenes, seg_name

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def build_fold_corr(front_corr, out_dir, within_ids):
    """Symlink every CORR/<seg>.npz whose name does NOT contain any within_id."""
    fold = os.path.join(out_dir, "unseen_corr")
    os.makedirs(fold, exist_ok=True)
    kept, dropped = 0, []
    for f in sorted(glob.glob(os.path.join(front_corr, "*.npz"))):
        base = os.path.basename(f)
        if any(wid in base for wid in within_ids):
            dropped.append(base)
            continue
        link = os.path.join(fold, base)
        if not os.path.exists(link):
            os.symlink(os.path.abspath(f), link)
        kept += 1
    assert kept > 0, f"no training corr left after excluding {within_ids}"
    print(f"[unseen fold] kept {kept} train seqs; excluded {len(dropped)}: {dropped}")
    return fold


def train_gsnet(corr_dir, model_dir, args, gpu):
    ckpt = os.path.join(model_dir, "gsnet_latest.pt")
    if os.path.exists(ckpt):
        print(f"[skip train] {ckpt} exists")
        return ckpt
    sh([PY, "-m", "gsnet.train_gsnet", "--corr_dir", corr_dir, "--out_dir", model_dir,
        "--encoder_type", "geom", "--color_activation", "tanh",
        "--w_rot", "0.1", "--w_pos", "10", "--T", str(args.T), "--M", str(args.M),
        "--in_memory", "1", "--epochs", str(args.epochs)], gpu)
    return ckpt


def run_sse(back_root, test_names, ckpt, out_dir, args, skip_baseline):
    cmd = [PY, "-m", "gsnet.waymo_sse", "--root", back_root,
           "--test_scenes", *test_names, "--ckpt", ckpt, "--out_dir", out_dir,
           "--gpus", *[str(g) for g in args.gpus],
           "--n_holdout", str(args.n_holdout), "--iterations", str(args.iterations)]
    if skip_baseline:
        cmd.append("--skip_baseline")
    sh(cmd)
    with open(os.path.join(out_dir, "sse_results.json")) as f:
        return {r["scene"]: r for r in json.load(f)["per_scene"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--back_root", required=True, help="root of the back (020-039) segments")
    ap.add_argument("--front_corr", default="CORR/waymo5", help="training corr dir")
    ap.add_argument("--within_ids", nargs="+", required=True,
                    help="substrings identifying the within-scene segments")
    ap.add_argument("--seen_ckpt", default="runs/waymo5_gsnet/gsnet_latest.pt")
    ap.add_argument("--out_dir", default="runs/waymo5_withinscene")
    ap.add_argument("--retrain_seen", action="store_true",
                    help="also retrain SEEN on full front_corr here (hyperparam-matched)")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--n_holdout", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # resolve the back segments (full dir names) from substrings
    backs = [seg_name(s) for s in discover_scenes(args.back_root)
             if any(wid in seg_name(s) for wid in args.within_ids)]
    assert len(backs) == len(args.within_ids), \
        f"matched {backs} but expected {len(args.within_ids)} within_ids"
    print(f"[within-scene] back segments: {backs}")

    # --- models ---
    unseen_ckpt = train_gsnet(build_fold_corr(args.front_corr, args.out_dir, args.within_ids),
                              os.path.join(args.out_dir, "unseen_model"), args, args.gpus[:1])
    if args.retrain_seen:
        seen_ckpt = train_gsnet(args.front_corr,
                                os.path.join(args.out_dir, "seen_model"), args, args.gpus[:1])
    else:
        seen_ckpt = args.seen_ckpt
        assert os.path.exists(seen_ckpt), f"missing seen ckpt {seen_ckpt}"

    # --- eval (baseline+seen once; unseen reuses baseline) ---
    seen = run_sse(args.back_root, backs, seen_ckpt,
                   os.path.join(args.out_dir, "seen"), args, skip_baseline=False)
    unseen = run_sse(args.back_root, backs, unseen_ckpt,
                     os.path.join(args.out_dir, "unseen"), args, skip_baseline=True)

    # --- comparison table ---
    rows, agg = [], {"base": [], "ds": [], "du": []}
    for nm in backs:
        b = seen.get(nm, {}).get("baseline", {}).get("PSNR")
        gs = seen.get(nm, {}).get("gsnet", {}).get("PSNR")
        gu = unseen.get(nm, {}).get("gsnet", {}).get("PSNR")
        ds = (gs - b) if (b is not None and gs is not None) else None
        du = (gu - b) if (b is not None and gu is not None) else None
        rows.append((nm, b, gs, ds, gu, du))
        if b is not None: agg["base"].append(b)
        if ds is not None: agg["ds"].append(ds)
        if du is not None: agg["du"].append(du)

    def fmt(x, s="{:.2f}"): return s.format(x) if x is not None else "-"
    lines = ["", "## Waymo WITHIN-SCENE SSE (test = back/020-039 of seen scenes)",
             "| scene | baseline | gsnet(SEEN) | Δ_seen | gsnet(UNSEEN) | Δ_unseen |",
             "|---|---|---|---|---|---|"]
    for nm, b, gs, ds, gu, du in rows:
        lines.append(f"| {nm[:24]} | {fmt(b)} | {fmt(gs)} | {fmt(ds,'{:+.2f}')} "
                     f"| {fmt(gu)} | {fmt(du,'{:+.2f}')} |")
    if agg["base"]:
        mb = sum(agg["base"]) / len(agg["base"])
        mds = sum(agg["ds"]) / len(agg["ds"]) if agg["ds"] else None
        mdu = sum(agg["du"]) / len(agg["du"]) if agg["du"] else None
        lines.append(f"| **Avg** | **{mb:.2f}** | | **{fmt(mds,'{:+.2f}')}** | "
                     f"| **{fmt(mdu,'{:+.2f}')}** |")
    lines += ["",
              "_Δ_seen vs Δ_unseen on identical backs isolates 'saw THIS scene'._",
              "_Compare both to the cross-scene SSE Δ (≈−0.9 at n_holdout=4)._"]
    table = "\n".join(lines)
    with open(os.path.join(args.out_dir, "withinscene_results.md"), "w") as f:
        f.write(table + "\n")
    json.dump({"seen": seen, "unseen": unseen},
              open(os.path.join(args.out_dir, "withinscene_results.json"), "w"), indent=2)
    print("\n" + table)
    print(f"\nResults -> {args.out_dir}/withinscene_results.{{md,json}}")


if __name__ == "__main__":
    main()
