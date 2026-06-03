#
# Re-aggregate any past run_sse / waymo_sse experiment EXCLUDING degenerate
# sequences (e.g. CARLA scene 310, which collapses to ~19-20 PSNR on every seed
# and every config -- see PROJECT_STATE -- and silently dragged down ablation
# averages that were computed over id sets containing it).
#
# No GPU / no re-run: it reads the per-sequence PSNR already stored in each
# sse_results.json and recomputes the mean over the kept ids.
#
# Usage:
#   python -m gsnet.reaggregate --root runs/encoder_ablation --exclude 310
#   python -m gsnet.reaggregate --root runs/design_sweep --exclude 310
#   # point --root at any dir; it finds every sse_results.json beneath it.
#

import argparse
import glob
import json
import os


def per_items(d):
    """run_sse stores 'per_sequence' (key 'id'); waymo_sse 'per_scene' ('scene')."""
    if "per_sequence" in d:
        return [(str(r.get("id")), r) for r in d["per_sequence"]]
    if "per_scene" in d:
        return [(str(r.get("scene")), r) for r in d["per_scene"]]
    return []


def reagg(path, exclude):
    d = json.load(open(path))
    items = per_items(d)
    out = {}
    for cfg in ("baseline", "gsnet"):
        vals = {k: [] for k in ("PSNR", "SSIM", "LPIPS")}
        ids = []
        for sid, r in items:
            if any(e in sid for e in exclude):
                continue
            if cfg in r:
                ids.append(sid)
                for k in vals:
                    if k in r[cfg]:
                        vals[k].append(r[cfg][k])
        if any(vals.values()):
            out[cfg] = {k: (sum(v) / len(v) if v else None) for k, v in vals.items()}
            out["_ids"] = ids
    return out, [sid for sid, _ in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir to search for sse_results.json")
    ap.add_argument("--exclude", nargs="+", default=["310"],
                    help="sequence ids/substrings to drop before averaging")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "**", "sse_results.json"),
                             recursive=True)
                   + glob.glob(os.path.join(args.root, "**", "cse_results.json"),
                               recursive=True))
    assert files, f"no sse_results.json/cse_results.json under {args.root}"

    rows = []
    for f in files:
        label = os.path.relpath(os.path.dirname(f), args.root).replace("/sse", "")
        res, all_ids = reagg(f, args.exclude)
        g = res.get("gsnet", {})
        b = res.get("baseline", {})
        delta = (g.get("PSNR") - b.get("PSNR")) if (g.get("PSNR") is not None
                                                    and b.get("PSNR") is not None) else None
        rows.append((label, g.get("PSNR"), b.get("PSNR"), delta, res.get("_ids", []), all_ids))

    print(f"\n# Re-aggregated EXCLUDING {args.exclude}  (under {args.root})\n")
    print("| config | gsnet PSNR | baseline | Δ | kept ids | all ids |")
    print("|---|---|---|---|---|---|")
    def fmt(x, f="{:.2f}"): return f.format(x) if isinstance(x, (int, float)) else "-"
    for label, gp, bp, dl, kept, allids in rows:
        print(f"| {label} | {fmt(gp)} | {fmt(bp)} | {fmt(dl,'{:+.2f}')} "
              f"| {','.join(kept)} | {','.join(allids)} |")


if __name__ == "__main__":
    main()
