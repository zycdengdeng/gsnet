#
# Read nuScenes SSE per-scene results at a single iteration (default 30000) from
# run_nusc's nusc_eval.json, and dump a per-clip markdown table per densify tag.
#
# nusc_eval.json keys: "<clip>|<init>|<densify>" -> {"iters": {"<it>": {PSNR,SSIM,LPIPS,ngauss}}}
#   init    in {sfm (baseline), mvs (ceiling), gsnet}
#   densify in {d0, d2000, d5000, d15000}   (d15000 = vanilla 3DGS densification)
#
# Usage:
#   python -m gsnet.dump_nusc --root runs/nusc_filt
#   python -m gsnet.dump_nusc --root runs/nusc_filt --densify d15000
#
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/nusc_filt", help="dir holding nusc_eval.json")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--densify", nargs="+", default=None,
                    help="densify tags to report (default: all present)")
    ap.add_argument("--out", default="", help="output md (default <root>/nusc_per_scene_<iter>.md)")
    args = ap.parse_args()

    jp = os.path.join(args.root, "nusc_eval.json")
    d = json.load(open(jp))
    it = str(args.iter)
    inits = ["sfm", "mvs", "gsnet"]

    clips, dtags = [], []
    for key in d:
        clip, init, dt = key.split("|")
        if clip not in clips:
            clips.append(clip)
        if dt not in dtags:
            dtags.append(dt)
    clips.sort()
    dtags = args.densify or sorted(dtags)

    def cell(clip, init, dt, metric):
        r = d.get(f"{clip}|{init}|{dt}", {}).get("iters", {}).get(it)
        return r[metric] if r and metric in r else None

    def fmt(x, f="{:.2f}"):
        return f.format(x) if isinstance(x, (int, float)) else "-"

    L = [f"# nuScenes SSE per-scene @ {args.iter}  (under {args.root})", ""]
    for dt in dtags:
        L += [f"## densify {dt}",
              "| clip | SfM PSNR | MVS PSNR | GSNet PSNR | Δ(GSNet−SfM) | "
              "SfM SSIM | GSNet SSIM | SfM LPIPS | GSNet LPIPS |",
              "|---|---|---|---|---|---|---|---|---|"]
        acc = {(init, m): [] for init in inits for m in ("PSNR", "SSIM", "LPIPS")}
        deltas = []
        for c in clips:
            sp, mp, gp = (cell(c, i, dt, "PSNR") for i in inits)
            dl = (gp - sp) if (gp is not None and sp is not None) else None
            if dl is not None:
                deltas.append(dl)
            for i in inits:
                for m in ("PSNR", "SSIM", "LPIPS"):
                    v = cell(c, i, dt, m)
                    if v is not None:
                        acc[(i, m)].append(v)
            L.append(f"| {c} | {fmt(sp)} | {fmt(mp)} | {fmt(gp)} | {fmt(dl,'{:+.2f}')} "
                     f"| {fmt(cell(c,'sfm',dt,'SSIM'),'{:.3f}')} "
                     f"| {fmt(cell(c,'gsnet',dt,'SSIM'),'{:.3f}')} "
                     f"| {fmt(cell(c,'sfm',dt,'LPIPS'),'{:.3f}')} "
                     f"| {fmt(cell(c,'gsnet',dt,'LPIPS'),'{:.3f}')} |")

        def mean(key, f="{:.2f}"):
            v = acc[key]
            return f.format(sum(v) / len(v)) if v else "-"
        dmean = (sum(deltas) / len(deltas)) if deltas else None
        L.append(f"| **mean** | {mean(('sfm','PSNR'))} | {mean(('mvs','PSNR'))} "
                 f"| {mean(('gsnet','PSNR'))} | {fmt(dmean,'{:+.2f}')} "
                 f"| {mean(('sfm','SSIM'),'{:.3f}')} | {mean(('gsnet','SSIM'),'{:.3f}')} "
                 f"| {mean(('sfm','LPIPS'),'{:.3f}')} | {mean(('gsnet','LPIPS'),'{:.3f}')} |")
        L.append("")

    table = "\n".join(L)
    out = args.out or os.path.join(args.root, f"nusc_per_scene_{args.iter}.md")
    open(out, "w").write(table + "\n")
    print(table)
    print(f"\n-> saved {out}")


if __name__ == "__main__":
    main()
