#
# Aggregate a multiseed nuScenes SSE eval into a clear, per-seed + per-clip
# record (saved to a .md, not just printed). Each seed's SfM / MVS / GS-Net and
# the GSNet-SfM delta are listed explicitly, then mean +- std across seeds.
#
# Usage:
#   python -m gsnet.dump_multiseed \
#       --runs runs/nusc_filt runs/nusc_filt_s1 runs/nusc_filt_s2 runs/nusc_filt_s3 \
#       --clips 348_clip_09 332_clip_09 331_clip_09 299_clip_09 325_clip_09 \
#       --densify d5000 d15000 --iter 30000 --out runs/nusc_filt_multiseed.md
#
import argparse
import json
import os
import statistics


def load(run):
    p = os.path.join(run, "nusc_eval.json")
    return json.load(open(p)) if os.path.exists(p) else None


def psnr(d, clip, init, dt, it):
    r = d.get(f"{clip}|{init}|{dt}", {}).get("iters", {}).get(str(it))
    return r["PSNR"] if r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="seed run dirs (s0 first)")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--densify", nargs="+", default=["d5000", "d15000"])
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--out", default="runs/nusc_filt_multiseed.md")
    args = ap.parse_args()

    data = [(os.path.basename(r), load(r)) for r in args.runs]
    data = [(n, d) for n, d in data if d is not None]
    L = ["", "# nuScenes filter SSE — multiseed (per-seed recorded explicitly)", "",
         f"runs: {[n for n, _ in data]}  clips: {len(args.clips)}", ""]

    for dt in args.densify:
        L += [f"## densify {dt} @ {args.iter}", "",
              "| seed | SfM | MVS | GS-Net | GSNet-SfM |", "|---|---|---|---|---|"]
        per_seed_delta = []
        per_seed_sfm, per_seed_gs, per_seed_mvs = [], [], []
        for name, d in data:
            def avg(init):
                v = [psnr(d, c, init, dt, args.iter) for c in args.clips]
                v = [x for x in v if x is not None]
                return sum(v) / len(v) if v else float("nan")
            s, m, g = avg("sfm"), avg("mvs"), avg("gsnet")
            per_seed_sfm.append(s); per_seed_mvs.append(m); per_seed_gs.append(g)
            per_seed_delta.append(g - s)
            L.append(f"| {name} | {s:.2f} | {m:.2f} | {g:.2f} | {g-s:+.2f} |")
        # mean +- std row
        def ms(v):
            v = [x for x in v if x == x]
            return (sum(v) / len(v), statistics.pstdev(v) if len(v) > 1 else 0.0)
        sm, ss = ms(per_seed_sfm); mm, ms_ = ms(per_seed_mvs)
        gm, gs = ms(per_seed_gs); dm, dd = ms(per_seed_delta)
        L.append(f"| **mean±std** | **{sm:.2f}±{ss:.2f}** | **{mm:.2f}±{ms_:.2f}** | "
                 f"**{gm:.2f}±{gs:.2f}** | **{dm:+.2f}±{dd:.2f}** |")
        L += ["", "per-clip GSNet-SfM per seed:", "",
              "| clip | " + " | ".join(n for n, _ in data) + " |",
              "|---|" + "---|" * len(data)]
        for c in args.clips:
            cells = []
            for _, d in data:
                s, g = psnr(d, c, "sfm", dt, args.iter), psnr(d, c, "gsnet", dt, args.iter)
                cells.append(f"{g-s:+.2f}" if (s and g) else "-")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L.append("")

    table = "\n".join(L)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write(table + "\n")
    print(table + f"\n\n-> saved {args.out}")


if __name__ == "__main__":
    main()
