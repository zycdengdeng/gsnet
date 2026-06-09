#
# Prepare our 3DGS render outputs into the Depth_Seg_eval library's expected
# layout:  <out>/<method>/_eval_frames/<camera>/{gen,gt}/<name>.png
# (gen vs gt matched by filename). We build ONE root per method so the library
# is run once per method and the metrics compared (baseline vs GS-Net).
#
# "camera" = sequence id (CARLA CSE) or clip name (nuScenes SSE) — a per-group
# breakdown; the per-image depth/seg/SAM metrics are unaffected by the grouping.
#
# CARLA CSE  (renders at <run>/<seq>/<method>/test/ours_<it>/{renders,gt}):
#   python -m gsnet.prep_downstream --mode cse \
#       --run runs/cse_d2000_s0 --groups 110 210 410 510 \
#       --methods baseline gsnet --out /mnt/zihanw/downstream_cse
#
# nuScenes SSE (renders at <eval>/<clip>/<init>_<dt>/test/ours_<it>/{renders,gt}):
#   python -m gsnet.prep_downstream --mode nusc \
#       --run runs/nusc_filt/eval --groups 348_clip_09 332_clip_09 331_clip_09 299_clip_09 325_clip_09 \
#       --methods sfm gsnet --densify d15000 --out /mnt/zihanw/downstream_nusc
#
import argparse
import os


def src_dirs(args, method, group):
    if args.mode == "cse":
        base = os.path.join(args.run, group, method, "test", f"ours_{args.iter}")
    else:  # nusc: method is the init (sfm/mvs/gsnet), combined with densify tag
        base = os.path.join(args.run, group, f"{method}_{args.densify}", "test", f"ours_{args.iter}")
    return os.path.join(base, "renders"), os.path.join(base, "gt")


def link_pngs(src, dst, copy):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(src)):
        if not f.lower().endswith((".png", ".jpg")):
            continue
        s, d = os.path.abspath(os.path.join(src, f)), os.path.join(dst, f)
        if os.path.lexists(d):
            os.remove(d)
        if copy:
            import shutil
            shutil.copy(s, d)
        else:
            os.symlink(s, d)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cse", "nusc"], required=True)
    ap.add_argument("--run", required=True, help="cse run dir OR nusc eval dir")
    ap.add_argument("--groups", nargs="+", required=True, help="seq ids (cse) or clip names (nusc)")
    ap.add_argument("--methods", nargs="+", default=["baseline", "gsnet"],
                    help="cse: baseline gsnet ; nusc: sfm gsnet (mvs)")
    ap.add_argument("--densify", default="d15000", help="(nusc) densify tag")
    ap.add_argument("--iter", type=int, default=30000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    args = ap.parse_args()

    roots = {}
    for method in args.methods:
        total = 0
        for g in args.groups:
            rd, gd = src_dirs(args, method, g)
            if not (os.path.isdir(rd) and os.path.isdir(gd)):
                print(f"[skip] {method}/{g}: missing {rd} or {gd}")
                continue
            ef = os.path.join(args.out, method, "_eval_frames", g)
            ng = link_pngs(rd, os.path.join(ef, "gen"), args.copy)
            nt = link_pngs(gd, os.path.join(ef, "gt"), args.copy)
            assert ng == nt, f"{method}/{g}: gen {ng} != gt {nt}"
            total += ng
        root = os.path.join(args.out, method, "_eval_frames")
        roots[method] = root
        print(f"[{method}] {total} frame pairs -> {root}")

    print("\n=== point the Depth_Seg_eval library (config.yaml) at each root ===")
    for m, r in roots.items():
        print(f"  {m:10s}: {r}")
    print("Run the library once per root; compare baseline-vs-gsnet metrics "
          "(gsnet should have lower depth abs_rel / higher seg mIoU / higher SAM edge_f1).")


if __name__ == "__main__":
    main()
