#
# Waymo 5-camera geometry + front/side coverage check.
#
# BEFORE running cross-sensor (reconstruct from front cam0/1/2 -> synthesize side
# cam3/4) we must verify the front cameras actually OBSERVE the regions the side
# cameras see. If side-visible 3D points are almost never seen by the front
# cameras, the front-only reconstruction has NO geometry there -> GS-Net has
# nothing to densify -> cross-sensor is hopeless (don't bother).
#
# Uses COLMAP tracks: each 3D point's observing images are known via images.bin
# (point3D_ids per image). coverage = |side-seen ∩ front-seen| / |side-seen|.
#
# Usage:
#   python -m gsnet.waymo_cam_overlap \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam \
#       --scenes 10275144660749673822_5755_561 15868625208244306149_4340_000 \
#       --front cam0 cam1 cam2 --side cam3 cam4 --out_dir runs/waymo_cam_overlap
#

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from gsnet.common import read_points_any
from gsnet.waymo import (discover_scenes, resolve_scene, seg_name,
                         sparse_model_dir, sparse_points)


def cam_of(name):
    # image name like "cam0/000.jpg" -> "cam0"
    b = name.replace("\\", "/")
    return b.split("/")[0]


def analyze(scene_path, front, side):
    sp0 = sparse_model_dir(scene_path)
    extr = read_extrinsics_binary(os.path.join(sp0, "images.bin"))

    per_cam = {}          # cam -> dict(centers[], dirs[], pts:set)
    for im in extr.values():
        cam = cam_of(im.name)
        R = qvec2rotmat(im.qvec)
        C = -R.T @ im.tvec                       # camera center in world
        d = R.T @ np.array([0, 0, 1.0])          # viewing direction (look +z)
        ids = im.point3D_ids
        ids = ids[ids > 0]
        e = per_cam.setdefault(cam, {"C": [], "d": [], "pts": set()})
        e["C"].append(C); e["d"].append(d); e["pts"].update(int(i) for i in ids)

    rows = {}
    for cam, e in per_cam.items():
        C = np.array(e["C"]); d = np.array(e["d"])
        rows[cam] = dict(n_img=len(C), center=C.mean(0), vdir=d.mean(0),
                         pts=e["pts"])

    front_pts = set().union(*[rows[c]["pts"] for c in front if c in rows])
    side_pts = set().union(*[rows[c]["pts"] for c in side if c in rows])
    inter = front_pts & side_pts
    cov_side_by_front = len(inter) / max(len(side_pts), 1)
    cov_front_by_side = len(inter) / max(len(front_pts), 1)
    return rows, dict(n_front_pts=len(front_pts), n_side_pts=len(side_pts),
                      n_shared=len(inter),
                      side_covered_by_front=cov_side_by_front,
                      front_covered_by_side=cov_front_by_side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", nargs="+", default=[])
    ap.add_argument("--front", nargs="+", default=["cam0", "cam1", "cam2"])
    ap.add_argument("--side", nargs="+", default=["cam3", "cam4"])
    ap.add_argument("--out_dir", default="runs/waymo_cam_overlap")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scenes = ([resolve_scene(args.root, s) for s in args.scenes]
              if args.scenes else discover_scenes(args.root))

    for sc in scenes:
        nm = seg_name(sc)
        rows, ov = analyze(sc, args.front, args.side)
        print(f"\n=== {nm[:40]} ===")
        print("| cam | #img | center(x,y,z) | view dir(x,y,z) | #pts |")
        print("|---|---|---|---|---|")
        for cam in sorted(rows):
            r = rows[cam]
            c = ", ".join(f"{v:.1f}" for v in r["center"])
            d = ", ".join(f"{v:.2f}" for v in r["vdir"])
            print(f"| {cam} | {r['n_img']} | {c} | {d} | {len(r['pts'])} |")
        print(f"front={args.front}  side={args.side}")
        print(f"  side-seen points        : {ov['n_side_pts']}")
        print(f"  also seen by front      : {ov['n_shared']}")
        print(f"  ** side covered by front: {ov['side_covered_by_front']*100:.1f}% **")
        print(f"  front covered by side   : {ov['front_covered_by_side']*100:.1f}%")
        verdict = ("VIABLE (front reconstruction covers much of the side views -> "
                   "GS-Net densification can help)"
                   if ov["side_covered_by_front"] > 0.25 else
                   "WEAK/HOPELESS (side regions barely observed by front -> "
                   "cross-sensor likely ~0 regardless of method)")
        print(f"  VERDICT: {verdict}")

        # --- leave-one-camera-out: which target is best covered by the rest? ---
        cams = sorted(rows)
        print("\n  Leave-one-camera-out coverage (target covered by the OTHER 4):")
        loco = {}
        for t in cams:
            others = set().union(*[rows[c]["pts"] for c in cams if c != t])
            cov = len(rows[t]["pts"] & others) / max(len(rows[t]["pts"]), 1)
            loco[t] = cov
            tag = " <- most viable target" if cov == max(
                len(rows[x]["pts"] & set().union(*[rows[c]["pts"] for c in cams if c != x]))
                / max(len(rows[x]["pts"]), 1) for x in cams) else ""
            print(f"    synth {t}: {cov*100:.1f}% covered by others{tag}")
        # --- pairwise IoU of observed point sets ---
        print("  Pairwise overlap (|ci∩cj|/|ci∪cj|):")
        hdr = "        " + " ".join(f"{c:>5}" for c in cams)
        print(hdr)
        for ci in cams:
            cells = []
            for cj in cams:
                a, b = rows[ci]["pts"], rows[cj]["pts"]
                iou = len(a & b) / max(len(a | b), 1)
                cells.append(f"{iou*100:4.0f}%")
            print(f"    {ci:>4} " + " ".join(cells))

        # top-down scatter (optional)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xyz, _ = read_points_any(sparse_points(sc))
            # subsample for plotting
            rng = np.random.default_rng(0)
            idx = rng.choice(len(xyz), min(30000, len(xyz)), replace=False)
            p = xyz[idx]
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(p[:, 0], p[:, 1], s=1, c="lightgray", label="SfM points")
            cols = {**{c: "tab:blue" for c in args.front},
                    **{c: "tab:red" for c in args.side}}
            for cam in sorted(rows):
                ctr = rows[cam]["center"]; vd = rows[cam]["vdir"]
                col = cols.get(cam, "k")
                ax.scatter(*ctr[:2], c=col, s=120, edgecolors="k", zorder=5)
                ax.annotate(cam, ctr[:2], fontsize=9)
                ax.arrow(ctr[0], ctr[1], vd[0]*5, vd[1]*5, color=col,
                         head_width=1.0, zorder=5)
            ax.set_title(f"{nm[:28]} top-down (blue=front, red=side)\n"
                         f"side covered by front = {ov['side_covered_by_front']*100:.1f}%")
            ax.set_aspect("equal"); ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"{nm[:30]}_topdown.png"), dpi=130)
            plt.close(fig)
        except Exception as e:
            print(f"  [warn] plot skipped: {e}")

    print(f"\n-> plots (if any) in {args.out_dir}/")


if __name__ == "__main__":
    main()
