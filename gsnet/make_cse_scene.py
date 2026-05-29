#
# Build a CSE COLMAP scene per test sequence: 60 odd cameras (TRAIN, original
# COLMAP poses from <id>_base) + 60 even cameras (TEST, poses derived by
# cse_poses.py). Writes a text COLMAP model + an images/ folder (symlinks) +
# test.txt listing the even images, ready for train.py --eval.
#
# Init point cloud = the odd sparse SfM points (copied), so baseline 3DGS keeps
# its standard sparse initialization; GS-Net+3DGS overrides via --gsnet_init.
#
# Usage:
#   python -m gsnet.make_cse_scene \
#       --io_dir /mnt/zihanw/carla/input_output \
#       --paired_dir /mnt/zihanw/carla/paired_120 \
#       --poses_dir runs/cse_poses --out_dir runs/cse_scenes \
#       --ids 110 210 310 410 510
#

import argparse
import json
import os
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary


def _symlink(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def build_scene(sid, io_dir, paired_dir, poses_dir, out_dir):
    odd_model = os.path.join(io_dir, f"{sid}_base", "sparse", "0")
    odd_images = os.path.join(io_dir, f"{sid}_base", "images")
    extr = read_extrinsics_binary(os.path.join(odd_model, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(odd_model, "cameras.bin"))
    cam = list(intr.values())[0]   # all 12 cameras share intrinsics

    out = os.path.join(out_dir, sid)
    img_dir = os.path.join(out, "images")
    sp_dir = os.path.join(out, "sparse", "0")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(sp_dir, exist_ok=True)

    # cameras.txt (single shared intrinsic, camera id 1)
    with open(os.path.join(sp_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        f.write(f"1 {cam.model} {cam.width} {cam.height} "
                + " ".join(str(p) for p in cam.params) + "\n")

    lines = ["# Image list"]
    iid = 0
    # --- odd images: TRAIN (original COLMAP poses) ---
    for k in sorted(extr, key=lambda x: extr[x].name):
        im = extr[k]
        iid += 1
        q = im.qvec
        t = im.tvec
        lines.append(f"{iid} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {im.name}")
        lines.append("")
        _symlink(os.path.join(odd_images, im.name), os.path.join(img_dir, im.name))

    # --- even images: TEST (derived poses) ---
    poses = json.load(open(os.path.join(poses_dir, f"{sid}.json")))["even_poses"]
    # Index even-camera frames by sorted position in the camera dir (robust to the
    # exact file naming / start index used in paired_120).
    cam_files = {}
    for c in sorted({p["cam"] for p in poses}):
        cam_dir = os.path.join(paired_dir, f"{sid}_dense", f"cam{c:02d}")
        if not os.path.isdir(cam_dir):
            raise FileNotFoundError(
                f"missing {cam_dir}; available: "
                f"{sorted(os.listdir(os.path.join(paired_dir, f'{sid}_dense')))}")
        fs = sorted(f for f in os.listdir(cam_dir) if f.lower().endswith((".png", ".jpg")))
        if not fs:
            raise FileNotFoundError(f"no images in {cam_dir}")
        cam_files[c] = (cam_dir, fs)

    test_names = []
    for p in poses:
        c, fr = p["cam"], p["frame"]
        cam_dir, fs = cam_files[c]
        if fr >= len(fs):
            raise IndexError(f"cam{c:02d} has {len(fs)} frames, need frame {fr}")
        name = f"e{c:02d}_{fr:02d}.png"
        src = os.path.join(cam_dir, fs[fr])
        _symlink(src, os.path.join(img_dir, name))
        iid += 1
        q, t = p["qvec"], p["tvec"]
        lines.append(f"{iid} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}")
        lines.append("")
        test_names.append(name)

    with open(os.path.join(sp_dir, "images.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # init point cloud = odd sparse SfM points
    for cand in ("points3D.ply", "points3D.bin", "points3D.txt"):
        srcp = os.path.join(odd_model, cand)
        if os.path.exists(srcp):
            shutil.copy(srcp, os.path.join(sp_dir, cand))
            break

    with open(os.path.join(sp_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_names) + "\n")

    print(f"[cse_scene] {sid}: 60 odd (train) + {len(test_names)} even (test) -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--paired_dir", required=True)
    ap.add_argument("--poses_dir", default="runs/cse_poses")
    ap.add_argument("--out_dir", default="runs/cse_scenes")
    ap.add_argument("--ids", nargs="+", default=["110", "210", "310", "410", "510"])
    args = ap.parse_args()
    for sid in args.ids:
        build_scene(sid, args.io_dir, args.paired_dir, args.poses_dir, args.out_dir)


if __name__ == "__main__":
    main()
