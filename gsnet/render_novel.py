#
# Qualitative cross-sensor / extrapolation viz (no GT, no metric): render the
# baseline-init and GS-Net-init 3DGS reconstructions from NOVEL camera poses
# (a base camera offset by translations + rotations, i.e. viewpoints NOT in the
# training set). GS-Net's denser/more-complete init should show fewer holes /
# floaters at extrapolated views. Output = side-by-side [baseline | GS-Net] PNGs.
#
# Usage:
#   python -m gsnet.render_novel \
#     --colmap /mnt/zihanw/gsnet_nusc/348_clip_09/colmap/dense/sparse/0 \
#     --base cam0/005.jpg \
#     --ply_baseline runs/nusc_filt/eval/348_clip_09/sfm_d15000/point_cloud/iteration_30000/point_cloud.ply \
#     --ply_gsnet   runs/nusc_filt/eval/348_clip_09/gsnet_d15000/point_cloud/iteration_30000/point_cloud.ply \
#     --out_dir /mnt/zihanw/nusc_novel/348_front
#
import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from PIL import Image

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.graphics_utils import focal2fov

PIPE = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                       debug=False, antialiasing=False)

# camera frame = OpenCV (x right, y down, z forward). dpos in METERS in cam frame
# (left=-x, up=-y, forward=+z). dyaw about y (down) axis, dpitch about x (right).
OFFSETS = [
    ("00_orig",        (0, 0, 0),    0,   0),
    ("01_left1",       (-1, 0, 0),   0,   0),
    ("02_left2",       (-2, 0, 0),   0,   0),
    ("03_right1",      (1, 0, 0),    0,   0),
    ("04_right2",      (2, 0, 0),    0,   0),
    ("05_up1",         (0, -1, 0),   0,   0),
    ("06_up2",         (0, -2, 0),   0,   0),
    ("07_fwd2",        (0, 0, 2),    0,   0),
    ("08_fwd4",        (0, 0, 4),    0,   0),
    ("09_back2",       (0, 0, -2),   0,   0),
    ("10_yawL10",      (0, 0, 0),  -10,   0),
    ("11_yawL25",      (0, 0, 0),  -25,   0),
    ("12_yawR10",      (0, 0, 0),   10,   0),
    ("13_yawR25",      (0, 0, 0),   25,   0),
    ("14_pitchUp10",   (0, 0, 0),    0, -10),
    ("15_pitchDn10",   (0, 0, 0),    0,  10),
    ("16_left2_yawR15",(-2, 0, 0),  15,   0),
    ("17_up1_pitchDn10",(0, -1, 0),  0,  10),
    ("18_fwd3_left1",  (-1, 0, 3),   0,   0),
    ("19_right2_yawL15",(2, 0, 0),  -15,  0),
]


def _Rx(a):
    a = np.deg2rad(a); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
    a = np.deg2rad(a); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def load_g(ply):
    g = GaussianModel(3)
    g.load_ply(ply)
    g.active_sh_degree = g.max_sh_degree
    return g


def get_base(colmap_dir, substr):
    extr = read_extrinsics_binary(os.path.join(colmap_dir, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(colmap_dir, "cameras.bin"))
    im = next(e for e in sorted(extr.values(), key=lambda x: x.name) if substr in e.name)
    return qvec2rotmat(im.qvec), np.array(im.tvec, float), intr[im.camera_id]


def make_cam(Rw2c, Tw2c, cam):
    fx = cam.params[0]
    fy = cam.params[1] if cam.model == "PINHOLE" else cam.params[0]
    FovX, FovY = focal2fov(fx, cam.width), focal2fov(fy, cam.height)
    dummy = Image.new("RGB", (cam.width, cam.height))
    return Camera((cam.width, cam.height), 0, Rw2c.T, Tw2c, FovX, FovY,
                  None, dummy, None, "novel", 0)


def perturb(Rw2c, Tw2c, dpos, dyaw, dpitch):
    C = -Rw2c.T @ Tw2c
    C = C + Rw2c.T @ np.array(dpos, float)          # offset in cam frame -> world
    Rn = (_Rx(dpitch) @ _Ry(dyaw)) @ Rw2c           # rotate the camera
    return Rn, -Rn @ C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True, help="<clip>/colmap/dense/sparse/0")
    ap.add_argument("--base", default="cam0/005.jpg", help="base camera image name substring")
    ap.add_argument("--ply_baseline", required=True)
    ap.add_argument("--ply_gsnet", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--white_bg", action="store_true")
    ap.add_argument("--dump_poses", default="",
                    help="also write the exact novel poses (world_to_cam 4x4 + intrinsics) "
                         "to this JSON so collaborators can render OTHER methods at the "
                         "same viewpoints")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    Rw2c, Tw2c, cam = get_base(args.colmap, args.base)
    fx = float(cam.params[0])
    fy = float(cam.params[1]) if cam.model == "PINHOLE" else float(cam.params[0])
    cx = float(cam.params[2]) if cam.model == "PINHOLE" else float(cam.params[1])
    cy = float(cam.params[3]) if cam.model == "PINHOLE" else float(cam.params[2])

    if args.dump_poses:
        import json
        out = []
        for tag, dpos, dyaw, dpitch in OFFSETS:
            Rn, Tn = perturb(Rw2c, Tw2c, dpos, dyaw, dpitch)
            w2c = np.eye(4); w2c[:3, :3] = Rn; w2c[:3, 3] = Tn
            out.append({"tag": tag, "dpos_cam_m": list(dpos), "dyaw_deg": dyaw,
                        "dpitch_deg": dpitch, "world_to_cam": w2c.tolist(),
                        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                        "width": cam.width, "height": cam.height})
        json.dump({"base_image": args.base, "convention": "OpenCV cam frame "
                   "(x right,y down,z forward); world_to_cam is 4x4 [R|t], "
                   "x_cam = R @ x_world + t", "views": out},
                  open(args.dump_poses, "w"), indent=2)
        print(f"[novel] dumped {len(out)} poses -> {args.dump_poses}")

    gb, gg = load_g(args.ply_baseline), load_g(args.ply_gsnet)
    bg = torch.tensor([1., 1, 1] if args.white_bg else [0., 0, 0], device="cuda")
    print(f"[novel] base={args.base} {cam.width}x{cam.height}; "
          f"baseline {gb.get_xyz.shape[0]} / gsnet {gg.get_xyz.shape[0]} gaussians")

    with torch.no_grad():
        for tag, dpos, dyaw, dpitch in OFFSETS:
            Rn, Tn = perturb(Rw2c, Tw2c, dpos, dyaw, dpitch)
            c = make_cam(Rn, Tn, cam)
            ib = render(c, gb, PIPE, bg)["render"].clamp(0, 1)
            ig = render(c, gg, PIPE, bg)["render"].clamp(0, 1)
            pad = torch.ones(3, ib.shape[1], 6, device=ib.device)
            torchvision.utils.save_image(torch.cat([ib, pad, ig], dim=2),
                                         os.path.join(args.out_dir, f"{tag}.png"))
    print(f"[novel] {len(OFFSETS)} views (left=baseline | right=GS-Net) -> {args.out_dir}")


if __name__ == "__main__":
    main()
