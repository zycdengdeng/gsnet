#
# Derive the EVEN (target) camera poses in the ODD-camera COLMAP frame for
# Cross-Sensor Evaluation (CSE), using the known CARLA rig + the existing odd
# COLMAP poses -- WITHOUT re-running SfM (which would change the frame and break
# G_dense / GS-Net consistency).
#
# Per frame, all 12 cameras are rigidly attached to the ego. We fit the per-frame
# ego->COLMAP similarity from the 6 odd cameras (Umeyama on centers) and apply it
# to the even cameras' known ego-frame mounting to get their COLMAP poses.
#
# The CARLA<->COLMAP axis convention (CARLA is left-handed; optical frames differ)
# is found automatically by searching signed-permutation basis changes (A on the
# ego/world frame, D on the camera/optical frame) and picking the one that best
# reproduces the odd cameras' true COLMAP poses (self-validation). For an exact
# rig this residual should be ~0 (sub-degree, sub-percent).
#
# Image-name convention for the odd model (input_output/<id>_base): "<n>.png",
# n=1..60, where cam = 2*((n-1)//10)+1 (odd cam number) and frame = (n-1)%10.
#
# Usage (batch over the 5 CARLA test sequences):
#   python -m gsnet.cse_poses --io_dir /mnt/zihanw/carla/input_output \
#       --rig gsnet/carla_rig.json --ids 110 210 310 410 510 \
#       --out_dir runs/cse_poses
#

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                 qvec2rotmat)


# --------------------------------------------------------------------------- #
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q = -q
    return q


def carla_R(pitch, yaw, roll):
    """CARLA actor-to-parent rotation matrix (degrees -> matrix, per CARLA source)."""
    p, y, r = np.deg2rad([pitch, yaw, roll])
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    cr, sr = np.cos(r), np.sin(r)
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp,      -cp * sr,                cp * cr]])


def umeyama(src, dst):
    """Proper-rotation similarity (s, R, t) with dst ~= s*R*src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (sc ** 2).sum() / len(src)
    scale = (D * np.diag(S)).sum() / var
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


def signed_perms():
    """All 48 signed permutation 3x3 matrices."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for row, col in enumerate(perm):
                M[row, col] = signs[row]
            out.append(M)
    return out


# --------------------------------------------------------------------------- #
def load_odd(model_dir):
    """Return {cam_num: {frame: (Rc2w(3,3), center(3,))}} and intrinsics dict."""
    extr = read_extrinsics_binary(os.path.join(model_dir, "images.bin"))
    intr = read_intrinsics_binary(os.path.join(model_dir, "cameras.bin"))
    odd = {}
    for k in extr:
        im = extr[k]
        n = int(os.path.splitext(os.path.basename(im.name))[0])
        cam = 2 * ((n - 1) // 10) + 1
        frame = (n - 1) % 10
        Rwc = qvec2rotmat(im.qvec)
        center = -Rwc.T @ np.array(im.tvec)
        odd.setdefault(cam, {})[frame] = (Rwc.T, center)
    return odd, intr


def load_rig(rig_path):
    rig = json.load(open(rig_path))
    out = {}
    for name, v in rig.items():
        if not name.startswith("cam"):
            continue
        rot = v["rotation"]
        out[int(name[3:])] = (np.array(v["location"], float),
                              carla_R(rot["pitch"], rot["yaw"], rot["roll"]))
    return out


def derive_sequence(model_dir, rig):
    odd_poses, intr = load_odd(model_dir)
    odd_cams = sorted(odd_poses.keys())             # 1,3,5,7,9,11
    even_cams = [c + 1 for c in odd_cams]           # 2,4,6,8,10,12
    frames = sorted(next(iter(odd_poses.values())).keys())
    scene_scale = np.std([c for cam in odd_poses for _, c in odd_poses[cam].values()])

    perms = signed_perms()
    best = None
    for A in perms:
        for D in perms:
            Rp = {c: A @ rig[c][1] @ D for c in rig}        # camera-optical -> ego
            ep = {c: A @ rig[c][0] for c in rig}            # center in ego
            rot_err, ctr_err, npts = 0.0, 0.0, 0
            ok = True
            for f in frames:
                src = np.array([ep[c] for c in odd_cams])
                dst = np.array([odd_poses[c][f][1] for c in odd_cams])
                s, Rf, t = umeyama(src, dst)
                for c in odd_cams:
                    pred_R = Rf @ Rp[c]
                    rot_err += angle_deg(pred_R, odd_poses[c][f][0])
                    pred_c = s * Rf @ ep[c] + t
                    ctr_err += np.linalg.norm(pred_c - odd_poses[c][f][1])
                    npts += 1
            score = rot_err / npts + (ctr_err / npts) / scene_scale * 90.0
            if best is None or score < best[0]:
                best = (score, A, D, rot_err / npts, ctr_err / npts)

    _, A, D, rot_res, ctr_res = best
    Rp = {c: A @ rig[c][1] @ D for c in rig}
    ep = {c: A @ rig[c][0] for c in rig}

    even = []
    for f in frames:
        src = np.array([ep[c] for c in odd_cams])
        dst = np.array([odd_poses[c][f][1] for c in odd_cams])
        s, Rf, t = umeyama(src, dst)
        for c in even_cams:
            Rc2w = Rf @ Rp[c]
            center = s * Rf @ ep[c] + t
            Rwc = Rc2w.T
            tvec = -Rwc @ center
            even.append({"cam": c, "frame": int(f),
                         "qvec": rotmat2qvec(Rwc).tolist(),
                         "tvec": tvec.tolist()})
    report = {"rot_residual_deg": float(rot_res),
              "center_residual": float(ctr_res),
              "center_residual_rel": float(ctr_res / scene_scale),
              "scene_scale": float(scene_scale)}
    return {"even_poses": even, "report": report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--rig", default="gsnet/carla_rig.json")
    ap.add_argument("--ids", nargs="+", default=["110", "210", "310", "410", "510"])
    ap.add_argument("--out_dir", default="runs/cse_poses")
    args = ap.parse_args()

    rig = load_rig(args.rig)
    os.makedirs(args.out_dir, exist_ok=True)
    for sid in args.ids:
        model = os.path.join(args.io_dir, f"{sid}_base", "sparse", "0")
        res = derive_sequence(model, rig)
        with open(os.path.join(args.out_dir, f"{sid}.json"), "w") as f:
            json.dump(res, f, indent=2)
        r = res["report"]
        print(f"[{sid}] self-validation: rot={r['rot_residual_deg']:.3f} deg, "
              f"center_rel={r['center_residual_rel']*100:.3f}%  "
              f"({len(res['even_poses'])} even poses)")


if __name__ == "__main__":
    main()
