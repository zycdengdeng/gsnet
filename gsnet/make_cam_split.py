#
# Generate a per-camera train/test split as a COLMAP sparse/0/test.txt, reading
# the *actual* image names from images.bin (robust to subfolder-style names like
# "cam0/000.jpg"). For each camera (grouped by the image-name directory prefix),
# holds out N frames spread uniformly over the interior of the sequence.
#
# Usage:
#   python -m gsnet.make_cam_split --source <scene>/colmap/dense --n_holdout 4
#

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text


def _interior_indices(L, n):
    """n indices spread over the interior of [0, L-1] (avoiding endpoints)."""
    if n >= L:
        return list(range(L))
    return sorted(set(int(round(x)) for x in np.linspace(0, L - 1, n + 2)[1:-1]))


def _holdout_indices(L, n, mode):
    """Which frame indices (per camera) to hold out as test.
    mode='interp' : n interior frames spread out (interpolation; default, easy,
                    no coverage holes -- frames sit between training frames).
    mode='extrap' : the LAST n frames (forward EXTRAPOLATION; the ego drives into
                    a region only glimpsed from afar in training -> real holes,
                    the regime where an init prior can actually help)."""
    if n >= L:
        return list(range(L))
    if mode == "extrap":
        return list(range(L - n, L))
    return _interior_indices(L, n)


def cam_split_test_names(source, n_holdout=4, mode="interp"):
    sparse = os.path.join(source, "sparse", "0")
    try:
        extr = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
    except Exception:
        extr = read_extrinsics_text(os.path.join(sparse, "images.txt"))
    names = [extr[k].name for k in extr]

    groups = {}
    for n in names:
        groups.setdefault(os.path.dirname(n), []).append(n)

    test = []
    for cam, ns in sorted(groups.items()):
        ns = sorted(ns)
        idx = _holdout_indices(len(ns), n_holdout, mode)
        test.extend(ns[i] for i in idx)
    return sorted(test)


def write_cam_split(source, n_holdout=4, mode="interp"):
    names = cam_split_test_names(source, n_holdout, mode)
    out = os.path.join(source, "sparse", "0", "test.txt")
    with open(out, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"[split] {out}: {len(names)} test images ({n_holdout}/camera, {mode}) -> {names}")
    return out


def write_camera_split(source, test_cams):
    """Hold out ENTIRE cameras as test (e.g. side cameras) for cross-sensor
    (front-train -> side-synthesis). test_cams: dir prefixes e.g. ['cam3','cam4']."""
    sparse = os.path.join(source, "sparse", "0")
    try:
        extr = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
    except Exception:
        extr = read_extrinsics_text(os.path.join(sparse, "images.txt"))
    tset = set(test_cams)
    names = sorted(extr[k].name for k in extr if os.path.dirname(extr[k].name) in tset)
    out = os.path.join(sparse, "test.txt")
    with open(out, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"[cam-split] {out}: {len(names)} test images from cameras {test_cams}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="COLMAP dir with sparse/0 (e.g. <scene>/colmap/dense)")
    ap.add_argument("--n_holdout", type=int, default=4)
    args = ap.parse_args()
    write_cam_split(args.source, args.n_holdout)


if __name__ == "__main__":
    main()
