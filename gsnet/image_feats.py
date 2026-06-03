#
# Per-point IMAGE features for image-conditioned GS-Net.
#
# Why: ablations showed encoder complexity / attribute heads don't move the
# needle -- the sparse points alone are information-saturated. On weak-structure
# real data (Waymo: near-disjoint cameras) GS-Net is "blind". This module gives
# every sparse SfM point a local image descriptor so the network can ground its
# Gaussian prediction in pixels (cf. pixelSplat/MVSplat).
#
# Robust-by-construction: we use each point's OWN 2-D observations recorded in
# COLMAP images.bin (xys + point3D_ids) -- no manual projection (no intrinsics/
# extrinsics math, no distortion) -> no projection bugs. For each observing
# image we sample a multi-scale color/gradient/laplacian descriptor at the
# observed pixel, then average over all images that saw the point.
#
# Features are aligned to read_points3D_binary's file order (== common
# read_points_any order), so build_correspondences/infer can concat them to
# [xyz,rgb] directly.
#

import os
import struct

import numpy as np

from scene.colmap_loader import read_extrinsics_binary, read_next_bytes

FEAT_DIM = 15  # 3 scales x (3 rgb + 1 grad-mag + 1 laplacian-mag); probed at runtime


def _read_point_ids(points3D_bin):
    """Return COLMAP point3D_id in the SAME order read_points3D_binary yields
    rows, so feats[row] aligns with xyz[row]."""
    ids = []
    with open(points3D_bin, "rb") as f:
        n = read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            props = read_next_bytes(f, 43, "QdddBBBd")
            ids.append(int(props[0]))
            tl = read_next_bytes(f, 8, "Q")[0]
            read_next_bytes(f, 8 * tl, "ii" * tl)  # skip track
    return ids


def _feature_map(img):
    """(H,W,3) uint8/float -> (H,W,C) multi-scale descriptor in [0,1]-ish."""
    from scipy.ndimage import gaussian_filter
    im = img.astype(np.float32)
    if im.max() > 1.5:
        im /= 255.0
    gray = im.mean(axis=2)
    chans = []
    for s in (0.0, 2.0, 4.0):
        rgb = im if s == 0 else gaussian_filter(im, sigma=(s, s, 0))
        g = gray if s == 0 else gaussian_filter(gray, sigma=s)
        gy, gx = np.gradient(g)
        grad = np.sqrt(gx * gx + gy * gy)
        lap = gaussian_filter(g, sigma=1.0) - g  # cheap laplacian-ish
        chans += [rgb[..., 0], rgb[..., 1], rgb[..., 2], grad, np.abs(lap)]
    return np.stack(chans, axis=-1).astype(np.float32)   # (H,W,15)... see FEAT_DIM


def compute_point_image_feats(sparse_dir, images_dir, max_imgs=0, verbose=True):
    """Return feats (N, F) aligned to read_points3D_binary order of points3D.bin."""
    pts_bin = os.path.join(sparse_dir, "points3D.bin")
    ids = _read_point_ids(pts_bin)
    id2row = {pid: r for r, pid in enumerate(ids)}
    N = len(ids)

    extr = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    images = list(extr.values())
    if max_imgs:
        images = images[:max_imgs]

    # probe feature dim
    F = None
    acc = None
    cnt = None
    from PIL import Image
    for im in images:
        path = os.path.join(images_dir, im.name)
        if not os.path.exists(path):
            continue
        arr = np.asarray(Image.open(path).convert("RGB"))
        fmap = _feature_map(arr)
        H, W, C = fmap.shape
        if acc is None:
            F = C
            acc = np.zeros((N, F), np.float32)
            cnt = np.zeros(N, np.float32)
        xys = im.xys
        pids = im.point3D_ids
        keep = pids > 0
        if not keep.any():
            continue
        xs = np.clip(np.round(xys[keep, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(xys[keep, 1]).astype(int), 0, H - 1)
        samp = fmap[ys, xs]                                   # (k, F)
        rows = np.array([id2row.get(int(p), -1) for p in pids[keep]])
        ok = rows >= 0
        np.add.at(acc, rows[ok], samp[ok])
        np.add.at(cnt, rows[ok], 1.0)

    if acc is None:
        raise RuntimeError(f"no images found under {images_dir}")
    feats = acc / np.maximum(cnt[:, None], 1.0)
    miss = int((cnt == 0).sum())
    if verbose:
        print(f"[image_feats] {N} pts, F={F}, {len(images)} imgs, "
              f"{miss} pts with no image obs ({100*miss/max(N,1):.1f}%)")
    return feats.astype(np.float32)
