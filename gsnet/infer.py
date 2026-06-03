#
# GS-Net inference (single forward pass): densify a sequence's sparse SfM point
# cloud into Gaussian primitives and export a standard 3DGS .ply consumed by the
# pipeline via --gsnet_init (see train.py / scene/__init__.py).
#
# Operates in the same per-sequence normalized frame as training, then inverts
# the normalization so the exported Gaussians live in the original COLMAP frame
# (matching the cameras for subsequent 3DGS optimization).
#
# Usage:
#   python -m gsnet.infer --ckpt runs/gsnet/gsnet_latest.pt \
#       --sparse sparse_point/S01/110_sparse.ply \
#       --out    input_output/110_base/gsnet_init.ply
#

import argparse
import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from gsnet.model import GSNet, GSNetConfig
from gsnet.io import save_gaussians_ply
from gsnet.common import read_points_any, compute_normalization


def build_neighbor_graph(xyz, M):
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(M + 1, len(xyz)))
    idx = np.atleast_2d(idx)[:, 1:M + 1]
    if idx.shape[1] < M:
        idx = np.concatenate([idx, np.repeat(idx[:, -1:], M - idx.shape[1], 1)], 1)
    return idx


@torch.no_grad()
def run(args):
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = GSNetConfig(**ckpt["cfg"])
    net = GSNet(cfg).to(device).eval()
    net.load_state_dict(ckpt["model"])

    cxyz, crgb = read_points_any(args.sparse)
    if args.no_normalize:
        center, scale = np.zeros(3, np.float32), 1.0
    else:
        center, scale = compute_normalization(cxyz)
    n_cxyz = (cxyz - center) / scale

    nn_idx = build_neighbor_graph(n_cxyz, cfg.M)
    cxyz_t = torch.from_numpy(n_cxyz).float()
    crgb_t = torch.from_numpy(crgb).float()
    nxyz_t = torch.from_numpy(n_cxyz[nn_idx]).float()
    nrgb_t = torch.from_numpy(crgb[nn_idx]).float()

    # Image-conditioning: if the model expects image features, compute them for
    # this scene (same images.bin-based featurizer used to build the corr).
    cimg_t = nimg_t = None
    if int(getattr(cfg, "feat_dim", 0)) > 0:
        assert args.image_feats, "model has feat_dim>0; pass --image_feats <images_dir>"
        from gsnet.image_feats import compute_point_image_feats
        feats = compute_point_image_feats(os.path.dirname(args.sparse), args.image_feats)
        assert feats.shape == (cxyz.shape[0], cfg.feat_dim), \
            f"feat shape {feats.shape} != ({cxyz.shape[0]}, {cfg.feat_dim})"
        cimg_t = torch.from_numpy(feats).float()
        nimg_t = torch.from_numpy(feats[nn_idx]).float()

    mus, rgbs, scales, quats, ops = [], [], [], [], []
    N = cxyz.shape[0]
    for s in range(0, N, args.chunk):
        e = min(s + args.chunk, N)
        pred = net(cxyz_t[s:e].to(device), crgb_t[s:e].to(device),
                   nxyz_t[s:e].to(device), nrgb_t[s:e].to(device),
                   center_img=(cimg_t[s:e].to(device) if cimg_t is not None else None),
                   neighbor_img=(nimg_t[s:e].to(device) if nimg_t is not None else None))
        # Denormalize geometry back to the original COLMAP frame.
        mu = pred["mu"].reshape(-1, 3).cpu().numpy() * scale + center
        sc = pred["scale"].reshape(-1, 3).cpu().numpy() * scale
        mus.append(mu)
        scales.append(sc)
        rgbs.append(pred["rgb"].reshape(-1, 3).cpu().numpy())
        quats.append(pred["quat"].reshape(-1, 4).cpu().numpy())
        ops.append(pred["opacity"].reshape(-1).cpu().numpy())

    mu_all = np.concatenate(mus)
    scale_all = np.concatenate(scales)
    # When scale/rotation are NOT predicted (ablation: --no_scale_rot / dens_only
    # / xyz_rgb), the model emits a fixed isotropic default scale. That is an
    # unfair, non-adaptive init. Instead fall back to the STANDARD 3DGS init
    # heuristic on the (denser) predicted points -- per-point distCUDA2 scale --
    # so "pure densification" is initialized exactly like create_from_pcd would.
    if not cfg.predict_scale_rot and not getattr(args, "keep_default_scale", False):
        tree = cKDTree(mu_all)
        k = min(4, len(mu_all))
        dd, _ = tree.query(mu_all, k=k)               # col 0 is self (dist 0)
        sq = (dd[:, 1:] ** 2).mean(axis=1) if k > 1 else np.full(len(mu_all), 1e-6)
        s_lin = np.sqrt(np.clip(sq, 1e-14, None)).astype(np.float32)
        scale_all = np.repeat(s_lin[:, None], 3, axis=1)
        print(f"[infer] scale/rot not predicted -> distCUDA2 std-init scale "
              f"(median={np.median(s_lin):.4f})")

    P = save_gaussians_ply(
        args.out,
        mu_all, np.concatenate(rgbs), scale_all,
        np.concatenate(quats), np.concatenate(ops),
        opacity_thresh=args.opacity_thresh,
    )
    dt = time.time() - t0
    print(f"[infer] {N} sparse pts -> {P} Gaussians (T={cfg.T}) in {dt:.1f}s -> {args.out}")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sparse", required=True, help="sparse SfM .ply/.bin/.txt")
    ap.add_argument("--out", required=True, help="output init .ply")
    ap.add_argument("--chunk", type=int, default=200000)
    ap.add_argument("--opacity_thresh", type=float, default=0.0)
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--image_feats", default="",
                    help="images dir for image-conditioned models (feat_dim>0); "
                         "must match how the corr was built")
    ap.add_argument("--keep_default_scale", action="store_true",
                    help="keep the fixed default scale when scale/rot are not "
                         "predicted (default: use distCUDA2 std-init scale instead)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
