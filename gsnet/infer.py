#
# GS-Net inference (single forward pass): densify a scene's sparse SfM point
# cloud into Gaussian primitives and export a standard 3DGS .ply that the
# existing pipeline consumes via --gsnet_init (see train.py).
#
# Usage:
#   python -m gsnet.infer \
#       --ckpt runs/gsnet/gsnet_latest.pt \
#       --sfm  SCENE/seq/sparse/0/points3D.bin \
#       --out  SCENE/seq/gsnet_init.ply
#

import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsnet.model import GSNet, GSNetConfig
from gsnet.io import save_gaussians_ply
from gsnet.build_correspondences import read_sparse_points


def build_neighbor_graph(xyz, M):
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(M + 1, len(xyz)))
    idx = np.atleast_2d(idx)[:, 1:M + 1]
    if idx.shape[1] < M:
        idx = np.concatenate([idx, np.repeat(idx[:, -1:], M - idx.shape[1], 1)], 1)
    return idx


@torch.no_grad()
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = GSNetConfig(**ckpt["cfg"])
    net = GSNet(cfg).to(device).eval()
    net.load_state_dict(ckpt["model"])

    cxyz, crgb = read_sparse_points(args.sfm)
    nn_idx = build_neighbor_graph(cxyz, cfg.M)
    cxyz_t = torch.from_numpy(cxyz).float()
    crgb_t = torch.from_numpy(crgb).float()
    nxyz_t = torch.from_numpy(cxyz[nn_idx]).float()
    nrgb_t = torch.from_numpy(crgb[nn_idx]).float()

    mus, rgbs, scales, quats, ops = [], [], [], [], []
    N = cxyz.shape[0]
    for s in range(0, N, args.chunk):
        e = min(s + args.chunk, N)
        pred = net(cxyz_t[s:e].to(device), crgb_t[s:e].to(device),
                   nxyz_t[s:e].to(device), nrgb_t[s:e].to(device))
        mus.append(pred["mu"].reshape(-1, 3).cpu().numpy())
        rgbs.append(pred["rgb"].reshape(-1, 3).cpu().numpy())
        scales.append(pred["scale"].reshape(-1, 3).cpu().numpy())
        quats.append(pred["quat"].reshape(-1, 4).cpu().numpy())
        ops.append(pred["opacity"].reshape(-1).cpu().numpy())

    P = save_gaussians_ply(
        args.out,
        np.concatenate(mus), np.concatenate(rgbs), np.concatenate(scales),
        np.concatenate(quats), np.concatenate(ops),
        opacity_thresh=args.opacity_thresh,
    )
    print(f"[infer] {N} sparse pts -> {P} Gaussians (T={cfg.T}) written to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sfm", required=True, help="COLMAP points3D.bin/.txt")
    ap.add_argument("--out", required=True, help="output init .ply")
    ap.add_argument("--chunk", type=int, default=200000)
    ap.add_argument("--opacity_thresh", type=float, default=0.0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
