#
# Iterative / recursive GS-Net densification (idea A): instead of expanding the
# SfM points ONCE, run GS-Net for several passes -- after each pass keep the
# HIGH-CONFIDENCE (high-opacity) predicted Gaussians and feed their (xyz,rgb)
# back in as the next pass's input. This lets density/coverage grow BEYOND the
# SfM anchors (each pass seeds new points in the gaps). The final init = the
# union of all passes' visible Gaussians.
#
# Inference-time only (no retraining) -- a quick test of the idea. Risk: the net
# was trained on SfM-density inputs, so feeding its own (denser) output is
# out-of-distribution; high-confidence selection is the guard. (Idea B trains the
# net to be density-agnostic so recursion is in-distribution.)
#
# Usage:
#   python -m gsnet.infer_iterative --ckpt runs/nusc/model/gsnet_latest.pt \
#       --sparse <clip>/colmap/dense/sparse/0/points3D.bin \
#       --out init.ply --passes 2 --conf_thresh 0.5 --norm_scale 45
#
import argparse
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from gsnet.model import GSNet, GSNetConfig
from gsnet.io import save_gaussians_ply, save_xyzrgb_ply
from gsnet.common import read_points_any, compute_normalization


def neighbor_graph(xyz, M):
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=min(M + 1, len(xyz)))
    idx = np.atleast_2d(idx)[:, 1:M + 1]
    if idx.shape[1] < M:
        idx = np.concatenate([idx, np.repeat(idx[:, -1:], M - idx.shape[1], axis=1)], axis=1)
    return idx


@torch.no_grad()
def one_pass(net, cfg, xyz, rgb, norm_scale, device, chunk):
    if norm_scale > 0:
        center, scale = np.median(xyz, axis=0).astype(np.float32), float(norm_scale)
    else:
        center, scale = compute_normalization(xyz)
    n = ((xyz - center) / scale).astype(np.float32)
    nn = neighbor_graph(n, cfg.M)
    cxyz = torch.from_numpy(n).float(); crgb = torch.from_numpy(rgb).float()
    nxyz = torch.from_numpy(n[nn]).float(); nrgb = torch.from_numpy(rgb[nn]).float()
    mus, rgbs, scs, qs, ops = [], [], [], [], []
    for s in range(0, xyz.shape[0], chunk):
        e = min(s + chunk, xyz.shape[0])
        pred = net(cxyz[s:e].to(device), crgb[s:e].to(device),
                   nxyz[s:e].to(device), nrgb[s:e].to(device),
                   center_img=None, neighbor_img=None)
        mus.append(pred["mu"].reshape(-1, 3).cpu().numpy() * scale + center)
        scs.append(pred["scale"].reshape(-1, 3).cpu().numpy() * scale)
        rgbs.append(pred["rgb"].reshape(-1, 3).cpu().numpy())
        qs.append(pred["quat"].reshape(-1, 4).cpu().numpy())
        ops.append(pred["opacity"].reshape(-1).cpu().numpy())
    return (np.concatenate(mus), np.concatenate(rgbs), np.concatenate(scs),
            np.concatenate(qs), np.concatenate(ops))


def run(args):
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = GSNetConfig(**ckpt["cfg"])
    net = GSNet(cfg).to(device).eval()
    net.load_state_dict(ckpt["model"])
    assert int(getattr(cfg, "feat_dim", 0)) == 0, "image-conditioned ckpt not supported here"

    xyz, rgb = read_points_any(args.sparse)
    allm, allr, alls, allq, allo = [], [], [], [], []
    cur_xyz, cur_rgb = xyz.astype(np.float32), rgb.astype(np.float32)
    for p in range(args.passes):
        mu, rg, sc, q, op = one_pass(net, cfg, cur_xyz, cur_rgb, args.norm_scale, device, args.chunk)
        allm.append(mu); allr.append(rg); alls.append(sc); allq.append(q); allo.append(op)
        keep = op > args.conf_thresh
        print(f"[iter] pass {p+1}/{args.passes}: {len(cur_xyz)} in -> {len(mu)} pred, "
              f"{int(keep.sum())} high-conf (op>{args.conf_thresh})", flush=True)
        if keep.sum() == 0 or p == args.passes - 1:
            break
        cur_xyz, cur_rgb = mu[keep].astype(np.float32), rg[keep].astype(np.float32)

    mu = np.concatenate(allm); rg = np.concatenate(allr); sc = np.concatenate(alls)
    q = np.concatenate(allq); op = np.concatenate(allo)
    # distCUDA2 scale fallback if the model doesn't predict scale (parity w/ infer)
    if not getattr(cfg, "predict_scale_rot", True):
        tree = cKDTree(mu); k = min(4, len(mu))
        dd, _ = tree.query(mu, k=k)
        s_lin = np.sqrt(np.clip((dd[:, 1:] ** 2).mean(axis=1), 1e-14, None)).astype(np.float32)
        sc = np.repeat(s_lin[:, None], 3, axis=1)

    P = 0
    if args.out:
        P = save_gaussians_ply(args.out, mu, rg, sc, q, op, opacity_thresh=args.opacity_thresh)
    if args.points_out:
        kk = op > args.opacity_thresh
        save_xyzrgb_ply(args.points_out, mu[kk], rg[kk])
    print(f"[iter] {args.passes} passes -> {len(mu)} gaussians ({P} saved) "
          f"in {time.time()-t0:.1f}s -> {args.out}", flush=True)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--points_out", default="")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--conf_thresh", type=float, default=0.5,
                    help="opacity threshold to keep a Gaussian as input to the next pass")
    ap.add_argument("--opacity_thresh", type=float, default=0.0, help="final-save opacity cut")
    ap.add_argument("--norm_scale", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=200000)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
