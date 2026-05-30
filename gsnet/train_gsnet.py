#
# Train GS-Net across all training scenes (Sec. V Implementation):
#   200 epochs, batch size 512, Adam lr 1e-3, T=5, M=3.
#
# By default the whole correspondence set (a few hundred MB) is held on-GPU and
# trained with permutation batching (no DataLoader) -- ~100x faster than the
# DataLoader path, which makes encoder / loss-weight sweeps cheap.
#
# Usage:
#   python -m gsnet.train_gsnet --corr_dir CORR/train --out_dir runs/gsnet
#

import argparse
import json
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsnet.model import GSNet, GSNetConfig
from gsnet.losses import gsnet_loss
from gsnet.dataset import CorrespondenceDataset


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GSNetConfig(
        T=args.T, M=args.M,
        embed_dim=args.embed_dim, context_dim=args.context_dim,
        pos_offset_scale=args.pos_offset_scale,
        encoder_type=args.encoder_type,
    )
    n_params = sum(p.numel() for p in GSNet(cfg).parameters())
    print(f"[cfg] encoder={cfg.encoder_type}  params={n_params/1e3:.1f}K  "
          f"weights={args.weights}")

    ds = CorrespondenceDataset(args.corr_dir)
    net = GSNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    os.makedirs(args.out_dir, exist_ok=True)

    def make_gt(b):
        return {"center_xyz": b["center_xyz"], "center_rgb": b["center_rgb"],
                "mu": b["gt_mu"], "rgb": b["gt_rgb"], "scale": b["gt_scale"],
                "quat": b["gt_quat"], "opacity": b["gt_opacity"]}

    def save(tag):
        torch.save({"model": net.state_dict(), "cfg": vars(cfg)},
                   os.path.join(args.out_dir, f"gsnet_{tag}.pt"))
        torch.save({"model": net.state_dict(), "cfg": vars(cfg)},
                   os.path.join(args.out_dir, "gsnet_latest.pt"))

    keys = ["center_xyz", "center_rgb", "neighbor_xyz", "neighbor_rgb",
            "gt_mu", "gt_rgb", "gt_scale", "gt_quat", "gt_opacity"]
    N = ds.n
    bs = args.batch_size
    net.train()
    t_start = time.time()
    epoch_times = []

    if args.in_memory:
        data = {k: ds.data[k].to(device) for k in keys}
        for epoch in range(args.epochs):
            t_ep = time.time()
            perm = torch.randperm(N, device=device)
            running = 0.0
            nb = 0
            for s in range(0, N - bs + 1, bs):
                idx = perm[s:s + bs]
                b = {k: data[k][idx] for k in keys}
                pred = net(b["center_xyz"], b["center_rgb"],
                           b["neighbor_xyz"], b["neighbor_rgb"])
                loss, logs = gsnet_loss(pred, make_gt(b), cfg, weights=args.weights)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += float(loss)
                nb += 1
            epoch_times.append(time.time() - t_ep)
            _log(epoch, args, running / max(1, nb), logs, epoch_times[-1], save)
    else:
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.num_workers,
                        drop_last=True, pin_memory=True)
        for epoch in range(args.epochs):
            t_ep = time.time()
            running = 0.0
            for batch in dl:
                b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                pred = net(b["center_xyz"], b["center_rgb"],
                           b["neighbor_xyz"], b["neighbor_rgb"])
                loss, logs = gsnet_loss(pred, make_gt(b), cfg, weights=args.weights)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += float(loss)
            epoch_times.append(time.time() - t_ep)
            _log(epoch, args, running / max(1, len(dl)), logs, epoch_times[-1], save)

    total = time.time() - t_start
    with open(os.path.join(args.out_dir, "train_times.json"), "w") as f:
        json.dump({"total_seconds": total, "epochs": args.epochs, "num_points": N,
                   "batch_size": bs, "encoder": cfg.encoder_type,
                   "weights": args.weights,
                   "avg_epoch_seconds": sum(epoch_times) / max(1, len(epoch_times))},
                  f, indent=2)
    print(f"Training done in {total:.1f}s "
          f"(avg {sum(epoch_times)/max(1,len(epoch_times)):.2f}s/epoch).")


def _log(epoch, args, avg, logs, dt, save):
    if (epoch + 1) % args.log_every == 0 or epoch == 0:
        print(f"epoch {epoch+1:3d}/{args.epochs}  loss={avg:.5f}  {dt:.2f}s  "
              + "  ".join(f"{k}={float(v):.4f}" for k, v in logs.items() if k != "loss"))
    if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
        save(str(epoch + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_dir", required=True)
    ap.add_argument("--out_dir", default="runs/gsnet")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--context_dim", type=int, default=256)
    ap.add_argument("--pos_offset_scale", type=float, default=1.0)
    ap.add_argument("--encoder_type", default="concat",
                    choices=["mlp_only", "concat", "edgeconv", "attention", "geom"])
    ap.add_argument("--in_memory", type=int, default=1, help="1=hold data on GPU (fast)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--w_pos", type=float, default=1.0)
    ap.add_argument("--w_rot", type=float, default=1.0)
    ap.add_argument("--w_scale", type=float, default=1.0)
    ap.add_argument("--w_rgb", type=float, default=1.0)
    ap.add_argument("--w_opacity", type=float, default=1.0)
    args = ap.parse_args()
    args.in_memory = bool(args.in_memory)
    args.weights = {"pos": args.w_pos, "rot": args.w_rot, "scale": args.w_scale,
                    "rgb": args.w_rgb, "opacity": args.w_opacity}
    train(args)


if __name__ == "__main__":
    main()
