#
# Train GS-Net across all training scenes (Sec. V Implementation):
#   200 epochs, batch size 512, Adam lr 1e-3, T=5, M=3.
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
from torch.utils.data import DataLoader

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
    )

    ds = CorrespondenceDataset(args.corr_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True, pin_memory=True)

    net = GSNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    net.train()
    t_start = time.time()
    epoch_times = []
    for epoch in range(args.epochs):
        t_ep = time.time()
        running = 0.0
        for batch in dl:
            b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            pred = net(b["center_xyz"], b["center_rgb"],
                       b["neighbor_xyz"], b["neighbor_rgb"])
            gt = {
                "center_xyz": b["center_xyz"], "center_rgb": b["center_rgb"],
                "mu": b["gt_mu"], "rgb": b["gt_rgb"], "scale": b["gt_scale"],
                "quat": b["gt_quat"], "opacity": b["gt_opacity"],
            }
            loss, logs = gsnet_loss(pred, gt, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss)
        avg = running / max(1, len(dl))
        dt_ep = time.time() - t_ep
        epoch_times.append(dt_ep)
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f"epoch {epoch+1:3d}/{args.epochs}  loss={avg:.5f}  {dt_ep:.1f}s  "
                  + "  ".join(f"{k}={float(v):.4f}" for k, v in logs.items()
                              if k != "loss"))
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            ckpt = os.path.join(args.out_dir, f"gsnet_{epoch+1}.pt")
            torch.save({"model": net.state_dict(), "cfg": vars(cfg)}, ckpt)
            torch.save({"model": net.state_dict(), "cfg": vars(cfg)},
                       os.path.join(args.out_dir, "gsnet_latest.pt"))
    total = time.time() - t_start
    with open(os.path.join(args.out_dir, "train_times.json"), "w") as f:
        json.dump({"total_seconds": total, "epochs": args.epochs,
                   "num_points": ds.n, "batch_size": args.batch_size,
                   "avg_epoch_seconds": sum(epoch_times) / max(1, len(epoch_times))},
                  f, indent=2)
    print(f"Training done in {total:.1f}s "
          f"(avg {sum(epoch_times)/max(1,len(epoch_times)):.1f}s/epoch).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_dir", required=True, help="dir of correspondence .npz")
    ap.add_argument("--out_dir", default="runs/gsnet")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--context_dim", type=int, default=256)
    ap.add_argument("--pos_offset_scale", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=50)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
