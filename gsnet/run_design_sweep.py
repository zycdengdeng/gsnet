#
# Network-design sweep (rebuttal R&D): compare GS-Net design choices
# (encoder x color-offset activation x loss weights) and rank them cheaply, then
# re-confirm the winner at full settings.
#
# Each config string: "encoder:color:wrot:wpos:wscale"
#   encoder in {concat, geom, geoedge, edgeconv, attention, mlp_only}
#   color   in {sigmoid, tanh}
#
# Training is fast (in-memory); to keep ranking cheap, SSE is run at reduced
# iterations on a sequence subset (--iterations, --eval_ids). Re-evaluate the
# winner at 30k on all 5 sequences afterwards.
#
# Usage:
#   python -m gsnet.run_design_sweep \
#       --corr_dir CORR/train --io_dir /mnt/zihanw/carla/input_output \
#       --sparse_root /mnt/zihanw/carla/sparse_point \
#       --out_dir runs/design --gpus 1 2 3 --iterations 7000 \
#       --eval_ids 110 310 510 \
#       --configs geom:sigmoid:1:1:1 geom:sigmoid:0.1:10:1 geom:tanh:0.1:10:1 \
#                 geoedge:sigmoid:0.1:10:1 geoedge:tanh:0.1:10:1 geoedge:tanh:0.1:10:10
#

import argparse
import concurrent.futures as cf
import json
import os
import queue
import subprocess
import sys

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n[gpu{gpu}] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def one(cfg, args, gpu):
    """Train + cheap SSE eval for one config, entirely on one GPU."""
    enc, color, wr, wp, ws = cfg.split(":")
    label = f"{enc}_{color}_wr{wr}_wp{wp}_ws{ws}"
    model = os.path.join(args.out_dir, "model", label)
    sse = os.path.join(args.out_dir, "sse", label)
    if not os.path.exists(os.path.join(model, "gsnet_latest.pt")):
        run([PY, "-m", "gsnet.train_gsnet", "--corr_dir", args.corr_dir,
             "--out_dir", model, "--encoder_type", enc, "--color_activation", color,
             "--epochs", str(args.epochs), "--in_memory", "1",
             "--w_rot", wr, "--w_pos", wp, "--w_scale", ws], gpu)
    run([PY, "-m", "gsnet.run_sse", "--io_dir", args.io_dir,
         "--sparse_root", args.sparse_root,
         "--ckpt", os.path.join(model, "gsnet_latest.pt"),
         "--out_dir", sse, "--skip_baseline", "--iterations", str(args.iterations),
         "--test_ids", *args.eval_ids, "--gpus", str(gpu)], gpu)
    g = json.load(open(os.path.join(sse, "sse_results.json")))["averages"]["gsnet"]
    return {"label": label, "config": cfg, **{k: g[k] for k in ("PSNR", "SSIM", "LPIPS")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_dir", default="CORR/train")
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--sparse_root", required=True)
    ap.add_argument("--out_dir", default="runs/design")
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=7000,
                    help="reduced 3DGS iterations for cheap ranking")
    ap.add_argument("--eval_ids", nargs="+", default=["110", "310", "510"])
    ap.add_argument("--configs", nargs="+", default=[
        "geom:sigmoid:1:1:1", "geom:sigmoid:0.1:10:1", "geom:tanh:0.1:10:1",
        "geoedge:sigmoid:0.1:10:1", "geoedge:tanh:0.1:10:1", "geoedge:tanh:0.1:10:10"])
    args = ap.parse_args()

    # Run configs in parallel, one per GPU (each config is fully self-contained
    # on its GPU: fast in-memory training + reduced-iter SSE on the subset).
    rows = []
    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)
    import threading
    lock = threading.Lock()
    os.makedirs(args.out_dir, exist_ok=True)

    def worker(cfg):
        gpu = gpu_q.get()
        try:
            r = one(cfg, args, gpu)
            with lock:
                rows.append(r)
                rows.sort(key=lambda x: -x["PSNR"])
                json.dump(rows, open(os.path.join(args.out_dir, "design.json"), "w"), indent=2)
            print(f"[done] {r['config']} -> PSNR={r['PSNR']:.2f}", flush=True)
        finally:
            gpu_q.put(gpu)

    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        for f in cf.as_completed([ex.submit(worker, c) for c in args.configs]):
            f.result()

    lines = ["", f"## Design sweep (rank @ {args.iterations} iters, seqs {args.eval_ids})",
             "| Config (enc:color:wr:wp:ws) | PSNR | SSIM | LPIPS |",
             "|------------------------------|------|------|-------|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['PSNR']:.2f} | {r['SSIM']:.3f} | {r['LPIPS']:.3f} |")
    table = "\n".join(lines)
    open(os.path.join(args.out_dir, "design.md"), "w").write(table + "\n")
    print(table)
    print(f"\nbest: {rows[0]['config']}  PSNR={rows[0]['PSNR']:.2f}  "
          f"(re-confirm at 30k on all 5 seqs)")


if __name__ == "__main__":
    main()
