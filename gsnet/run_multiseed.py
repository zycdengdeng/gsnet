#
# Multi-seed variance study: the baseline 3DGS is deterministic (fixed seed in
# safe_state), but GS-Net training is stochastic, so its downstream PSNR varies
# run-to-run. This trains one config with several seeds and evaluates SSE and/or
# CSE for each, reporting mean +/- std -- needed to make honest claims about the
# gain over the (deterministic) baseline.
#
# Seeds are run in parallel, one per GPU (train + eval on that GPU).
#
# Usage:
#   python -m gsnet.run_multiseed \
#       --config geom:tanh:0.1:10:1 --M 3 --corr_dir CORR/train \
#       --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point \
#       --scenes_dir runs/cse_scenes --eval sse cse \
#       --seeds 0 1 2 3 4 --gpus 0 1 2 3 4 5 6 7 --out_dir runs/multiseed
#

import argparse
import concurrent.futures as cf
import json
import os
import queue
import statistics
import subprocess
import sys
import threading

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"\n[gpu{gpu}] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def avg_psnr(results_json, key):
    d = json.load(open(results_json))
    return d["averages"][key]


def one_seed(seed, args, gpu):
    enc, color, wr, wp, ws = args.config.split(":")
    model = os.path.join(args.out_dir, f"model_s{seed}")
    if not os.path.exists(os.path.join(model, "gsnet_latest.pt")):
        run([PY, "-m", "gsnet.train_gsnet", "--corr_dir", args.corr_dir,
             "--out_dir", model, "--encoder_type", enc, "--color_activation", color,
             "--epochs", str(args.epochs), "--in_memory", "1", "--M", str(args.M),
             "--w_rot", wr, "--w_pos", wp, "--w_scale", ws, "--seed", str(seed)], gpu)
    ckpt = os.path.join(model, "gsnet_latest.pt")
    out = {"seed": seed}
    if "sse" in args.eval:
        sse = os.path.join(args.out_dir, f"sse_s{seed}")
        run([PY, "-m", "gsnet.run_sse", "--io_dir", args.io_dir,
             "--sparse_root", args.sparse_root, "--ckpt", ckpt, "--out_dir", sse,
             "--skip_baseline", "--iterations", "30000",
             "--test_ids", *args.eval_ids, "--gpus", str(gpu)], gpu)
        out["sse"] = avg_psnr(os.path.join(sse, "sse_results.json"), "gsnet")
    if "cse" in args.eval:
        cse = os.path.join(args.out_dir, f"cse_s{seed}")
        run([PY, "-m", "gsnet.run_cse", "--scenes_dir", args.scenes_dir,
             "--sparse_root", args.sparse_root, "--ckpt", ckpt, "--out_dir", cse,
             "--skip_baseline", "--ids", *args.eval_ids, "--gpus", str(gpu)], gpu)
        out["cse"] = avg_psnr(os.path.join(cse, "cse_results.json"), "gsnet")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="geom:tanh:0.1:10:1")
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--corr_dir", default="CORR/train")
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--sparse_root", required=True)
    ap.add_argument("--scenes_dir", default="runs/cse_scenes")
    ap.add_argument("--eval", nargs="+", default=["sse"], choices=["sse", "cse"])
    ap.add_argument("--eval_ids", nargs="+", default=["110", "210", "310", "410", "510"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out_dir", default="runs/multiseed")
    args = ap.parse_args()

    rows = []
    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)
    lock = threading.Lock()
    os.makedirs(args.out_dir, exist_ok=True)

    def worker(seed):
        gpu = gpu_q.get()
        try:
            r = one_seed(seed, args, gpu)
            with lock:
                rows.append(r)
                json.dump(rows, open(os.path.join(args.out_dir, "multiseed.json"), "w"), indent=2)
            print(f"[done] seed {seed}: " + " ".join(
                f"{k}={r[k]['PSNR']:.2f}" for k in ("sse", "cse") if k in r), flush=True)
        finally:
            gpu_q.put(gpu)

    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        for f in cf.as_completed([ex.submit(worker, s) for s in args.seeds]):
            f.result()

    lines = [f"\n## Multi-seed ({args.config}, M={args.M}, {len(rows)} seeds)"]
    for ev in args.eval:
        vals = [r[ev]["PSNR"] for r in rows if ev in r]
        if vals:
            m = statistics.mean(vals)
            s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            lines.append(f"- **{ev.upper()} PSNR**: {m:.2f} ± {s:.2f}  "
                         f"(seeds: {[round(v,2) for v in vals]})")
    table = "\n".join(lines)
    open(os.path.join(args.out_dir, "multiseed.md"), "w").write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
