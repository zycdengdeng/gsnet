#
# CARLA attribute-ablation study -- the CARLA counterpart of run_waymo_ablation,
# so the two can be CONTRASTED: a factor that helps on CARLA (sim) but hurts on
# Waymo (real) is a sim-vs-real non-transferable factor (the opacity hypothesis).
#
# Same variants (disable a head -> fixed default + drop from loss), trained on
# CORR/train and SSE-evaluated on the CARLA test sequences (310 excluded by
# default: its GS-Net optimization degenerates on all seeds, see PROJECT_STATE).
#
# Usage:
#   python -m gsnet.run_carla_ablation \
#       --corr_dir CORR/train \
#       --io_dir /mnt/zihanw/carla/input_output \
#       --sparse_root /mnt/zihanw/carla/sparse_point \
#       --out_dir runs/carla_ablation --gpus 0 1 2 3 4
#

import argparse
import json
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VARIANTS = {
    "full":         [],
    "no_opacity":   ["--no_opacity"],
    "no_color":     ["--no_color"],
    "no_scale_rot": ["--no_scale_rot"],
    "xyz_rgb":      ["--no_opacity", "--no_scale_rot"],
    "dens_only":    ["--no_color", "--no_opacity", "--no_scale_rot"],
}


def sh(cmd, gpus):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def train_variant(name, flags, args):
    model = os.path.join(args.out_dir, name, "model")
    ckpt = os.path.join(model, "gsnet_latest.pt")
    if os.path.exists(ckpt):
        print(f"[skip train] {ckpt}")
        return ckpt
    sh([PY, "-m", "gsnet.train_gsnet", "--corr_dir", args.corr_dir, "--out_dir", model,
        "--encoder_type", "geom", "--color_activation", "tanh",
        "--w_rot", "0.1", "--w_pos", "10", "--T", str(args.T), "--M", str(args.M),
        "--in_memory", "1", "--epochs", str(args.epochs), *flags], args.gpus[:1])
    return ckpt


def sse(name, ckpt, args, skip_baseline):
    out = os.path.join(args.out_dir, name, "sse")
    cmd = [PY, "-m", "gsnet.run_sse", "--io_dir", args.io_dir,
           "--sparse_root", args.sparse_root, "--ckpt", ckpt, "--out_dir", out,
           "--test_ids", *args.test_ids, "--iterations", str(args.iterations),
           "--gpus", *[str(g) for g in args.gpus]]
    if skip_baseline:
        cmd.append("--skip_baseline")
    sh(cmd, args.gpus)
    return json.load(open(os.path.join(out, "sse_results.json")))["averages"]


def write_table(rows, baseline, out_dir):
    lines = ["", "## CARLA attribute ablation (SSE, geom:tanh:0.1:10, M=3, 310 excluded)",
             f"baseline (sparse-SfM init) = {baseline:.2f}" if baseline else "baseline = (pending)",
             "", "| variant | gsnet PSNR | Δ vs baseline |", "|---|---|---|"]
    for name, psnr in rows.items():
        d = f"{psnr - baseline:+.2f}" if baseline else "-"
        lines.append(f"| {name} | {psnr:.2f} | {d} |")
    json.dump({"baseline": baseline, "gsnet": rows},
              open(os.path.join(out_dir, "ablation_results.json"), "w"), indent=2)
    open(os.path.join(out_dir, "ablation_results.md"), "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr_dir", default="CORR/train")
    ap.add_argument("--io_dir", required=True)
    ap.add_argument("--sparse_root", required=True)
    ap.add_argument("--out_dir", default="runs/carla_ablation")
    ap.add_argument("--test_ids", nargs="+", default=["110", "210", "410", "510"],
                    help="default excludes 310 (degenerate on all seeds)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=30000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Resume: preload completed variants so re-running merges instead of clobbering.
    rows, baseline = {}, None
    for name in args.variants:
        rp = os.path.join(args.out_dir, name, "sse", "sse_results.json")
        if os.path.exists(rp):
            d = json.load(open(rp)).get("averages", {})
            if "gsnet" in d:
                rows[name] = d["gsnet"]["PSNR"]
            if baseline is None and "baseline" in d:
                baseline = d["baseline"]["PSNR"]
    if rows:
        print(f"[resume] reusing completed variants: {list(rows)}")

    for name in args.variants:
        assert name in VARIANTS, f"unknown variant {name}"
        if name in rows:
            continue
        ckpt = train_variant(name, VARIANTS[name], args)
        d = sse(name, ckpt, args, skip_baseline=(baseline is not None))
        if baseline is None and "baseline" in d:
            baseline = d["baseline"]["PSNR"]
        rows[name] = d["gsnet"]["PSNR"]
        write_table(rows, baseline, args.out_dir)
    write_table(rows, baseline, args.out_dir)
    print("\n" + open(os.path.join(args.out_dir, "ablation_results.md")).read())


if __name__ == "__main__":
    main()
