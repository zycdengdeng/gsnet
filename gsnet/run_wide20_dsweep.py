#
# Densification sweep on the Waymo wide20 test scenes -- to establish a
# CREDIBLE baseline (the default 0.0002 over-densifies these ~30-image
# wide-baseline scenes -> baseline craters to ~18; gentler/no densification
# recovers the expected ~21). For each densification setting we run BOTH
# baseline and GS-Net so the comparison is at the baseline's strongest point.
#
# Reuses the already-trained wide20 ckpt; quick (2 scenes, existing model).
#
# Usage:
#   nohup python -m gsnet.run_wide20_dsweep \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam_wide20 \
#       --ckpt runs/wide20/model/gsnet_latest.pt \
#       --gpus 4 5 6 7 > runs/wide20_dsweep.log 2>&1 &
#
import argparse
import json
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_SCENES = ["10275144660749673822_5755_561", "15868625208244306149_4340_000"]

# (tag, train_extra) -- densification from most aggressive (default) to off.
CONFIGS = [
    ("default_g2e4", ""),                              # 0.0002 (default, over-densifies)
    ("g4e4",  "--densify_grad_threshold 0.0004"),
    ("g8e4",  "--densify_grad_threshold 0.0008"),
    ("g16e4", "--densify_grad_threshold 0.0016"),
    ("until7k", "--densify_until_iter 7000"),
    ("off",   "--densify_until_iter 0"),
]


def sh(cmd, gpus):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def done(out_dir):
    rp = os.path.join(out_dir, "sse_results.json")
    if not os.path.exists(rp):
        return False
    per = json.load(open(rp)).get("per_scene", [])
    return len(per) >= len(TEST_SCENES) and all(
        "baseline" in r and "gsnet" in r for r in per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", default="runs/wide20/model/gsnet_latest.pt")
    ap.add_argument("--out_root", default="runs/wide20_dsweep")
    ap.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--n_holdout", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=30000)
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    for tag, extra in CONFIGS:
        out_dir = os.path.join(args.out_root, tag)
        if done(out_dir):
            print(f"[skip] {tag} (done)")
            continue
        cmd = [PY, "-m", "gsnet.waymo_sse", "--root", args.root,
               "--test_scenes", *TEST_SCENES, "--ckpt", args.ckpt,
               "--out_dir", out_dir, "--n_holdout", str(args.n_holdout),
               "--iterations", str(args.iterations),
               "--gpus", *[str(g) for g in args.gpus]]
        if extra:
            cmd += ["--train_extra", extra]
        sh(cmd, args.gpus)

    # aggregate
    rows = []
    for tag, extra in CONFIGS:
        rp = os.path.join(args.out_root, tag, "sse_results.json")
        if not os.path.exists(rp):
            continue
        a = json.load(open(rp)).get("averages", {})
        b = a.get("baseline", {}).get("PSNR", float("nan"))
        g = a.get("gsnet", {}).get("PSNR", float("nan"))
        rows.append((tag, extra or "(default 0.0002)", b, g, g - b))
    lines = ["", "# Waymo wide20 densification sweep (2 scenes, existing ckpt)",
             "", "| config | densify setting | baseline | gsnet | Delta |",
             "|---|---|---|---|---|"]
    for tag, ex, b, g, d in rows:
        lines.append(f"| {tag} | `{ex}` | {b:.2f} | {g:.2f} | {d:+.2f} |")
    if rows:
        best = max(rows, key=lambda r: r[2])  # strongest baseline
        lines += ["", f"**Strongest baseline = `{best[0]}` "
                  f"(baseline {best[2]:.2f}, gsnet {best[3]:.2f}, "
                  f"Delta {best[4]:+.2f})** -- compare GS-Net here (credible regime)."]
    table = "\n".join(lines)
    with open(os.path.join(args.out_root, "dsweep.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\n-> {args.out_root}/dsweep.md")


if __name__ == "__main__":
    main()
