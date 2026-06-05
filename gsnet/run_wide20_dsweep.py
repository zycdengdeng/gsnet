#
# DECISIVE Waymo diagnostic: sweep densification for ALL THREE inits
# (SfM / MVS / GS-Net) on the wide20 test scenes, so we compare each at the
# baseline's STRONGEST (credible) operating point -- not the default 0.0002
# that over-densifies these ~30-image wide-baseline scenes (baseline -> ~18).
#
# Question it settles:
#   * best-MVS  >> best-SfM  -> headroom exists, GS-Net underperforms it
#                               => PREDICTION-QUALITY problem (fixable: more
#                                  wide-baseline training scenes / better G_dense).
#   * best-MVS  ~= best-SfM  -> NO init can help even at the credible baseline
#                               => Waymo PSNR is closed; pivot the contribution.
#
# Reuses the trained wide20 ckpt. Each densify config runs SfM/MVS/GS-Net via
# run_waymo_initcmp (resumable). Aggregates a per-config table.
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

# (tag, train_extra) -- default (over-densifies) -> gentler -> off.
CONFIGS = [
    ("default_g2e4", ""),                       # 0.0002 default
    ("g4e4",  "--densify_grad_threshold 0.0004"),
    ("g8e4",  "--densify_grad_threshold 0.0008"),
    ("off",   "--densify_until_iter 0"),
]


def sh(cmd, gpus):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, env=env)


def cfg_done(out_dir):
    rp = os.path.join(out_dir, "initcmp.json")
    if not os.path.exists(rp):
        return False
    rows = json.load(open(rp))
    return (len(rows) >= len(TEST_SCENES)
            and all(all(k in r for k in ("sfm", "mvs", "gsnet")) for r in rows.values()))


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
        if cfg_done(out_dir):
            print(f"[skip] {tag} (done)")
            continue
        cmd = [PY, "-m", "gsnet.run_waymo_initcmp", "--root", args.root,
               "--test_scenes", *TEST_SCENES, "--ckpt", args.ckpt,
               "--out_dir", out_dir, "--n_holdout", str(args.n_holdout),
               "--iterations", str(args.iterations),
               "--gpus", *[str(g) for g in args.gpus]]
        if extra:
            cmd += ["--train_extra", extra]
        sh(cmd, args.gpus)

    # aggregate: per-config avg of sfm/mvs/gsnet across scenes
    def avg(rows, k):
        v = [r[k] for r in rows.values() if k in r]
        return sum(v) / len(v) if v else float("nan")

    table = []
    for tag, extra in CONFIGS:
        rp = os.path.join(args.out_root, tag, "initcmp.json")
        if not os.path.exists(rp):
            continue
        rows = json.load(open(rp))
        s, m, g = avg(rows, "sfm"), avg(rows, "mvs"), avg(rows, "gsnet")
        table.append((tag, extra or "(default 0.0002)", s, m, g))

    lines = ["", "# Waymo wide20 densification sweep x {SfM, MVS, GS-Net} (2 scenes)",
             "", "| config | densify | SfM | MVS | GS-Net | MVS-SfM | GSNet-SfM |",
             "|---|---|---|---|---|---|---|"]
    for tag, ex, s, m, g in table:
        lines.append(f"| {tag} | `{ex}` | {s:.2f} | {m:.2f} | {g:.2f} "
                     f"| {m-s:+.2f} | {g-s:+.2f} |")
    if table:
        best_sfm = max(table, key=lambda r: r[2])
        best_mvs = max(table, key=lambda r: r[3])
        best_gs = max(table, key=lambda r: r[4])
        lines += ["",
                  f"**Best SfM baseline** = {best_sfm[2]:.2f} @ `{best_sfm[0]}` "
                  f"(the credible number to beat).",
                  f"**Best MVS (ceiling)** = {best_mvs[3]:.2f} @ `{best_mvs[0]}` "
                  f"-> headroom over best SfM = **{best_mvs[3]-best_sfm[2]:+.2f}**.",
                  f"**Best GS-Net** = {best_gs[4]:.2f} @ `{best_gs[0]}` "
                  f"-> vs best SfM = **{best_gs[4]-best_sfm[2]:+.2f}**.",
                  "",
                  "Read: if (Best MVS - Best SfM) is large but GS-Net lags -> "
                  "prediction-quality problem (fixable). If it's ~0 -> no init "
                  "headroom at a credible baseline (Waymo PSNR closed)."]
    out = "\n".join(lines)
    with open(os.path.join(args.out_root, "dsweep.md"), "w") as f:
        f.write(out + "\n")
    print("\n" + out)
    print(f"\n-> {args.out_root}/dsweep.md")


if __name__ == "__main__":
    main()
