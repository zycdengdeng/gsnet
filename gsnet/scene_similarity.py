#
# Scene-distribution similarity diagnostic.
#
# Question: GS-Net is a *generalizable* model -- cross-scene transfer can only
# work if the train/test scenes share an input distribution. This script
# measures, in the EXACT feature space GS-Net sees (per-sequence normalized
# sparse points + local geometry + color), how similar the scenes are to each
# other, and -- crucially -- whether the held-out TEST scenes sit INSIDE or
# OUTSIDE the cloud of the TRAIN scenes. If the test scenes are distributional
# outliers, the negative cross-scene result is explained by the split, not by
# GS-Net being broken.
#
# Per-point feature (10-D):  [x_n, y_n, z_n, r, g, b, linearity, planarity,
#                             sphericity, log local-density]
# computed after the same median/p95 normalization used for training, then
# globally standardized so every dim contributes comparably.
#
# Pairwise scene distance = sliced-Wasserstein (full-distribution, no Gaussian
# assumption) with an MMD(rbf) cross-check. Outputs a heatmap, a 2-D MDS
# embedding (test scenes highlighted), the raw matrix, and a summary that
# reports the train/test gap explicitly.
#
# Usage:
#   python -m gsnet.scene_similarity \
#       --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam \
#       --test_scenes 10275144660749673822_5755_561 15868625208244306149_4340_000 \
#       --out_dir runs/scene_similarity
#
# To compare WITHIN-scene (front vs back) later, pass explicit dirs via
# --scenes <front_dir> <back_dir> ... (overrides auto-discovery).
#

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsnet.common import read_points_any, compute_normalization
from gsnet.waymo import discover_scenes, seg_name, sparse_points, resolve_scene

FEATURE_NAMES = ["x_n", "y_n", "z_n", "r", "g", "b",
                 "linearity", "planarity", "sphericity", "log_density"]


def file_features(path, k=10, max_pts=20000, seed=0):
    """Per-point 10-D features for ONE point file, in the GS-Net-normalized
    space. Each file is its own coherent SfM, so it is normalized and has its
    local geometry computed independently (correct for CARLA where each segment
    is a separate reconstruction)."""
    from scipy.spatial import cKDTree
    xyz, rgb = read_points_any(path)
    center, scale = compute_normalization(xyz)
    xyzn = (xyz - center) / scale  # same transform GS-Net trains/infers in

    tree = cKDTree(xyzn)
    rng = np.random.default_rng(seed)
    n = xyzn.shape[0]
    idx = (rng.choice(n, size=max_pts, replace=False) if n > max_pts
           else np.arange(n))
    q = xyzn[idx]
    # k+1 because the point itself is its own nearest neighbour
    dist, nbr = tree.query(q, k=min(k + 1, n))
    dist, nbr = dist[:, 1:], nbr[:, 1:]            # drop self
    log_density = np.log(dist.mean(axis=1) + 1e-9)  # mean kNN dist (normalized)

    # local covariance eigen-features
    nb = xyzn[nbr]                                  # (m, k, 3)
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", nb, nb) / nb.shape[1]
    ev = np.linalg.eigvalsh(cov)                    # ascending
    e3, e2, e1 = ev[:, 0], ev[:, 1], ev[:, 2]       # e1 >= e2 >= e3 >= 0
    e1c = np.clip(e1, 1e-12, None)
    linearity = (e1 - e2) / e1c
    planarity = (e2 - e3) / e1c
    sphericity = e3 / e1c

    feat = np.column_stack([
        q, rgb[idx], linearity, planarity, sphericity, log_density
    ]).astype(np.float32)
    return feat, int(n)


def group_features(paths, k=10, max_pts=20000, seed=0):
    """A 'scene/group' = one or more point files. Features are computed PER
    FILE (each independently normalized) then pooled; the pool is subsampled to
    max_pts so every group contributes equally."""
    feats, npts = [], 0
    for p in paths:
        f, n = file_features(p, k, max_pts, seed)
        feats.append(f)
        npts += n
    allf = np.vstack(feats)
    if allf.shape[0] > max_pts:
        rng = np.random.default_rng(seed)
        allf = allf[rng.choice(allf.shape[0], max_pts, replace=False)]
    return allf, dict(n_points=npts, n_files=len(paths), used=int(allf.shape[0]))


def sliced_wasserstein(A, B, n_proj=200, seed=0):
    """1-Wasserstein averaged over random 1-D projections (standardized space)."""
    rng = np.random.default_rng(seed)
    d = A.shape[1]
    P = rng.standard_normal((d, n_proj))
    P /= np.linalg.norm(P, axis=0, keepdims=True).clip(1e-9)
    pa = np.sort(A @ P, axis=0)
    pb = np.sort(B @ P, axis=0)
    m = min(pa.shape[0], pb.shape[0])
    # quantile-match by resampling both to m points
    qa = pa[np.linspace(0, pa.shape[0] - 1, m).astype(int)]
    qb = pb[np.linspace(0, pb.shape[0] - 1, m).astype(int)]
    return float(np.abs(qa - qb).mean())


def mmd_rbf(A, B, gamma=None, sub=4000, seed=0):
    """Unbiased-ish MMD^2 with RBF kernel; median-heuristic bandwidth."""
    rng = np.random.default_rng(seed)
    a = A[rng.choice(A.shape[0], min(sub, A.shape[0]), replace=False)]
    b = B[rng.choice(B.shape[0], min(sub, B.shape[0]), replace=False)]
    if gamma is None:
        c = np.vstack([a, b])
        s = rng.choice(c.shape[0], min(1000, c.shape[0]), replace=False)
        dd = np.linalg.norm(c[s][:, None] - c[s][None], axis=2)
        med = np.median(dd[dd > 0])
        gamma = 1.0 / (2 * med ** 2 + 1e-9)

    def k(x, y):
        d2 = (np.linalg.norm(x[:, None] - y[None], axis=2)) ** 2
        return np.exp(-gamma * d2)

    return float(k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean())


def classical_mds(D, dim=2):
    """Classical MDS -> dim-D coordinates from a distance matrix."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:dim]
    L = np.clip(w[order], 0, None)
    return V[:, order] * np.sqrt(L)


def build_groups(args):
    """Return list of (label, [point_file_paths]).

    Inputs COMBINE (so CARLA + Waymo can share one run / one standardization,
    which is required to compare their inter-scene spread on the same axis):
      --point_specs LABEL=GLOB ...  generic; pools all files matching GLOB per
                                    LABEL (CARLA: one scene = its 10 segment
                                    plys). QUOTE each entry so the shell does
                                    not expand the glob.
      --scenes DIR ...              explicit Waymo scene dirs (1 file each).
      --root DIR                    auto-discover Waymo segment-* dirs.
    """
    import glob as _glob
    groups = []
    for spec in args.point_specs:
        assert "=" in spec, f"--point_specs entry must be LABEL=GLOB, got {spec}"
        label, pat = spec.split("=", 1)
        paths = sorted(_glob.glob(pat))
        assert paths, f"no files match {pat} (label {label})"
        groups.append((label, paths))
    waymo = list(args.scenes) if args.scenes else (
        discover_scenes(args.root) if args.root else [])
    for s in waymo:
        groups.append((seg_name(s), [sparse_points(s)]))
    assert groups, "no scenes found (give --root, --scenes, or --point_specs)"
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="dir containing segment-* scenes")
    ap.add_argument("--scenes", nargs="+", default=[],
                    help="explicit scene dirs (overrides --root discovery)")
    ap.add_argument("--point_specs", nargs="+", default=[],
                    help="generic groups: LABEL=GLOB ... (pools files per label; "
                         "e.g. carla_s1=/mnt/.../sparse_point/S01/*_sparse.ply)")
    ap.add_argument("--test_scenes", nargs="+", default=[],
                    help="names/substrings of held-out test scenes to highlight")
    ap.add_argument("--out_dir", default="runs/scene_similarity")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max_pts", type=int, default=20000)
    ap.add_argument("--n_proj", type=int, default=200)
    args = ap.parse_args()

    groups = build_groups(args)
    names = [g[0] for g in groups]
    is_test = [any(t in nm for t in args.test_scenes) for nm in names]
    os.makedirs(args.out_dir, exist_ok=True)

    # --- per-group features ---
    feats, meta = [], {}
    for nm, paths in groups:
        f, info = group_features(paths, args.k, args.max_pts)
        feats.append(f)
        meta[nm] = info
        print(f"[feat] {nm[:28]:28s} files={info['n_files']:2d} "
              f"pts={info['n_points']:7d} used={info['used']:5d}", flush=True)

    # global standardization so every feature dim weighs comparably
    allf = np.vstack(feats)
    mu, sd = allf.mean(0), allf.std(0).clip(1e-6)
    feats = [(f - mu) / sd for f in feats]

    # --- pairwise distances ---
    N = len(groups)
    SW = np.zeros((N, N))
    MMD = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            SW[i, j] = SW[j, i] = sliced_wasserstein(feats[i], feats[j], args.n_proj)
            MMD[i, j] = MMD[j, i] = mmd_rbf(feats[i], feats[j])
        print(f"[dist] row {i+1}/{N} done", flush=True)

    # --- train/test gap analysis (on sliced-Wasserstein) ---
    tr = [i for i in range(N) if not is_test[i]]
    te = [i for i in range(N) if is_test[i]]

    def block_mean(rows, cols, exclude_diag=False):
        vals = [SW[i, j] for i in rows for j in cols
                if not (exclude_diag and i == j)]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"),) * 2

    tt_mean, tt_std = block_mean(tr, tr, exclude_diag=True)   # train<->train
    summary = {
        "scenes": names, "is_test": is_test,
        "train_train_meanstd": [tt_mean, tt_std],
        "per_test_scene": {},
        "sliced_wasserstein": SW.tolist(), "mmd_rbf": MMD.tolist(),
        "feature_names": FEATURE_NAMES, "meta": meta,
    }
    for i in te:
        d_to_train = [SW[i, j] for j in tr]
        # z-score of this test scene's mean distance-to-train vs the
        # train<->train baseline: >2 => clear distributional outlier
        z = ((np.mean(d_to_train) - tt_mean) / tt_std) if tt_std > 0 else float("nan")
        summary["per_test_scene"][names[i]] = {
            "mean_dist_to_train": float(np.mean(d_to_train)),
            "min_dist_to_train": float(np.min(d_to_train)),
            "nearest_train": names[tr[int(np.argmin(d_to_train))]],
            "z_vs_train_baseline": float(z),
        }

    with open(os.path.join(args.out_dir, "similarity.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        short = [f"{i}{'*' if is_test[i] else ''}" for i in range(N)]
        fig, ax = plt.subplots(figsize=(1 + 0.6 * N, 1 + 0.6 * N))
        im = ax.imshow(SW, cmap="viridis")
        ax.set_xticks(range(N)); ax.set_xticklabels(short)
        ax.set_yticks(range(N)); ax.set_yticklabels(short)
        for i in range(N):
            for j in range(N):
                ax.text(j, i, f"{SW[i,j]:.2f}", ha="center", va="center",
                        color="w", fontsize=7)
        ax.set_title("Sliced-Wasserstein scene distance (* = test)")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "heatmap.png"), dpi=130)
        plt.close(fig)

        coords = classical_mds(SW, 2)
        fig, ax = plt.subplots(figsize=(6, 5))
        for i in range(N):
            c = "crimson" if is_test[i] else "steelblue"
            ax.scatter(*coords[i], c=c, s=80,
                       edgecolors="k", zorder=3)
            ax.annotate(str(i), coords[i], fontsize=9,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_title("MDS embedding of scenes (red = held-out test)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "embedding.png"), dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] plotting skipped: {e}", flush=True)

    # --- markdown summary ---
    lines = ["# Scene-distribution similarity", "",
             "Index legend (* = held-out test scene):"]
    for i, nm in enumerate(names):
        lines.append(f"- {i}{'*' if is_test[i] else ''}: {nm}")
    lines += ["",
              f"**Train<->train distance**: mean={tt_mean:.3f} std={tt_std:.3f} "
              "(the natural spread among training scenes)", ""]
    if te:
        lines += ["**Are the test scenes inside the training distribution?**", "",
                  "| test scene | mean dist→train | nearest train | z vs train-baseline |",
                  "|---|---|---|---|"]
        for nm, d in summary["per_test_scene"].items():
            flag = " ⚠️OUTLIER" if d["z_vs_train_baseline"] > 2 else ""
            lines.append(f"| {nm[:26]} | {d['mean_dist_to_train']:.3f} | "
                         f"{d['nearest_train'][:18]} | "
                         f"{d['z_vs_train_baseline']:+.2f}{flag} |")
        lines += ["",
                  "_z>2 means the test scene's distance to the training set is far "
                  "beyond the training scenes' own spread → a distributional outlier "
                  "→ cross-scene generalization was unlikely to work for it._"]
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nOutputs -> {args.out_dir}/{{heatmap.png, embedding.png, "
          "similarity.json, summary.md}}")


if __name__ == "__main__":
    main()
