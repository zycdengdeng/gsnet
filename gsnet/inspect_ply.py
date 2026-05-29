#
# Diagnostic helper: inspect a .ply's vertex fields, coordinate ranges, and
# (optionally) the alignment between a sparse cloud and G_dense. Run this on the
# server and paste the output so we can confirm color fields and calibrate the
# coordinate scale.
#
# Usage:
#   python -m gsnet.inspect_ply --ply sparse_point/S01/101_sparse.ply
#   python -m gsnet.inspect_ply \
#       --ply sparse_point/S01/101_sparse.ply \
#       --gdense input_output/output_101_dense/point_cloud/iteration_30000/point_cloud.ply
#

import argparse

import numpy as np
from plyfile import PlyData


def describe(path):
    ply = PlyData.read(path)
    v = ply.elements[0]
    print(f"\n=== {path} ===")
    print(f"  vertices: {len(v.data)}")
    print("  fields:", [(p.name, str(p.val_dtype)) for p in v.properties])
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    center = np.median(xyz, axis=0)
    d = np.linalg.norm(xyz - center, axis=1)
    print(f"  xyz min: {xyz.min(0)}")
    print(f"  xyz max: {xyz.max(0)}")
    print(f"  extent (max-min): {xyz.max(0) - xyz.min(0)}")
    print(f"  dist-to-median: mean={d.mean():.3f} p50={np.percentile(d,50):.3f} "
          f"p95={np.percentile(d,95):.3f} p99={np.percentile(d,99):.3f} max={d.max():.3f}")
    # Print a couple of sample rows for the first few fields.
    sample_fields = [p.name for p in v.properties][:9]
    print("  sample row 0:", {f: float(v[f][0]) for f in sample_fields})
    return xyz


def alignment(sparse_xyz, gdense_path):
    from scipy.spatial import cKDTree
    g = PlyData.read(gdense_path).elements[0]
    gxyz = np.stack([g["x"], g["y"], g["z"]], axis=1).astype(np.float64)
    tree = cKDTree(gxyz)
    dist, _ = tree.query(sparse_xyz, k=1)
    print(f"\n=== alignment sparse -> G_dense nearest-neighbor distance ===")
    print(f"  mean={dist.mean():.4f} p50={np.percentile(dist,50):.4f} "
          f"p90={np.percentile(dist,90):.4f} p99={np.percentile(dist,99):.4f} "
          f"max={dist.max():.4f}")
    print("  (small relative to the scene extent => sparse & G_dense are aligned)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--gdense", default="")
    args = ap.parse_args()
    sparse_xyz = describe(args.ply)
    if args.gdense:
        describe(args.gdense)
        alignment(sparse_xyz, args.gdense)


if __name__ == "__main__":
    main()
