#
# Path helpers for the Waymo real-data experiment (reviewer-requested validation
# on a real autonomous-driving dataset).
#
# Per-scene COLMAP layout (front 3 cameras x 20 frames = 60 images):
#   <scene>/colmap/dense/
#       fused.ply                      # MVS dense point cloud (3DGS init for G_dense)
#       images/cam{0,1,2}/0NN.jpg       # undistorted images
#       sparse/0/{cameras,images,points3D}.bin   # sparse SfM (poses + sparse points)
#
# The dense dir is a self-contained COLMAP scene that the 3DGS pipeline reads
# directly (it has sparse/0 + images/).
#

import glob
import os


def discover_scenes(root):
    return sorted(glob.glob(os.path.join(root, "segment-*")))


def seg_name(scene_path):
    return os.path.basename(scene_path.rstrip("/"))


def dense_dir(scene_path):
    return os.path.join(scene_path, "colmap", "dense")


def fused_ply(scene_path):
    return os.path.join(dense_dir(scene_path), "fused.ply")


def resolve_scene(root, name_or_path):
    """Accept either a full path or a segment name under root."""
    if os.path.isdir(name_or_path):
        return name_or_path
    return os.path.join(root, name_or_path)
