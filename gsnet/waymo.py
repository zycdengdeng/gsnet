#
# Path helpers for the Waymo real-data experiment, robust to two COLMAP layouts:
#   (A) dense layout : <scene>/colmap/dense/{fused.ply, images/, sparse/0/}
#   (B) sparse layout: <scene>/colmap/sparse/0/  + <scene>/images/   (no dense/)
#
# `scene_source(scene)` returns a 3DGS-ready source directory (containing
# sparse/0 and images) by symlinking whatever is found, so train.py/run_sse work
# unchanged for either layout.
#

import glob
import os
import shutil


def discover_scenes(root):
    return sorted(glob.glob(os.path.join(root, "segment-*")))


def seg_name(scene_path):
    return os.path.basename(scene_path.rstrip("/"))


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def sparse_model_dir(scene_path):
    """The COLMAP sparse model dir (containing cameras/images/points3D.bin).
    Prefers an undistorted (PINHOLE) dense model when present."""
    return _first_existing([
        os.path.join(scene_path, "colmap", "dense", "sparse", "0"),
        os.path.join(scene_path, "colmap", "dense", "sparse"),
        os.path.join(scene_path, "colmap", "sparse", "0"),
        os.path.join(scene_path, "colmap", "sparse"),
        os.path.join(scene_path, "sparse", "0"),
    ])


def images_dir(scene_path):
    return _first_existing([
        os.path.join(scene_path, "colmap", "dense", "images"),
        os.path.join(scene_path, "images"),
        os.path.join(scene_path, "colmap", "images"),
    ])


def _link(link_path, target):
    target = os.path.abspath(target)
    try:
        if os.path.islink(link_path):
            if os.path.realpath(link_path) == os.path.realpath(target):
                return
            os.remove(link_path)
        os.symlink(target, link_path)
    except FileExistsError:
        pass  # created concurrently by a sibling worker


def scene_source(scene_path, tag=""):
    """Return a 3DGS-ready source dir (sparse/0 + images). The sparse model is
    COPIED (not symlinked) into a per-`tag` dir so each experiment has its OWN
    test.txt -- concurrent runs with DIFFERENT splits (frame-holdout vs
    camera-holdout) must NOT share/overwrite test.txt. Images are symlinked
    (read-only, safe to share). Pass a unique `tag` per experiment (e.g. the
    out_dir basename)."""
    sp0 = sparse_model_dir(scene_path)
    img = images_dir(scene_path)
    assert sp0 and img, f"missing sparse/images for {scene_path}"
    name = "_gsnet_src" + (f"_{tag}" if tag else "")
    src = os.path.join(scene_path, "colmap", name)
    dst0 = os.path.join(src, "sparse", "0")
    os.makedirs(dst0, exist_ok=True)
    for fn in os.listdir(sp0):
        s, d = os.path.join(sp0, fn), os.path.join(dst0, fn)
        if os.path.isfile(s) and not os.path.exists(d):
            shutil.copy(s, d)
    _link(os.path.join(src, "images"), img)
    return src


# Backwards-compatible alias used by drivers.
def dense_dir(scene_path):
    return scene_source(scene_path)


def fused_ply(scene_path):
    return os.path.join(scene_path, "colmap", "dense", "fused.ply")


def sparse_points(scene_path):
    """Sparse SfM points (GS-Net input)."""
    return os.path.join(sparse_model_dir(scene_path), "points3D.bin")


def gdense_path(gdense_dir, scene_path, iterations=30000):
    return os.path.join(gdense_dir, seg_name(scene_path), "point_cloud",
                        f"iteration_{iterations}", "point_cloud.ply")


def resolve_scene(root, name_or_path):
    if os.path.isdir(name_or_path):
        return name_or_path
    return os.path.join(root, name_or_path)
