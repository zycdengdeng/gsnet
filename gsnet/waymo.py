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
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def build_subset_source(scene_path, keep_cams, tag):
    """Build an isolated COLMAP text model containing ONLY images from `keep_cams`
    (source ∪ target cameras); all other cameras are DROPPED. Used for
    cross-sensor 'reconstruct from a few cameras -> synthesize another', e.g.
    keep_cams=['cam1','cam2','cam0'] for FL+FR -> FRONT. Cameras normalized to
    PINHOLE; all points3D copied as init; images symlinked. Returns source dir."""
    from scene.colmap_loader import (read_extrinsics_binary, read_extrinsics_text,
                                     read_intrinsics_binary, read_intrinsics_text)
    sp0 = sparse_model_dir(scene_path)
    img = images_dir(scene_path)
    try:
        extr = read_extrinsics_binary(os.path.join(sp0, "images.bin"))
        intr = read_intrinsics_binary(os.path.join(sp0, "cameras.bin"))
    except Exception:
        extr = read_extrinsics_text(os.path.join(sp0, "images.txt"))
        intr = read_intrinsics_text(os.path.join(sp0, "cameras.txt"))
    cam0 = list(intr.values())[0]

    src = os.path.join(scene_path, "colmap", "_gsnet_src" + (f"_{tag}" if tag else ""))
    dst0 = os.path.join(src, "sparse", "0")
    os.makedirs(dst0, exist_ok=True)
    # cameras.txt: ALL cameras as PINHOLE, preserving per-camera intrinsics
    # (Waymo's 5 cameras have DIFFERENT intrinsics -- must NOT collapse to one).
    with open(os.path.join(dst0, "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        for cid, c in sorted(intr.items()):
            p = list(c.params)
            fx, fy, cx, cy = ((p[0], p[1], p[2], p[3]) if c.model == "PINHOLE"
                              else (p[0], p[0], p[1], p[2]))
            f.write(f"{cid} PINHOLE {c.width} {c.height} {fx} {fy} {cx} {cy}\n")
    keep = set(keep_cams)
    lines, iid = ["# Image list"], 0
    for k in sorted(extr, key=lambda x: extr[x].name):
        im = extr[k]
        if os.path.dirname(im.name) in keep:
            iid += 1
            q, t = im.qvec, im.tvec
            lines.append(f"{iid} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} "
                         f"{im.camera_id} {im.name}")
            lines.append("")
    with open(os.path.join(dst0, "images.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for cand in ("points3D.bin", "points3D.txt"):
        s = os.path.join(sp0, cand)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(dst0, cand))
            break
    _link(os.path.join(src, "images"), img)
    print(f"[subset] {src}: kept cameras {sorted(keep)} ({iid} images)")
    return src


def resolve_scene(root, name_or_path):
    if os.path.isdir(name_or_path):
        return name_or_path
    direct = os.path.join(root, name_or_path)
    if os.path.isdir(direct):
        return direct
    # Fall back to a substring match against discovered scenes, so short ids
    # (e.g. "10275144660749673822_5755_561") resolve to the full segment dir
    # ("segment-10275..._5775_561_with_camera_labels").
    matches = [s for s in discover_scenes(root) if name_or_path in seg_name(s)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"ambiguous scene '{name_or_path}' matches "
                         f"{[seg_name(m) for m in matches]}")
    return direct  # let it fail downstream with a clear missing-path message
