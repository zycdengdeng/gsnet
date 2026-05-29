#
# I/O helpers: write GS-Net predicted Gaussians as a standard 3DGS .ply so the
# existing scene/gaussian_model.py:load_ply consumes them unchanged.
#
# Attribute layout matches GaussianModel.construct_list_of_attributes:
#   x,y,z, nx,ny,nz, f_dc_0..2, f_rest_*, opacity, scale_0..2, rot_0..3
# with stored conventions:
#   opacity -> inverse_sigmoid(alpha)   (sigmoid activation in 3DGS)
#   scale   -> log(scale)               (exp activation in 3DGS)
#   f_dc    -> RGB2SH(rgb)              (0-order SH); f_rest zero-initialized
#   rot     -> (w, x, y, z) quaternion
#

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def _rgb_to_sh(rgb):
    return (rgb - 0.5) / C0


def _inverse_sigmoid(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def save_gaussians_ply(
    path,
    mu,            # (P, 3) absolute position
    rgb,           # (P, 3) diffuse color in [0, 1]
    scale,         # (P, 3) actual scale (will be log-stored)
    quat,          # (P, 4) (w, x, y, z), assumed normalized
    opacity,       # (P, 1) or (P,) alpha; primitives with alpha <= thresh dropped
    sh_degree=3,
    opacity_thresh=0.0,
):
    mu = np.asarray(mu, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    scale = np.asarray(scale, dtype=np.float32).reshape(-1, 3)
    quat = np.asarray(quat, dtype=np.float32).reshape(-1, 4)
    opacity = np.asarray(opacity, dtype=np.float32).reshape(-1)

    # Discard non-positive opacity primitives (Tanh opacity < 0 => not rendered).
    keep = opacity > opacity_thresh
    mu, rgb, scale, quat, opacity = (
        mu[keep], rgb[keep], scale[keep], quat[keep], opacity[keep]
    )
    P = mu.shape[0]

    normals = np.zeros((P, 3), dtype=np.float32)
    f_dc = _rgb_to_sh(rgb).astype(np.float32)                        # (P, 3)
    n_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    f_rest = np.zeros((P, n_rest), dtype=np.float32)
    op = _inverse_sigmoid(opacity).reshape(P, 1).astype(np.float32)
    log_scale = np.log(np.clip(scale, 1e-8, None)).astype(np.float32)

    attr_names = ["x", "y", "z", "nx", "ny", "nz"]
    attr_names += [f"f_dc_{i}" for i in range(3)]
    attr_names += [f"f_rest_{i}" for i in range(n_rest)]
    attr_names += ["opacity"]
    attr_names += [f"scale_{i}" for i in range(3)]
    attr_names += [f"rot_{i}" for i in range(4)]

    attributes = np.concatenate(
        [mu, normals, f_dc, f_rest, op, log_scale, quat], axis=1
    )
    dtype_full = [(name, "f4") for name in attr_names]
    elements = np.empty(P, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
    return P
