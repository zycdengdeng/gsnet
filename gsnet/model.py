#
# GS-Net model: geometry-aware feature encoding + multi-head Gaussian expansion.
#
# Faithful to the paper:
#   (a) Geometry-Aware Feature Encoding (Sec. IV-A / Eq. 3)
#   (b) Multi-Head Gaussian Primitive Expansion (Sec. IV-B / Eq. 4)
#
# Notation follows the paper. Network-predicted quantities carry a hat (`_hat`),
# `delta_*` denote offsets relative to the input point attributes.
#

from dataclasses import dataclass, field

import torch
import torch.nn as nn


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """Device-agnostic, batched unit-quaternion -> rotation matrix.

    Args:
        q: (..., 4) quaternion in (w, x, y, z) order. Need not be normalized.
    Returns:
        (..., 3, 3) rotation matrix. Differentiable.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    R = torch.stack(
        [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1
    ).reshape(q.shape[:-1] + (3, 3))
    return R


def _mlp(dims, act=nn.ReLU, last_act=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


@dataclass
class GSNetConfig:
    # Input point feature: position (3) + color (3) = 6-d raw feature.
    in_dim: int = 6
    # Latent per-point embedding dim d (f_n in Eq. 3).
    embed_dim: int = 128
    # Context-aware representation dim D (F_n in Eq. 3).
    context_dim: int = 256
    # Decoder hidden width.
    decoder_dim: int = 256
    # Expansion factor T (= K, number of Gaussians predicted per input point).
    T: int = 5
    # Neighborhood size M for the M-nearest-neighbor graph.
    M: int = 3
    # Hidden layout for the shared point encoder (6 -> ... -> embed_dim).
    point_mlp_hidden: tuple = (64, 128)
    # Bound (in scene units) applied to the Tanh-activated position offset.
    # Paper bounds the raw offset to (-1, 1); set >1 if the scene is metric and
    # the K nearest dense Gaussians may sit farther than 1 unit from the SfM point.
    pos_offset_scale: float = 1.0

    # --- Ablation toggles (Table IV). When an attribute is disabled it falls
    # back to a fixed default and is excluded from the loss. ---
    predict_color: bool = True
    predict_opacity: bool = True
    predict_scale_rot: bool = True
    # Fixed defaults used for disabled attributes (mirrors 3DGS heuristics).
    default_opacity: float = 0.1
    default_scale: float = 0.01


class GSNet(nn.Module):
    """Point-to-Gaussian network.

    Given sparse SfM points (each with position + color) and, for every point,
    its M nearest neighbors, predict T Gaussian primitives per point using an
    incremental (offset-based) parameterization.
    """

    def __init__(self, cfg: GSNetConfig = None):
        super().__init__()
        cfg = cfg or GSNetConfig()
        self.cfg = cfg
        T = cfg.T

        # Weight-shared per-point encoder: 6 -> ... -> embed_dim (f_n).
        self.point_encoder = _mlp(
            (cfg.in_dim, *cfg.point_mlp_hidden, cfg.embed_dim)
        )

        # Context aggregator h_Theta: concat(center, M neighbors) -> F_n.
        self.context_encoder = _mlp(
            (cfg.embed_dim * (cfg.M + 1), cfg.context_dim, cfg.context_dim)
        )

        # Per-attribute prediction heads, each emitting T * dim values from F_n.
        self.shared_decoder = _mlp((cfg.context_dim, cfg.decoder_dim), last_act=True)
        self.head_mu = nn.Linear(cfg.decoder_dim, T * 3)   # delta mu_hat
        self.head_rgb = nn.Linear(cfg.decoder_dim, T * 3)  # delta C_rgb_hat
        self.head_quat = nn.Linear(cfg.decoder_dim, T * 4)  # q_hat
        self.head_scale = nn.Linear(cfg.decoder_dim, T * 3)  # s_hat
        self.head_opacity = nn.Linear(cfg.decoder_dim, T * 1)  # alpha_hat

    # ------------------------------------------------------------------ #
    def encode(self, center_feat: torch.Tensor, neighbor_feat: torch.Tensor):
        """Geometry-aware feature encoding (Eq. 3).

        Args:
            center_feat:   (B, 6)        raw [xyz; rgb] of each point.
            neighbor_feat: (B, M, 6)     raw [xyz; rgb] of its M neighbors.
        Returns:
            F_n: (B, context_dim)
        """
        f_center = self.point_encoder(center_feat)               # (B, d)
        f_neigh = self.point_encoder(neighbor_feat)              # (B, M, d)
        cat = torch.cat(
            [f_center, f_neigh.flatten(start_dim=1)], dim=-1
        )                                                        # (B, (M+1)*d)
        return self.context_encoder(cat)                         # (B, D)

    # ------------------------------------------------------------------ #
    def forward(self, center_xyz, center_rgb, neighbor_xyz, neighbor_rgb):
        """Predict T Gaussian primitives per input point.

        Args:
            center_xyz:   (B, 3)
            center_rgb:   (B, 3)   in [0, 1]
            neighbor_xyz: (B, M, 3)
            neighbor_rgb: (B, M, 3) in [0, 1]
        Returns dict of predictions (all batched as (B, T, ...)).
        """
        cfg = self.cfg
        B = center_xyz.shape[0]
        T = cfg.T

        center_feat = torch.cat([center_xyz, center_rgb], dim=-1)
        neighbor_feat = torch.cat([neighbor_xyz, neighbor_rgb], dim=-1)

        F = self.encode(center_feat, neighbor_feat)              # (B, D)
        h = self.shared_decoder(F)                               # (B, decoder_dim)

        # --- Position: incremental, bounded by Tanh (Eq. 4) ---
        delta_mu = torch.tanh(self.head_mu(h).view(B, T, 3)) * cfg.pos_offset_scale
        mu = center_xyz.unsqueeze(1) + delta_mu                  # mu_hat

        # --- Color: incremental, Sigmoid offset on the 0-order SH (diffuse) ---
        if cfg.predict_color:
            delta_rgb = torch.sigmoid(self.head_rgb(h).view(B, T, 3))
        else:
            delta_rgb = torch.zeros(B, T, 3, device=center_xyz.device)
        rgb = (center_rgb.unsqueeze(1) + delta_rgb).clamp(0.0, 1.0)

        # --- Rotation & scale -> covariance (Eq. 4) ---
        if cfg.predict_scale_rot:
            quat = self.head_quat(h).view(B, T, 4)
            quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scale = torch.sigmoid(self.head_scale(h).view(B, T, 3))
        else:
            quat = torch.zeros(B, T, 4, device=center_xyz.device)
            quat[..., 0] = 1.0
            scale = torch.full((B, T, 3), cfg.default_scale, device=center_xyz.device)
        rot = quat_to_rot(quat)                                  # (B, T, 3, 3)

        # --- Opacity: Tanh in (-1, 1); negatives discarded at render time ---
        if cfg.predict_opacity:
            opacity = torch.tanh(self.head_opacity(h).view(B, T, 1))
        else:
            opacity = torch.full((B, T, 1), cfg.default_opacity, device=center_xyz.device)

        return {
            "delta_mu": delta_mu,      # applied (post-Tanh) position offset
            "mu": mu,                  # absolute position mu_hat
            "delta_rgb": delta_rgb,    # applied (post-Sigmoid) color offset
            "rgb": rgb,                # absolute diffuse color in [0, 1]
            "quat": quat,              # unit quaternion
            "rot": rot,                # rotation matrix R_hat
            "scale": scale,            # diagonal scale entries S_hat in (0, 1)
            "opacity": opacity,        # alpha_hat in (-1, 1)
        }
