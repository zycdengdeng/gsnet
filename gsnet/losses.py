#
# GS-Net training objective (Sec. IV-C, Eq. 5-7).
#
# Decoupled geometric and appearance losses between each predicted primitive
# g_hat_{n,t} and its paired pseudo-ground-truth g^gt_{n,t} (the t-th of the K
# nearest dense Gaussians for input point p_n). T = K, so pairing is by index.
#

import torch
import torch.nn.functional as F

from .model import quat_to_rot


def gsnet_loss(pred, gt, cfg=None):
    """Compute the GS-Net loss.

    Args:
        pred: dict from GSNet.forward, each value (B, T, ...).
        gt: dict of pseudo-GT targets, each (B, T, ...):
            - "mu":      (B, T, 3) absolute target positions mu^gt
            - "rgb":     (B, T, 3) target diffuse color in [0, 1]
            - "rot":     (B, T, 3, 3) OR "quat": (B, T, 4) target rotation
            - "scale":   (B, T, 3) target diagonal scale (actual, not log)
            - "opacity": (B, T, 1) target opacity in (0, 1)
            - "center_xyz": (B, 3) input position, to form delta mu^gt
            - "center_rgb": (B, 3) input color, to form delta C^gt
        cfg: optional GSNetConfig controlling which terms are active.
    Returns:
        (total_loss, dict of scalar sub-losses) — already averaged over B and T.
    """
    predict_color = getattr(cfg, "predict_color", True) if cfg else True
    predict_opacity = getattr(cfg, "predict_opacity", True) if cfg else True
    predict_scale_rot = getattr(cfg, "predict_scale_rot", True) if cfg else True

    center_xyz = gt["center_xyz"].unsqueeze(1)   # (B, 1, 3)
    center_rgb = gt["center_rgb"].unsqueeze(1)   # (B, 1, 3)

    # ---- Geometric loss (Eq. 5) ----
    # Position offset: ||delta_mu_hat - delta_mu^gt||_2^2
    delta_mu_gt = gt["mu"] - center_xyz                       # (B, T, 3)
    loss_mu = ((pred["delta_mu"] - delta_mu_gt) ** 2).sum(-1)  # (B, T)

    loss_geom = loss_mu
    loss_rot = torch.zeros_like(loss_mu)
    loss_scale = torch.zeros_like(loss_mu)
    if predict_scale_rot:
        gt_rot = gt.get("rot")
        if gt_rot is None:
            gt_rot = quat_to_rot(gt["quat"])
        # ||R_hat - R^gt||_F^2
        loss_rot = ((pred["rot"] - gt_rot) ** 2).sum(dim=(-2, -1))  # (B, T)
        # ||S_hat - S^gt||_F^2 (diagonal -> sum of squared diagonal diffs)
        loss_scale = ((pred["scale"] - gt["scale"]) ** 2).sum(-1)  # (B, T)
        loss_geom = loss_geom + loss_rot + loss_scale

    # ---- Appearance loss (Eq. 6) ----
    loss_app = torch.zeros_like(loss_mu)
    loss_rgb = torch.zeros_like(loss_mu)
    loss_opacity = torch.zeros_like(loss_mu)
    if predict_color:
        delta_rgb_gt = gt["rgb"] - center_rgb                 # (B, T, 3)
        loss_rgb = ((pred["delta_rgb"] - delta_rgb_gt) ** 2).sum(-1)
        loss_app = loss_app + loss_rgb
    if predict_opacity:
        # (max(alpha_hat, 0) - alpha^gt)^2
        alpha = pred["opacity"].clamp_min(0.0)
        loss_opacity = ((alpha - gt["opacity"]) ** 2).sum(-1)
        loss_app = loss_app + loss_opacity

    per_primitive = loss_geom + loss_app                      # (B, T)
    total = per_primitive.mean()                              # avg over N and T

    logs = {
        "loss": total.detach(),
        "loss_mu": loss_mu.mean().detach(),
        "loss_rot": loss_rot.mean().detach(),
        "loss_scale": loss_scale.mean().detach(),
        "loss_rgb": loss_rgb.mean().detach(),
        "loss_opacity": loss_opacity.mean().detach(),
    }
    return total, logs
