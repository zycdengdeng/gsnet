#
# Pluggable geometry-aware encoders for the GS-Net encoder-design ablation
# (rebuttal). All encoders consume identical inputs -- the center point's raw
# [xyz; rgb] feature, its M neighbors' raw features, and their coordinates --
# and output a context-aware representation F_n of dimension cfg.context_dim, so
# only the encoder is swapped while the expansion head, losses, and training
# schedule remain fixed.
#
# Variants (addressing the reviewer's request for stronger encoders):
#   "mlp_only"  : (a) per-point MLP, NO neighborhood aggregation (lower bound)
#   "concat"    : (b) center+neighbor feature concatenation -> MLP   (Ours)
#   "edgeconv"  : (c) EdgeConv / DGCNN graph encoder (edge feats + max-pool)
#   "attention" : (d) local self-attention (lightweight point-transformer)
#   "geom"      : (e) explicit geometric features (relative coords + distance)
#

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims, act=nn.ReLU, last_act=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


class MLPOnlyEncoder(nn.Module):
    """(a) Per-point MLP without neighborhood aggregation."""

    def __init__(self, cfg):
        super().__init__()
        self.point = _mlp((cfg.in_dim, *cfg.point_mlp_hidden, cfg.embed_dim))
        self.head = _mlp((cfg.embed_dim, cfg.context_dim, cfg.context_dim))

    def forward(self, center_feat, neighbor_feat, center_xyz, neighbor_xyz):
        return self.head(self.point(center_feat))


class ConcatMLPEncoder(nn.Module):
    """(b) Ours: concatenate center + M neighbor features, then MLP."""

    def __init__(self, cfg):
        super().__init__()
        self.point = _mlp((cfg.in_dim, *cfg.point_mlp_hidden, cfg.embed_dim))
        self.context = _mlp((cfg.embed_dim * (cfg.M + 1),
                             cfg.context_dim, cfg.context_dim))

    def forward(self, center_feat, neighbor_feat, center_xyz, neighbor_xyz):
        f_c = self.point(center_feat)                       # (B, d)
        f_n = self.point(neighbor_feat)                     # (B, M, d)
        cat = torch.cat([f_c, f_n.flatten(start_dim=1)], dim=-1)
        return self.context(cat)


class EdgeConvEncoder(nn.Module):
    """(c) DGCNN-style: edge features h([f_i; f_j - f_i]) + max-pooling."""

    def __init__(self, cfg):
        super().__init__()
        self.point = _mlp((cfg.in_dim, *cfg.point_mlp_hidden, cfg.embed_dim))
        self.edge = _mlp((2 * cfg.embed_dim, cfg.context_dim, cfg.context_dim))

    def forward(self, center_feat, neighbor_feat, center_xyz, neighbor_xyz):
        f_c = self.point(center_feat)                       # (B, d)
        f_n = self.point(neighbor_feat)                     # (B, M, d)
        M = f_n.shape[1]
        f_c_rep = f_c.unsqueeze(1).expand(-1, M, -1)        # (B, M, d)
        edge = torch.cat([f_c_rep, f_n - f_c_rep], dim=-1)  # (B, M, 2d)
        e = self.edge(edge)                                 # (B, M, D)
        return e.max(dim=1).values                          # (B, D)


class AttentionEncoder(nn.Module):
    """(d) Lightweight local self-attention (point-transformer style).

    Center forms the query; the M neighbors form keys/values. Relative
    coordinates are injected as a learned positional bias on values.
    """

    def __init__(self, cfg):
        super().__init__()
        d = cfg.embed_dim
        self.point = _mlp((cfg.in_dim, *cfg.point_mlp_hidden, d))
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.pos = _mlp((3, d), last_act=True)              # relative-coord PE
        self.out = _mlp((2 * d, cfg.context_dim, cfg.context_dim))
        self.scale = d ** -0.5

    def forward(self, center_feat, neighbor_feat, center_xyz, neighbor_xyz):
        f_c = self.point(center_feat)                       # (B, d)
        f_n = self.point(neighbor_feat)                     # (B, M, d)
        rel = neighbor_xyz - center_xyz.unsqueeze(1)        # (B, M, 3)
        pe = self.pos(rel)                                  # (B, M, d)
        q = self.q(f_c).unsqueeze(1)                        # (B, 1, d)
        k = self.k(f_n) + pe                                # (B, M, d)
        v = self.v(f_n) + pe                                # (B, M, d)
        attn = (q * k).sum(-1) * self.scale                 # (B, M)
        attn = F.softmax(attn, dim=-1).unsqueeze(-1)        # (B, M, 1)
        ctx = (attn * v).sum(1)                             # (B, d)
        return self.out(torch.cat([f_c, ctx], dim=-1))     # (B, D)


class GeomEncoder(nn.Module):
    """(e) Explicit geometry: augment neighbor features with relative
    coordinates and distance before concat-MLP aggregation."""

    def __init__(self, cfg):
        super().__init__()
        # center: raw feat padded with zeros for the 4 geometric channels.
        self.point = _mlp((cfg.in_dim + 4, *cfg.point_mlp_hidden, cfg.embed_dim))
        self.context = _mlp((cfg.embed_dim * (cfg.M + 1),
                             cfg.context_dim, cfg.context_dim))

    def forward(self, center_feat, neighbor_feat, center_xyz, neighbor_xyz):
        B, M, _ = neighbor_feat.shape
        rel = neighbor_xyz - center_xyz.unsqueeze(1)        # (B, M, 3)
        dist = rel.norm(dim=-1, keepdim=True)               # (B, M, 1)
        n_in = torch.cat([neighbor_feat, rel, dist], dim=-1)   # (B, M, 10)
        c_in = torch.cat([center_feat,
                          torch.zeros(B, 4, device=center_feat.device)], dim=-1)
        f_c = self.point(c_in)                              # (B, d)
        f_n = self.point(n_in)                              # (B, M, d)
        cat = torch.cat([f_c, f_n.flatten(start_dim=1)], dim=-1)
        return self.context(cat)


class GeoEdgeConvEncoder(nn.Module):
    """Geometric EdgeConv: combines graph aggregation (c) with explicit geometry
    (e) -- edge features include relative coordinates and distance, then max-pool.
    Motivated by the ablation where (e) explicit-geometry beat plain (c) edgeconv."""

    def __init__(self, cfg):
        super().__init__()
        self.point = _mlp((cfg.in_dim, *cfg.point_mlp_hidden, cfg.embed_dim))
        self.edge = _mlp((2 * cfg.embed_dim + 4, cfg.context_dim, cfg.context_dim))

    def forward(self, center_feat, neighbor_feat, center_xyz, neighbor_xyz):
        f_c = self.point(center_feat)
        f_n = self.point(neighbor_feat)
        M = f_n.shape[1]
        f_c_rep = f_c.unsqueeze(1).expand(-1, M, -1)
        rel = neighbor_xyz - center_xyz.unsqueeze(1)            # (B, M, 3)
        dist = rel.norm(dim=-1, keepdim=True)                  # (B, M, 1)
        edge = torch.cat([f_c_rep, f_n - f_c_rep, rel, dist], dim=-1)  # (B,M,2d+4)
        return self.edge(edge).max(dim=1).values               # (B, D)


_ENCODERS = {
    "mlp_only": MLPOnlyEncoder,
    "concat": ConcatMLPEncoder,
    "edgeconv": EdgeConvEncoder,
    "attention": AttentionEncoder,
    "geom": GeomEncoder,
    "geoedge": GeoEdgeConvEncoder,
}


def build_encoder(cfg):
    if cfg.encoder_type not in _ENCODERS:
        raise ValueError(f"unknown encoder_type '{cfg.encoder_type}'; "
                         f"choose from {list(_ENCODERS)}")
    return _ENCODERS[cfg.encoder_type](cfg)
