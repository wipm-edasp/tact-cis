"""TACT model: residue-blind pocket encoder + distance-aware typed contact head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N_TYPES


class _MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        self.net = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.net(x)


class LigandEncoder(nn.Module):
    def __init__(self, in_dim: int, dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.inp = _MLP([in_dim, dim, dim])
        self.msg = nn.ModuleList([_MLP([2 * dim + 6, dim, dim]) for _ in range(n_layers)])
        self.upd = nn.ModuleList([_MLP([2 * dim, dim, dim]) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])

    def forward(self, x, edge_index, edge_attr):
        h = self.inp(x)
        if edge_index.shape[1] == 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            m = msg(torch.cat([h[src], h[dst], edge_attr], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            h = norm(h + upd(torch.cat([h, agg], dim=-1)))
        return h


class GeoCrossAttention(nn.Module):
    """Cross-attention with a learned RBF bias on ligand-pocket distance."""

    def __init__(self, dim: int = 64, n_heads: int = 4, n_rbf: int = 16, rbf_max: float = 10.0):
        super().__init__()
        assert dim % n_heads == 0
        self.h = n_heads
        self.dk = dim // n_heads
        self.dv = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.register_buffer("centers", torch.linspace(0.0, rbf_max, n_rbf))
        self.gamma = rbf_max / n_rbf
        self.bias = nn.Linear(n_rbf, n_heads)

    def forward(self, h, p, lig_xyz, pocket_xyz):
        n, m = h.shape[0], p.shape[0]
        q = self.q(h).view(n, self.h, self.dk)
        k = self.k(p).view(m, self.h, self.dk)
        v = self.v(p).view(m, self.h, self.dv)
        d = torch.cdist(lig_xyz, pocket_xyz)
        rbf = torch.exp(-((d.unsqueeze(-1) - self.centers) ** 2) / (2 * self.gamma ** 2))
        bias = self.bias(rbf).permute(2, 0, 1)  # (h, n, m)
        logits = torch.einsum("nhd,mhd->hnm", q, k) / (self.dk ** 0.5) + bias
        a = torch.softmax(logits, dim=-1)
        out = torch.einsum("hnm,mhd->nhd", a, v).reshape(n, self.h * self.dv)
        return self.out(out)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class TACTModel(nn.Module):
    def __init__(
        self,
        lig_dim: int = 37,
        pocket_dim: int = 24,
        dim: int = 64,
        n_targets: int = 2,
        n_heads: int = 4,
        n_out: int = N_TYPES,
        use_adversary: bool = False,
    ):
        super().__init__()
        self.use_adversary = use_adversary
        self.lig = LigandEncoder(lig_dim, dim)
        self.pocket = _MLP([pocket_dim, dim, dim])
        self.null_pocket = nn.Parameter(torch.zeros(1, dim))
        self.attn = GeoCrossAttention(dim, n_heads)
        self.attn_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_out)
        self.pool = _MLP([dim, dim, dim])
        self.adv = nn.Linear(dim, n_targets)

    def forward_single(
        self,
        lig_feats,
        lig_xyz,
        edge_index,
        edge_attr,
        pocket_feats,
        pocket_xyz,
        lig_mask=None,
        use_pocket: bool = True,
    ):
        h = self.lig(lig_feats, edge_index, edge_attr)
        if use_pocket and pocket_feats is not None and len(pocket_feats) > 0:
            p = self.pocket(pocket_feats)
            pxyz = pocket_xyz
        else:
            p = self.null_pocket
            pxyz = lig_xyz.mean(dim=0, keepdim=True)
        a = self.attn(h, p, lig_xyz, pxyz)
        h = self.attn_norm(h + a)
        logits = self.head(h)
        if lig_mask is None:
            pooled = h.mean(dim=0)
        else:
            w = lig_mask.float().unsqueeze(-1)
            pooled = (h * w).sum(0) / w.sum().clamp_min(1.0)
        pair_vec = self.pool(pooled)
        return logits, pair_vec

    def forward(
        self,
        lig_feats,
        lig_xyz,
        edge_index,
        edge_attr,
        pocket_feats,
        pocket_xyz,
        lig_mask=None,
        use_pocket: bool = True,
        return_adv: bool = False,
    ):
        logits, pair_vec = self.forward_single(
            lig_feats, lig_xyz, edge_index, edge_attr, pocket_feats, pocket_xyz,
            lig_mask, use_pocket,
        )
        if return_adv and self.use_adversary:
            adv = self.adv(GradientReversal.apply(pair_vec, 1.0))
            return logits, pair_vec, adv
        return logits, pair_vec


def masked_bce(logits, labels, mask, pos_weight=None):
    if mask.sum() == 0:
        return logits.sum() * 0.0
    pw = None if pos_weight is None else pos_weight.to(logits.device)
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask], pos_weight=pw)


def swap_margin(p_true, p_swap, target_vec, margin: float = 0.2):
    t = F.normalize(target_vec, dim=-1, eps=1e-6)
    a = F.normalize(p_true, dim=-1, eps=1e-6)
    b = F.normalize(p_swap, dim=-1, eps=1e-6)
    sim_true = (a * t).sum()
    sim_swap = (b * t).sum()
    return F.relu(p_true.new_tensor(margin) + sim_swap - sim_true)
