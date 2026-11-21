"""Minimal E(n)-equivariant GNN (EGNN) backbone for COD.

Joint graph over ligand + pocket atoms: ligand bonds, ligand-pocket edges within
a cutoff, and pocket kNN edges. Feature and coordinate updates follow Satorras
et al. (EGNN), with scalar gates on the coordinate displacement so the output
features stay invariant and the coordinates stay equivariant.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EGNNLayer(nn.Module):
    def __init__(self, dim: int, edge_dim: int = 22):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * dim + edge_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.coord = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1)
        )
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, pos, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        m = self.msg(torch.cat([h[src], h[dst], edge_attr], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        h = self.norm(h + self.upd(torch.cat([h, agg], dim=-1)))
        w = self.coord(m)                       # (E, 1) scalar gate
        diff = pos[src] - pos[dst]              # (E, 3)
        dpos = torch.zeros_like(pos)
        dpos.index_add_(0, dst, diff * w)
        pos = pos + self.scale * dpos
        return h, pos


class JointEGNN(nn.Module):
    def __init__(self, lig_dim: int = 37, pocket_dim: int = 24, dim: int = 64,
                 n_layers: int = 2, k: int = 12, cutoff: float = 8.0,
                 pocket_cutoff: float = 10.0, n_rbf: int = 16):
        super().__init__()
        self.lig_proj = nn.Linear(lig_dim, dim)
        self.pocket_proj = nn.Linear(pocket_dim, dim)
        self.layers = nn.ModuleList([EGNNLayer(dim) for _ in range(n_layers)])
        self.k = k
        self.cutoff = cutoff
        self.pocket_cutoff = pocket_cutoff
        self.register_buffer("centers", torch.linspace(0.0, 10.0, n_rbf))
        self.gamma = 10.0 / n_rbf

    def rbf(self, d):
        return torch.exp(-((d.unsqueeze(-1) - self.centers) ** 2) / (2 * self.gamma ** 2))

    def forward(self, lig_feats, lig_xyz, edge_index_lig, edge_attr_lig,
                pocket_feats, pocket_xyz):
        n, m = lig_feats.shape[0], pocket_feats.shape[0]
        if n == 0 or m == 0:
            return (self.lig_proj(lig_feats), self.pocket_proj(pocket_feats))
        h = torch.cat([self.lig_proj(lig_feats), self.pocket_proj(pocket_feats)], dim=0)
        pos = torch.cat([lig_xyz, pocket_xyz], dim=0)
        edges, eattrs = [], []
        zero6 = torch.zeros(1, 6, device=h.device)

        # ligand bonds
        if edge_index_lig.shape[1] > 0:
            rbf0 = torch.zeros(edge_attr_lig.shape[0], 16, device=h.device)
            edges.append(edge_index_lig)
            eattrs.append(torch.cat([rbf0, edge_attr_lig], dim=-1))

        # ligand-pocket proximity (both directions)
        d = torch.cdist(lig_xyz, pocket_xyz)
        ii, jj = torch.where(d <= self.cutoff)
        if ii.numel() > 0:
            rbf = self.rbf(d[ii, jj])
            e = torch.cat([rbf, zero6.expand(ii.numel(), 6)], dim=-1)
            edges.append(torch.stack([ii, jj + n], dim=0))
            eattrs.append(e)
            edges.append(torch.stack([jj + n, ii], dim=0))
            eattrs.append(e)

        # pocket kNN (both directions)
        dpp = torch.cdist(pocket_xyz, pocket_xyz)
        k = min(self.k, m - 1)
        if k > 0:
            vals, idx = torch.topk(dpp, k + 1, largest=False)
            src = torch.arange(m, device=h.device).repeat_interleave(k)
            dst = idx[:, 1:k + 1].reshape(-1)
            dd = vals[:, 1:k + 1].reshape(-1)
            keep = dd <= self.pocket_cutoff
            src, dst, dd = src[keep], dst[keep], dd[keep]
            if src.numel() > 0:
                rbf = self.rbf(dd)
                e = torch.cat([rbf, zero6.expand(src.numel(), 6)], dim=-1)
                edges.append(torch.stack([src + n, dst + n], dim=0))
                eattrs.append(e)
                edges.append(torch.stack([dst + n, src + n], dim=0))
                eattrs.append(e)

        if edges:
            edge_index = torch.cat(edges, dim=1)
            edge_attr = torch.cat(eattrs, dim=0)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=h.device)
            edge_attr = torch.zeros(0, 22, device=h.device)

        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)
        return h[:n], h[n:]
