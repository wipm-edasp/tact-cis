"""COD: Capability-Opportunity Decomposition for typed protein-ligand contacts.

Interaction logit for ligand atom i and type k:
    s_ik = capability_ik + logsumexp_j( opportunity_jk + complementarity_ijk )
where capability is ligand-only, opportunity is pocket-only, and
complementarity is pairwise (ligand atom x pocket atom). Non-capable atoms are
gated out. Training adds an opportunity-perturbation loss: masking the true
partner pocket atoms must lower the interaction logit.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N_TYPES, SEED
from .metrics import typed_metrics
from .model import GeoCrossAttention, LigandEncoder, _MLP, masked_bce
from .splits import leakage_summary
from .train import set_seed
from .egnn import JointEGNN

CUTOFFS = (3.6, 4.2, 5.5, 5.0, 4.0, 3.5)


class CODModel(nn.Module):
    def __init__(self, lig_dim=37, pocket_dim=24, dim=64, n_types=N_TYPES,
                 temp=0.5, use_decomp=True, n_heads=4, backbone="mlp",
                 use_cap_gate=True):
        super().__init__()
        self.use_decomp = use_decomp
        self.temp = temp
        self.backbone = backbone
        self.use_cap_gate = use_cap_gate
        if backbone == "egnn":
            self.joint = JointEGNN(lig_dim, pocket_dim, dim, n_layers=2,
                                   k=12, cutoff=8.0, pocket_cutoff=10.0)
        else:
            self.lig = LigandEncoder(lig_dim, dim)
            self.pocket = _MLP([pocket_dim, dim, dim])
        self.cap_head = nn.Linear(dim, n_types)
        self.opp_head = nn.Linear(dim, n_types)
        self.comp = nn.Sequential(
            nn.Linear(2 * dim + 16, dim), nn.SiLU(), nn.Linear(dim, n_types)
        )
        self.attn = GeoCrossAttention(dim, n_heads)
        self.plain = nn.Linear(dim, n_types)
        self.null_pocket = nn.Parameter(torch.zeros(dim))
        self.register_buffer("centers", torch.linspace(0.0, 12.0, 16))
        self.gamma = 12.0 / 16.0

    def forward(self, lig_feats, lig_xyz, edge_index, edge_attr,
                pocket_feats, pocket_xyz, cap, ablate="none"):
        gate = cap if self.use_cap_gate else torch.ones_like(cap)
        if self.backbone == "egnn":
            h, p = self.joint(lig_feats, lig_xyz, edge_index, edge_attr,
                              pocket_feats, pocket_xyz)
            pxyz = pocket_xyz
        else:
            h = self.lig(lig_feats, edge_index, edge_attr)
            if ablate == "no_pocket":
                p = self.null_pocket.unsqueeze(0)
                pxyz = lig_xyz.mean(dim=0, keepdim=True)
            else:
                p = self.pocket(pocket_feats)
                pxyz = pocket_xyz
        if not self.use_decomp:
            a = self.attn(h, p, lig_xyz, pxyz)
            logits = self.plain(h + a) - 20.0 * (1.0 - gate)
            return logits, None, None
        n, m = h.shape[0], p.shape[0]
        d = torch.cdist(lig_xyz, pxyz)
        rbf = torch.exp(-((d.unsqueeze(-1) - self.centers) ** 2) / (2 * self.gamma ** 2))
        if ablate == "no_geometry":
            rbf = torch.zeros_like(rbf)
        hi = h[:, None, :].expand(n, m, -1)
        pj = p[None, :, :].expand(n, -1, -1)
        z = self.comp(torch.cat([hi, pj, rbf], dim=-1))       # (n, m, T)
        o = self.opp_head(p)                                  # (m, T)
        s = o[None, :, :] + z                                 # (n, m, T)
        s_opp = self.temp * torch.logsumexp(s / self.temp, dim=1)  # (n, T)
        c = self.cap_head(h)                                  # (n, T)
        logits = (c + s_opp) - 20.0 * (1.0 - gate)
        return logits, c, s



def masked_bce_cap(logits, labels, sup, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none",
        pos_weight=None if pos_weight is None else pos_weight.to(logits.device),
    )
    w = sup.float()
    return (loss * w).sum() / w.sum().clamp_min(1.0)
def prepare_cod(samples, device):
    out = []
    for s in samples:
        out.append({
            "lig_feats": torch.tensor(s["lig_feats"], dtype=torch.float32, device=device),
            "lig_xyz": torch.tensor(s["lig_xyz"], dtype=torch.float32, device=device),
            "edge_index": torch.tensor(s["edge_index"], dtype=torch.long, device=device),
            "edge_attr": torch.tensor(s["edge_attr"], dtype=torch.float32, device=device),
            "pocket_feats": torch.tensor(s["pocket_feats"], dtype=torch.float32, device=device),
            "pocket_xyz": torch.tensor(s["pocket_xyz"], dtype=torch.float32, device=device),
            "labels": torch.tensor(s["labels"], dtype=torch.float32, device=device),
            "mask": torch.tensor(s["mask"], dtype=torch.bool, device=device),
            "capability": torch.tensor(s["capability"], dtype=torch.float32, device=device),
            "pocket_roles": torch.tensor(s["pocket_roles"], dtype=torch.float32, device=device),
            "np_labels": s["labels"],
            "np_mask": s["mask"],
            "np_capability": s["capability"],
        })
    return out


def compute_pos_weight_cod(prepared, train_idx, use_cap_gate=True):
    parts = []
    for i in train_idx:
        t = prepared[i]
        if t["np_mask"].sum() == 0:
            continue
        if use_cap_gate:
            sup = t["np_mask"][:, None] & (t["np_capability"] > 0.5)
        else:
            sup = t["np_mask"][:, None] & np.ones_like(t["np_capability"], dtype=bool)
        if sup.sum() > 0:
            parts.append(t["np_labels"][sup])
    parts = [p for p in parts if p.size > 0]
    if not parts:
        n_types = prepared[0]["np_labels"].shape[1] if prepared else N_TYPES
        return torch.ones(n_types, dtype=torch.float32)
    labs = np.concatenate(parts)
    pos = labs.sum(axis=0)
    neg = (1.0 - labs).sum(axis=0)
    return torch.tensor(np.clip(neg / np.maximum(pos, 1.0), 1.0, 50.0),
                        dtype=torch.float32)


def perturb_loss(model, t, s, c, logits, rng, n_max=8, margin=0.5):
    labels, mask = t["labels"], t["mask"]
    cap, roles = t["capability"], t["pocket_roles"]
    pos = (labels > 0.5) & mask[:, None] & (cap > 0.5)
    idx = torch.nonzero(pos, as_tuple=False)
    if idx.shape[0] == 0:
        return logits.sum() * 0.0
    pick = idx[torch.randperm(idx.shape[0], device=idx.device)[:n_max]]
    d = torch.cdist(t["lig_xyz"], t["pocket_xyz"])
    total = logits.sum() * 0.0
    used = 0
    for i, k in pick.tolist():
        partner = (roles[:, k] > 0.5) & (d[i] <= CUTOFFS[k])
        if partner.sum() == 0:
            continue
        s_ik = s[i, :, k].clone()
        s_ik[partner] = -1e9
        s_opp_m = model.temp * torch.logsumexp(s_ik / model.temp, dim=0)
        logit_m = c[i, k] + s_opp_m
        total = total + F.relu(margin + logit_m - logits[i, k])
        used += 1
    return total / max(1, used)


def train_cod_epoch(model, prepared, train_idx, opt, batch_size=8,
                    lambda_perturb=0.1, pos_weight=None, rng=None, ablate="none",
                    use_cap_gate=True):
    model.train()
    rng = rng or random.Random(SEED)
    order = list(train_idx)
    rng.shuffle(order)
    total, n = 0.0, 0
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        opt.zero_grad()
        loss = 0.0
        for i in batch:
            t = prepared[i]
            logits, c, s = model(
                t["lig_feats"], t["lig_xyz"], t["edge_index"], t["edge_attr"],
                t["pocket_feats"], t["pocket_xyz"], t["capability"], ablate=ablate,
            )
            if use_cap_gate:
                sup = t["mask"][:, None] & (t["capability"] > 0.5)
            else:
                sup = t["mask"][:, None] & torch.ones_like(t["capability"], dtype=torch.bool)
            loss = loss + masked_bce_cap(logits, t["labels"], sup, pos_weight)
            if lambda_perturb > 0 and s is not None:
                loss = loss + lambda_perturb * perturb_loss(model, t, s, c, logits, rng)
        loss = loss / max(1, len(batch))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += float(loss.detach())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def predict_cod(model, prepared, idx, ablate="none"):
    model.eval()
    logits, labels, mask, cap, sample_index = [], [], [], [], []
    for i in idx:
        t = prepared[i]
        lg, _, _ = model(
            t["lig_feats"], t["lig_xyz"], t["edge_index"], t["edge_attr"],
            t["pocket_feats"], t["pocket_xyz"], t["capability"], ablate=ablate,
        )
        logits.append(lg.cpu().numpy())
        labels.append(t["np_labels"])
        mask.append(t["np_mask"])
        cap.append(t["np_capability"])
        sample_index.append(np.full(len(t["np_labels"]), i, dtype=np.int64))
    return {
        "logits": np.concatenate(logits), "labels": np.concatenate(labels),
        "mask": np.concatenate(mask), "capability": np.concatenate(cap),
        "sample_index": np.concatenate(sample_index),
    }


def run_cod_fold(samples, train_idx, test_idx, use_decomp=True,
                 lambda_perturb=0.1, epochs=20, lr=2e-3, batch_size=8,
                 dim=64, device=None, seed=SEED, prepared=None,
                 return_preds=False, verbose=False, ablate="none",
                 backbone="mlp", use_cap_gate=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    if prepared is None:
        prepared = prepare_cod(samples, device)
    lig_dim = samples[0]["lig_feats"].shape[1]
    pocket_dim = samples[0]["pocket_feats"].shape[1]
    model = CODModel(lig_dim=lig_dim, pocket_dim=pocket_dim, dim=dim,
                     use_decomp=use_decomp, backbone=backbone,
                     use_cap_gate=use_cap_gate).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pw = compute_pos_weight_cod(prepared, list(train_idx), use_cap_gate=use_cap_gate)
    rng = random.Random(seed)
    for ep in range(epochs):
        loss = train_cod_epoch(model, prepared, list(train_idx), opt,
                              batch_size=batch_size, lambda_perturb=lambda_perturb,
                              pos_weight=pw, rng=rng, ablate=ablate,
                              use_cap_gate=use_cap_gate)
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  epoch {ep+1}/{epochs} loss={loss:.4f}", flush=True)
    preds = predict_cod(model, prepared, list(test_idx), ablate=ablate)
    metrics = typed_metrics(preds["logits"], preds["labels"], preds["mask"],
                            preds["capability"])
    metrics.update(leakage_summary(samples, train_idx, test_idx))
    if return_preds:
        return model, metrics, preds
    return model, metrics
