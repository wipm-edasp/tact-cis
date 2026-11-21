"""Conformal interaction sets (CIS) for typed protein-ligand contacts.

Scores are s_ik = 1 - p_ik. Split-conformal thresholds are fitted on a
target-disjoint calibration set, per interaction type (marginal) or per
(type, pocket family) cell (Mondrian). Sets contain every capable type whose
score is below the cell threshold.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

FAMILY = {
    "kinase": {"EGFR", "ABL1", "JAK2", "CDK2", "SRC", "MET", "p38", "VEGFR2", "PI3Ka", "AKT1"},
    "protease": {"Thrombin", "FXa", "Caspase3", "CathepsinK"},
    "gpcr": {"A2A", "M1"},
    "nr": {"ERa", "PPARg", "AR", "VDR", "GR"},
    "other": {"COX2", "CA2", "HSP90", "BCL2", "DHFR", "PDE5", "ACE", "HMGCR"},
}
TARGET_FAMILY = {t: f for f, ts in FAMILY.items() for t in ts}


def family_of(target: str) -> str:
    return TARGET_FAMILY.get(target, "other")


def quantile_threshold(scores, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(np.asarray(scores, float), level, method="higher"))


def fit_thresholds(probs, labels, mask, cap, groups, alpha, mode="type"):
    """Return a dict of thresholds for mode in {global_single, type, family}."""
    q = {}
    tp = mask[:, None] & (cap > 0.5) & (labels > 0.5)
    all_scores = (1.0 - probs)[tp]
    q["global_single"] = quantile_threshold(all_scores, alpha)
    for k in range(probs.shape[1]):
        base = tp[:, k]
        if base.sum() < 5:
            q[("type", k)] = 1.0
            continue
        q[("type", k)] = quantile_threshold(1.0 - probs[base, k], alpha)
        for g in np.unique(groups):
            sel = base & (groups == g)
            if sel.sum() >= 10:
                q[("family", k, g)] = quantile_threshold(1.0 - probs[sel, k], alpha)
            else:
                q[("family", k, g)] = q[("type", k)]
    return q


def predict_sets(probs, cap, q, groups, mode="type"):
    n, k = probs.shape
    sets = np.zeros((n, k), dtype=bool)
    if mode == "global_single":
        th = q["global_single"]
        sets = ((1.0 - probs) <= th) & (cap > 0.5)
        return sets
    th = np.zeros((n, k), dtype=float)
    for kk in range(k):
        if mode == "family":
            mapping = {g: q.get(("family", kk, g), q.get(("type", kk), 1.0))
                       for g in np.unique(groups)}
            th[:, kk] = [mapping[g] for g in groups]
        else:
            th[:, kk] = q.get(("type", kk), 1.0)
    return ((1.0 - probs) <= th) & (cap > 0.5)


def topk_sets(probs, cap, k):
    sets = np.zeros_like(probs, dtype=bool)
    for i in range(probs.shape[0]):
        idx = np.argsort(-probs[i])[:k]
        sets[i, idx] = True
    return sets & (cap > 0.5)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(logits, labels, mask, cap):
    """Fit a single temperature on calibration logits by minimizing BCE."""
    from scipy.optimize import minimize_scalar
    sel = mask[:, None] & (cap > 0.5)
    x = np.asarray(logits[sel], float)
    y = np.asarray(labels[sel], float)
    if len(x) == 0 or len(np.unique(y)) < 2:
        return 1.0
    def nll(log_t):
        p = sigmoid(x / np.exp(log_t))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    res = minimize_scalar(nll, bounds=(-1.0, 1.5), method="bounded")
    return float(np.exp(res.x))



def crc_threshold(scores, alpha, delta=0.1):
    """Smallest threshold tau with Hoeffding UCB on FNR <= alpha."""
    scores = np.asarray(scores, float)
    n = len(scores)
    if n == 0:
        return 1.0
    grid = np.unique(scores)
    bound = np.sqrt(np.log(1.0 / delta) / (2.0 * n))
    for tau in grid:
        fnr = float(np.mean(scores > tau))
        if fnr + bound <= alpha:
            return float(tau)
    return float(grid[-1])


def chemistry_score(probs, lig_xyz, pocket_xyz, pocket_roles, cutoffs, lam=0.5, mode="geo_add", water_xyz=None):
    """Interaction-aware nonconformity: model score + geometric/partner term.

    geo_add:     s = 1 - p + lam * clip(d_min / cutoff, 0, 3)
    geo_partner: s = 1 - p + lam / (1 + n_partners_within_cutoff)
    """
    s = 1.0 - np.asarray(probs, float)
    d = np.linalg.norm(np.asarray(lig_xyz)[:, None, :] - np.asarray(pocket_xyz)[None, :, :], axis=2)
    for k in range(s.shape[1]):
        if k == 5 and water_xyz is not None and len(water_xyz):
            dw = np.linalg.norm(
                np.asarray(lig_xyz)[:, None, :] - np.asarray(water_xyz)[None, :, :],
                axis=2)
            dk = dw.min(axis=1)
            nk = (dw <= cutoffs[k]).sum(axis=1)
            if mode == "geo_add":
                s[:, k] += lam * np.clip(dk / max(cutoffs[k], 1e-6), 0.0, 3.0)
            elif mode == "geo_partner":
                s[:, k] += lam / (1.0 + nk)
            continue
        comp = np.asarray(pocket_roles)[:, k] > 0.5
        if not comp.any():
            continue
        dk = d[:, comp].min(axis=1)
        nk = (d[:, comp] <= cutoffs[k]).sum(axis=1)
        if mode == "geo_add":
            s[:, k] += lam * np.clip(dk / max(cutoffs[k], 1e-6), 0.0, 3.0)
        elif mode == "geo_partner":
            s[:, k] += lam / (1.0 + nk)
    return s
def evaluate_sets(sets, labels, mask, cap, groups, targets=None):
    tp = mask[:, None] & (cap > 0.5) & (labels > 0.5)
    cov = float((sets & tp).sum() / max(1, tp.sum()))
    size = float(sets.sum(axis=1).mean()) if len(sets) else float("nan")
    impossible = sets & (cap <= 0.5)
    impossible_rate = float(impossible.sum() / max(1, sets.sum()))
    per_family = {}
    for g in np.unique(groups):
        sel = tp & (groups == g)[:, None]
        if sel.sum() == 0:
            continue
        per_family[g] = {
            "coverage": float((sets & sel).sum() / sel.sum()),
            "n_true": int(sel.sum()),
        }
    per_type = {}
    for k in range(sets.shape[1]):
        sel = tp[:, k]
        if sel.sum() == 0:
            continue
        cmask = cap[:, k] > 0.5
        per_type[str(k)] = {
            "coverage": float((sets[:, k] & sel).sum() / sel.sum()),
            "n_true": int(sel.sum()),
            "size": float(sets[cmask, k].mean()) if cmask.any() else None,
        }
    per_target = {}
    if targets is not None:
        for t in np.unique(targets):
            atom_sel = (targets == t) & mask
            if not atom_sel.any():
                continue
            sel = tp & (targets == t)[:, None]
            if sel.sum() == 0:
                continue
            per_target[str(t)] = {
                "covered": int((sets & sel).sum()),
                "n_true": int(sel.sum()),
                "n_atoms": int(atom_sel.sum()),
                "size": float(sets[atom_sel].sum(axis=1).mean()),
            }
    return {"coverage": cov, "size": size, "per_family": per_family,
            "per_type": per_type, "per_target": per_target,
            "n_true": int(tp.sum()), "impossible_rate": impossible_rate}
