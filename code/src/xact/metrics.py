"""Metrics for typed and untyped contact prediction."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import INTERACTION_TYPES


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def safe_auc(y, p) -> float:
    y = np.asarray(y)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y, p) -> float:
    y = np.asarray(y)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def typed_metrics(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray,
                  capability: np.ndarray | None = None,
                  types=INTERACTION_TYPES) -> dict:
    probs = sigmoid(logits)
    labels = labels.astype(int)
    mask = mask.astype(bool)
    out = {}
    aucs, aps, cap_aucs, cap_aps = [], [], [], []
    for k, name in enumerate(types):
        y = labels[mask, k]
        p = probs[mask, k]
        auc = safe_auc(y, p)
        ap = safe_ap(y, p)
        out[f"auc_{name}"] = auc
        out[f"ap_{name}"] = ap
        if not np.isnan(auc):
            aucs.append(auc)
        if not np.isnan(ap):
            aps.append(ap)
        if capability is not None:
            cm = mask & (capability[:, k] > 0.5)
            yc = labels[cm, k]
            pc = probs[cm, k]
            cauc = safe_auc(yc, pc)
            cap = safe_ap(yc, pc)
            out[f"auc_cap_{name}"] = cauc
            out[f"ap_cap_{name}"] = cap
            out[f"n_cap_{name}"] = int(cm.sum())
            out[f"n_pos_cap_{name}"] = int(yc.sum())
            out[f"positive_rate_cap_{name}"] = (
                float(yc.mean()) if len(yc) else float("nan"))
            if not np.isnan(cauc):
                cap_aucs.append(cauc)
            if not np.isnan(cap):
                cap_aps.append(cap)
    out["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["macro_ap"] = float(np.mean(aps)) if aps else float("nan")
    if capability is not None:
        out["macro_auc_cap"] = float(np.mean(cap_aucs)) if cap_aucs else float("nan")
        out["macro_ap_cap"] = float(np.mean(cap_aps)) if cap_aps else float("nan")
        cm_all = mask[:, None] & (capability > 0.5)
        y_all = labels[cm_all]
        p_all = probs[cm_all]
        out["auc_all_cap"] = safe_auc(y_all, p_all)
        out["ap_all_cap"] = safe_ap(y_all, p_all)
        out["n_capable_pairs"] = int(cm_all.sum())
        out["n_capable_pos"] = int(y_all.sum())
        out["positive_rate_cap"] = (
            float(y_all.mean()) if len(y_all) else float("nan"))
    any_y = (labels[mask].sum(axis=1) > 0).astype(int)
    any_p = probs[mask].max(axis=1)
    out["auc_contact_any"] = safe_auc(any_y, any_p)
    out["ap_contact_any"] = safe_ap(any_y, any_p)
    out["n_atoms"] = int(mask.sum())
    out["positive_rate"] = float(labels[mask].mean()) if mask.sum() else float("nan")
    return out


def untyped_metrics(logits: np.ndarray, contact_any: np.ndarray, mask: np.ndarray) -> dict:
    probs = sigmoid(logits).reshape(-1)
    mask = mask.astype(bool)
    y = contact_any[mask].astype(int)
    return {
        "auc_contact": safe_auc(y, probs[mask]),
        "ap_contact": safe_ap(y, probs[mask]),
        "n_atoms": int(mask.sum()),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
    }
