"""Training and evaluation loops for TACT."""
from __future__ import annotations

import random

import numpy as np
import torch

from .config import N_TYPES, SEED
from .metrics import typed_metrics, untyped_metrics
from .model import TACTModel, masked_bce, swap_margin
from .splits import leakage_summary


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare(samples, device):
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
            "contact_any": torch.tensor(s["contact_any"], dtype=torch.float32, device=device),
            # numpy copies for metric computation
            "np_labels": s["labels"],
            "np_mask": s["mask"],
            "np_contact_any": s["contact_any"],
            "np_capability": s.get("capability"),
        })
    return out


def _labels_for(t, typed):
    if typed:
        return t["labels"]
    return t["contact_any"].float().unsqueeze(1)


def compute_pos_weight(prepared, train_idx, typed=True):
    if typed:
        parts = [
            prepared[i]["np_labels"][prepared[i]["np_mask"]] for i in train_idx
            if prepared[i]["np_mask"].sum() > 0
        ]
        if not parts:
            return torch.ones(N_TYPES, dtype=torch.float32)
        labs = np.concatenate(parts)
        pos = labs.sum(axis=0)
        neg = (1.0 - labs).sum(axis=0)
        pw = np.clip(neg / np.maximum(pos, 1.0), 1.0, 50.0)
        return torch.tensor(pw, dtype=torch.float32)
    parts = [
        prepared[i]["np_contact_any"][prepared[i]["np_mask"]] for i in train_idx
        if prepared[i]["np_mask"].sum() > 0
    ]
    if not parts:
        return torch.ones(1, dtype=torch.float32)
    labs = np.concatenate(parts)
    pos = float(labs.sum())
    neg = float(len(labs) - pos)
    return torch.tensor([min(max(neg / max(pos, 1.0), 1.0), 50.0)], dtype=torch.float32)


def train_epoch(model, prepared, train_idx, opt, typed=True, use_pocket=True,
                lambda_swap=0.0, batch_size=8, rng=None, pos_weight=None):
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
            sup_mask = t["mask"] if typed else torch.ones_like(t["mask"])
            logits, pair_vec = model(
                t["lig_feats"], t["lig_xyz"], t["edge_index"], t["edge_attr"],
                t["pocket_feats"], t["pocket_xyz"], sup_mask, use_pocket,
            )
            labels = _labels_for(t, typed)
            loss = loss + masked_bce(logits, labels, sup_mask, pos_weight)

            if typed and lambda_swap > 0 and len(train_idx) > 1:
                j = rng.choice([k for k in train_idx if k != i])
                tj = prepared[j]
                shift = t["pocket_xyz"].mean(0) - tj["pocket_xyz"].mean(0)
                logits_s, _ = model(
                    t["lig_feats"], t["lig_xyz"], t["edge_index"], t["edge_attr"],
                    tj["pocket_feats"], tj["pocket_xyz"] + shift, t["mask"], use_pocket,
                )
                if t["mask"].sum() > 0:
                    tv = labels[t["mask"]].mean(0)
                    pt = torch.sigmoid(logits[t["mask"]]).mean(0)
                    ps = torch.sigmoid(logits_s[t["mask"]]).mean(0)
                    loss = loss + lambda_swap * swap_margin(pt, ps, tv)
        loss = loss / max(1, len(batch))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += float(loss.detach())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def predict(model, prepared, idx, use_pocket=True, typed=True):
    model.eval()
    logits, labels, mask, contact_any, pair_vecs, capability, sample_index = (
        [], [], [], [], [], [], []
    )
    for i in idx:
        t = prepared[i]
        sup_mask = t["mask"] if typed else torch.ones_like(t["mask"])
        out = model(
            t["lig_feats"], t["lig_xyz"], t["edge_index"], t["edge_attr"],
            t["pocket_feats"], t["pocket_xyz"], sup_mask, use_pocket,
        )
        lg, pv = out
        logits.append(lg.cpu().numpy())
        pair_vecs.append(pv.cpu().numpy())
        labels.append(t["np_labels"])
        mask.append(t["np_mask"] if typed else np.ones_like(t["np_mask"]))
        contact_any.append(t["np_contact_any"])
        sample_index.append(np.full(len(t["np_labels"]), i, dtype=np.int64))
        if t["np_capability"] is not None:
            capability.append(t["np_capability"])
    return {
        "logits": np.concatenate(logits, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "mask": np.concatenate(mask, axis=0),
        "contact_any": np.concatenate(contact_any, axis=0),
        "capability": np.concatenate(capability, axis=0) if capability else None,
        "pair_vecs": np.asarray(pair_vecs),
        "sample_index": np.concatenate(sample_index, axis=0),
    }


def evaluate(model, prepared, idx, typed=True, use_pocket=True):
    p = predict(model, prepared, idx, use_pocket=use_pocket, typed=typed)
    if typed:
        return typed_metrics(p["logits"], p["labels"], p["mask"], p["capability"])
    return untyped_metrics(p["logits"], p["contact_any"], p["mask"])


def run_fold(
    samples,
    train_idx,
    test_idx,
    typed=True,
    use_pocket=True,
    lambda_swap=0.0,
    epochs=20,
    lr=1e-3,
    batch_size=8,
    dim=64,
    device=None,
    seed=SEED,
    verbose=False,
    prepared=None,
    return_preds=False,
):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if prepared is None:
        prepared = prepare(samples, device)
    lig_dim = samples[0]["lig_feats"].shape[1]
    pocket_dim = samples[0]["pocket_feats"].shape[1]
    model = TACTModel(
        lig_dim=lig_dim, pocket_dim=pocket_dim, dim=dim,
        n_out=(N_TYPES if typed else 1), use_adversary=False,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)
    pos_weight = compute_pos_weight(prepared, list(train_idx), typed)
    for ep in range(epochs):
        loss = train_epoch(
            model, prepared, list(train_idx), opt, typed=typed,
            use_pocket=use_pocket, lambda_swap=lambda_swap,
            batch_size=batch_size, rng=rng, pos_weight=pos_weight,
        )
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  epoch {ep + 1}/{epochs} loss={loss:.4f}", flush=True)
    preds = predict(model, prepared, list(test_idx), use_pocket=use_pocket, typed=typed)
    if typed:
        metrics = typed_metrics(preds["logits"], preds["labels"], preds["mask"],
                                preds["capability"])
    else:
        metrics = untyped_metrics(preds["logits"], preds["contact_any"], preds["mask"])
    metrics.update(leakage_summary(samples, train_idx, test_idx))
    if return_preds:
        return model, metrics, preds
    return model, metrics
