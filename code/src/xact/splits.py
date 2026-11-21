"""Evaluation splits and nearest-neighbour leakage audit."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import GroupKFold, KFold

from .config import SEED


def _groups(samples, key):
    return np.asarray([s[key] for s in samples], dtype=object)


def target_folds(samples, n_splits: int = 5):
    groups = _groups(samples, "target")
    n_groups = len(set(groups))
    k = min(n_splits, n_groups)
    return list(GroupKFold(n_splits=k).split(np.arange(len(samples)), groups=groups))


def scaffold_folds(samples, n_splits: int = 5):
    groups = _groups(samples, "scaffold")
    n_groups = len(set(groups))
    k = min(n_splits, max(2, n_groups))
    return list(GroupKFold(n_splits=k).split(np.arange(len(samples)), groups=groups))


def random_folds(samples, n_splits: int = 5, seed: int = SEED):
    k = min(n_splits, len(samples))
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(len(samples))))


def morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None


def nearest_train_tanimoto(samples, train_idx, test_idx) -> np.ndarray:
    """Max Tanimoto of each test ligand to any training ligand."""
    train_fps = [(i, morgan(samples[i]["smiles"])) for i in train_idx]
    train_fps = [(i, f) for i, f in train_fps if f is not None]
    out = np.zeros(len(test_idx), dtype=np.float32)
    for j, ti in enumerate(test_idx):
        fp = morgan(samples[ti]["smiles"])
        if fp is None or not train_fps:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, [f for _, f in train_fps])
        out[j] = float(np.max(sims))
    return out


def leakage_summary(samples, train_idx, test_idx) -> dict:
    sims = nearest_train_tanimoto(samples, train_idx, test_idx)
    return {
        "n_test": int(len(test_idx)),
        "median_nn_tanimoto": float(np.median(sims)) if len(sims) else None,
        "frac_nn_ge_0.6": float((sims >= 0.6).mean()) if len(sims) else None,
        "frac_nn_ge_0.8": float((sims >= 0.8).mean()) if len(sims) else None,
    }


def assert_disjoint(samples, train_idx, test_idx, key):
    tr = {samples[i][key] for i in train_idx}
    te = {samples[i][key] for i in test_idx}
    overlap = tr & te
    if overlap:
        raise AssertionError(f"{key} overlap between train and test: {sorted(overlap)[:5]}")
    return True


def temporal_folds(samples, cutoff, dates_path="results/pdb_release_dates.json"):
    """One external temporal split: release date < cutoff train, >= cutoff test."""
    dates = json.load(open(dates_path))
    tr, te = [], []
    for i, s in enumerate(samples):
        d = dates.get(s["pdb"])
        if d is None:
            continue
        if d < cutoff:
            tr.append(i)
        else:
            te.append(i)
    return [(tr, te)]


def strict_temporal_folds(samples, cutoff,
                          dates_path="results/pdb_release_dates.json"):
    """Target-disjoint and date-strict split.

    Targets are assigned by median release date; only pre-cutoff complexes of
    train targets and post-cutoff complexes of test targets are kept.
    """
    dates = json.load(open(dates_path))
    by = defaultdict(list)
    for s in samples:
        by[s["target"]].append(dates.get(s["pdb"], "9999"))
    train_t, test_t = set(), set()
    for t, ds in by.items():
        ds = sorted(ds)
        med = ds[len(ds) // 2]
        (test_t if med >= cutoff else train_t).add(t)
    tr, te = [], []
    for i, s in enumerate(samples):
        d = dates.get(s["pdb"])
        if d is None:
            continue
        if s["target"] in train_t and d < cutoff:
            tr.append(i)
        elif s["target"] in test_t and d >= cutoff:
            te.append(i)
    return [(tr, te)]
