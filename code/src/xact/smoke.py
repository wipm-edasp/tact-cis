"""End-to-end smoke test: tiny target-held-out run of all three arms."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch

from .dataset import build_sample
from .splits import assert_disjoint, target_folds
from .train import run_fold, set_seed


def build_subset(csv_path, pdb_dir, per_target, max_targets):
    df = pd.read_csv(csv_path)
    targets = sorted(df.target.unique())[:max_targets]
    rows = pd.concat([df[df.target == t].head(per_target) for t in targets])
    samples, reasons = [], Counter()
    for r in rows.itertuples():
        path = os.path.join(pdb_dir, f"{r.pdb}.pdb")
        if not os.path.exists(path):
            reasons["missing_pdb"] += 1
            continue
        try:
            s = build_sample(path, r.target, r.pdb, r.ccd, r.smiles, aware=False)
        except Exception as exc:  # noqa: BLE001
            reasons[f"error:{type(exc).__name__}"] += 1
            continue
        if s is None:
            reasons["no_ligand_or_bonds"] += 1
            continue
        if s["labels"].sum() == 0:
            reasons["no_typed_contact"] += 1
            continue
        samples.append(s)
        reasons["ok"] += 1
    return samples, reasons


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--per-target", type=int, default=4)
    ap.add_argument("--max-targets", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lambda-swap", type=float, default=0.2)
    ap.add_argument("--out", default="results/smoke.json")
    args = ap.parse_args(argv)

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    samples, reasons = build_subset(
        args.csv, args.pdb_dir, args.per_target, args.max_targets
    )
    print("build reasons:", dict(reasons))
    if len(samples) < 8:
        print("NOT ENOUGH SAMPLES")
        return
    n_targets = len({s["target"] for s in samples})
    print(f"samples={len(samples)} targets={n_targets}")
    type_totals = {
        name: int(sum(s["labels"][:, k].sum() for s in samples))
        for k, name in enumerate(("hbond", "hydrophobic", "pi", "salt", "halogen"))
    }
    print("typed contacts:", type_totals)

    folds = target_folds(samples, n_splits=min(3, n_targets))
    train_idx, test_idx = folds[0]
    assert_disjoint(samples, train_idx, test_idx, "target")
    print(f"fold0 train={len(train_idx)} test={len(test_idx)}")

    arms = {
        "typed_pocket": dict(typed=True, use_pocket=True, lambda_swap=args.lambda_swap),
        "typed_ligand_only": dict(typed=True, use_pocket=False, lambda_swap=0.0),
        "untyped_pocket": dict(typed=False, use_pocket=True, lambda_swap=0.0),
    }
    results = {}
    for name, cfg in arms.items():
        print(f"\n== {name} ==")
        _, m = run_fold(
            samples, train_idx, test_idx, epochs=args.epochs, lr=2e-3,
            batch_size=8, dim=64, device=device, verbose=True, **cfg,
        )
        results[name] = m
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in m.items()}, indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"n_samples": len(samples), "n_targets": n_targets,
                   "typed_contacts": type_totals, "arms": results}, fh, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
