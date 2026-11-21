"""Build the typed-contact dataset from the SV-Hop crystal benchmark."""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from .dataset import build_dataset, save_dataset
from .splits import leakage_summary, target_folds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--aware", action="store_true")
    ap.add_argument("--cutoff-shift", type=float, default=0.0)
    ap.add_argument("--include-water", action="store_true")
    args = ap.parse_args(argv)

    samples, reasons = build_dataset(
        args.csv, args.pdb_dir, limit=args.limit, targets=args.targets,
        aware=args.aware, cutoff_shift=args.cutoff_shift,
        include_water=args.include_water
    )
    save_dataset(samples, args.out)

    targets = Counter(s["target"] for s in samples)
    type_counts = Counter()
    names = ("hbond", "hydrophobic", "pi", "salt", "halogen", "water")
    for s in samples:
        for k in range(s["labels"].shape[1]):
            type_counts[names[k]] += int(s["labels"][:, k].sum())
    payload = {
        "n_samples": len(samples),
        "n_targets": len(targets),
        "target_counts": dict(targets),
        "typed_contact_counts": dict(type_counts),
        "reasons": dict(reasons),
        "aware": args.aware,
        "include_water": args.include_water,
    }
    first = target_folds(samples, n_splits=5)[0] if len(samples) >= 4 else None
    if first is not None:
        payload["example_fold_leakage"] = leakage_summary(samples, first[0], first[1])
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()
