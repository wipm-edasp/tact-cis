"""Sharded COD experiment + combiner.

Arms:
  cod             decomp + opportunity perturbation
  cod_no_perturb  decomp, no perturbation (ablation)
  no_decomp       standard cross-attention head with capability gate (ablation)
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from .cod import prepare_cod, run_cod_fold
from .config import SEED
from .dataset import build_dataset, load_dataset, save_dataset
from .experiment import subset_metrics
from .splits import nearest_train_tanimoto, scaffold_folds, target_folds
from .train import set_seed

ARMS = {
    "cod": dict(use_decomp=True, lambda_perturb=0.1, ablate="none", backbone="mlp"),
    "cod_no_perturb": dict(use_decomp=True, lambda_perturb=0.0, ablate="none", backbone="mlp"),
    "no_decomp": dict(use_decomp=False, lambda_perturb=0.0, ablate="none", backbone="mlp"),
    "cod_no_pocket": dict(use_decomp=True, lambda_perturb=0.0, ablate="no_pocket", backbone="mlp"),
    "cod_no_geometry": dict(use_decomp=True, lambda_perturb=0.1, ablate="no_geometry", backbone="mlp"),
    "egnn": dict(use_decomp=True, lambda_perturb=0.1, ablate="none", backbone="egnn"),
    "egnn_no_perturb": dict(use_decomp=True, lambda_perturb=0.0, ablate="none", backbone="egnn"),
}
METRIC_KEYS = ("macro_auc_cap", "macro_ap_cap", "macro_auc", "auc_contact_any")


def load_or_build(path, csv, pdb_dir):
    if os.path.exists(path):
        return load_dataset(path)
    samples, reasons = build_dataset(csv, pdb_dir, aware=False)
    save_dataset(samples, path)
    print(f"built {path}: {len(samples)} {dict(reasons)}", flush=True)
    return samples


def run_shard(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_or_build(args.data, args.csv, args.pdb_dir)
    print(f"device={device} n={len(samples)}", flush=True)
    combos = []
    for split in args.splits:
        folds = target_folds(samples, args.folds) if split == "target" else scaffold_folds(samples, args.folds)
        for seed in args.seeds:
            for fi, (tr, te) in enumerate(folds):
                combos.append((split, seed, fi, tr, te))
    mine = [c for k, c in enumerate(combos) if k % args.nshards == args.shard]
    prepared = {}
    records = []
    for split, seed, fi, tr, te in mine:
        nn = nearest_train_tanimoto(samples, tr, te)
        matched = [int(te[j]) for j in range(len(te)) if nn[j] < 0.6]
        print(f"[{split} seed={seed} fold={fi}] train={len(tr)} test={len(te)}", flush=True)
        for arm in args.arms:
            cfg = ARMS[arm]
            key = device
            if key not in prepared:
                prepared[key] = prepare_cod(samples, device)
            _, metrics, preds = run_cod_fold(
                samples, tr, te, use_decomp=cfg["use_decomp"],
                lambda_perturb=cfg["lambda_perturb"], epochs=args.epochs,
                lr=args.lr, batch_size=args.batch_size, dim=args.dim,
                device=device, seed=seed, prepared=prepared[key],
                return_preds=True, ablate=cfg["ablate"],
                backbone=cfg["backbone"],
            )
            rec = {
                "split": split, "seed": int(seed), "fold": int(fi), "arm": arm,
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                "matched": subset_metrics(preds, matched, True),
            }
            if arm == "cod":
                per_target = {}
                for t in sorted({samples[i]["target"] for i in te}):
                    ids = [int(i) for i in te if samples[i]["target"] == t]
                    if len(ids) >= 2:
                        m = subset_metrics(preds, ids, True)
                        per_target[t] = {"n": len(ids),
                                         "macro_auc_cap": m.get("macro_auc_cap")}
                rec["per_target"] = per_target
            records.append(rec)
            cap = metrics.get("macro_auc_cap", float("nan"))
            print(f"  {arm}: capAUC={cap:.3f}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"shard_{args.shard}.json")
    with open(out, "w") as fh:
        json.dump(records, fh)
    print(f"saved {out} ({len(records)})", flush=True)


def _ci(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if len(vals) == 0:
        return None, None, None
    if len(vals) == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    m = float(vals.mean())
    half = 1.96 * float(vals.std(ddof=1)) / np.sqrt(len(vals))
    return m, m - half, m + half


def combine(args):
    files = sorted(os.path.join(args.outdir, f) for f in os.listdir(args.outdir)
                   if f.startswith("shard_") and f.endswith(".json"))
    records = []
    for f in files:
        records.extend(json.load(open(f)))
    rows = []
    for r in records:
        row = {"split": r["split"], "seed": r["seed"], "fold": r["fold"], "arm": r["arm"]}
        for k, v in r["metrics"].items():
            row[f"m_{k}"] = v
        for k, v in r["matched"].items():
            if isinstance(v, (int, float)):
                row[f"mm_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "per_combo.csv"), index=False)
    summary = {}
    for split in df.split.unique():
        summary[split] = {}
        for arm in df.arm.unique():
            d = df[(df.split == split) & (df.arm == arm)]
            e = {"n_runs": int(len(d))}
            for k in METRIC_KEYS:
                for pre in ("m", "mm"):
                    col = f"{pre}_{k}"
                    if col in d.columns:
                        mean, lo, hi = _ci(d[col].values)
                        if mean is not None:
                            e[f"{'matched_' if pre=='mm' else ''}{k}"] = {
                                "mean": mean, "ci_lo": lo, "ci_hi": hi}
            summary[split][arm] = e

    def paired(a, b, key, split):
        da = df[(df.split == split) & (df.arm == a)].set_index(["seed", "fold"])
        db = df[(df.split == split) & (df.arm == b)].set_index(["seed", "fold"])
        col = f"m_{key}"
        if col not in da.columns or col not in db.columns:
            return None
        idx = da.index.intersection(db.index)
        deltas = (da.loc[idx, col] - db.loc[idx, col]).values
        mean, lo, hi = _ci(deltas)
        return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_pairs": int(len(idx))}

    comps = {}
    for split in df.split.unique():
        comps[split] = {
            "cod_minus_no_perturb": paired("cod", "cod_no_perturb", "macro_auc_cap", split),
            "cod_minus_no_decomp": paired("cod", "no_decomp", "macro_auc_cap", split),
        }
    out = {"summary": summary, "comparisons": comps}
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "combine"])
    ap.add_argument("--data", default="results/dataset.pkl")
    ap.add_argument("--csv", default="/home/xiao/workspace/drug-design/data/pdb_hops/ligands.csv")
    ap.add_argument("--pdb-dir", default="/home/xiao/workspace/drug-design/data/pdb_hops/pdb")
    ap.add_argument("--outdir", default="results/cod")
    ap.add_argument("--splits", nargs="+", default=["target"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args(argv)
    set_seed(SEED)
    if args.mode == "run":
        run_shard(args)
    else:
        combine(args)


if __name__ == "__main__":
    main()
