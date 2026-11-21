"""Full target-held-out experiment, shardable across GPUs, plus combiner.

Arms:
  main          typed contacts, residue-blind pocket, pocket-swap loss
  ligand_only   typed contacts, no pocket (shortcut control)
  residue_aware typed contacts, residue-aware pocket (H2 ablation)
  untyped       contact presence only (H1 baseline)

Primary endpoint: capacity-controlled macro AUC (per interaction type, only
ligand atoms chemically capable of that interaction).
Secondary: similarity-matched subset (nearest train Tanimoto < 0.6).
H3: per-target swap sensitivity vs per-target transfer.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .config import INTERACTION_TYPES, SEED
from .dataset import build_dataset, load_dataset, save_dataset
from .metrics import typed_metrics, untyped_metrics
from .splits import nearest_train_tanimoto, scaffold_folds, target_folds
from .train import prepare, run_fold, set_seed

ARMS = {
    "main": dict(typed=True, use_pocket=True, lambda_swap=0.2, aware=False),
    "pocket_no_swap": dict(typed=True, use_pocket=True, lambda_swap=0.0, aware=False),
    "ligand_only": dict(typed=True, use_pocket=False, lambda_swap=0.0, aware=False),
    "residue_aware": dict(typed=True, use_pocket=True, lambda_swap=0.2, aware=True),
    "untyped": dict(typed=False, use_pocket=True, lambda_swap=0.0, aware=False),
}

METRIC_KEYS = (
    "macro_auc_cap", "macro_ap_cap", "macro_auc", "macro_ap",
    "auc_contact_any", "auc_contact", "ap_contact",
)


def load_or_build(path, csv, pdb_dir, aware):
    if os.path.exists(path):
        return load_dataset(path)
    samples, reasons = build_dataset(csv, pdb_dir, aware=aware)
    save_dataset(samples, path)
    print(f"built {path}: {len(samples)} samples, {dict(reasons)}", flush=True)
    return samples


def subset_metrics(preds, keep_ids, typed):
    keep = np.asarray(sorted(set(int(i) for i in keep_ids)), dtype=np.int64)
    atom_keep = np.isin(preds["sample_index"], keep)
    if atom_keep.sum() == 0:
        return {}
    cap = preds["capability"][atom_keep] if preds["capability"] is not None else None
    if typed:
        return typed_metrics(preds["logits"][atom_keep], preds["labels"][atom_keep],
                             preds["mask"][atom_keep], cap)
    return untyped_metrics(preds["logits"][atom_keep], preds["contact_any"][atom_keep],
                           preds["mask"][atom_keep])


@torch.no_grad()
def swap_sensitivity(model, prepared, test_idx, samples, n_swap=3, seed=0):
    model.eval()
    rng = random.Random(seed)
    by_target = defaultdict(list)
    for i in test_idx:
        by_target[samples[i]["target"]].append(i)
    rows = []
    for i in test_idx:
        ti = prepared[i]
        if ti["mask"].sum() == 0:
            continue
        logits, _ = model(
            ti["lig_feats"], ti["lig_xyz"], ti["edge_index"], ti["edge_attr"],
            ti["pocket_feats"], ti["pocket_xyz"], ti["mask"], True,
        )
        p_true = torch.sigmoid(logits[ti["mask"]]).mean(0)
        other = [t for t in by_target if t != samples[i]["target"]]
        if not other:
            continue
        vals = []
        for _ in range(n_swap):
            t = rng.choice(other)
            j = rng.choice(by_target[t])
            tj = prepared[j]
            shift = ti["pocket_xyz"].mean(0) - tj["pocket_xyz"].mean(0)
            ls, _ = model(
                ti["lig_feats"], ti["lig_xyz"], ti["edge_index"], ti["edge_attr"],
                tj["pocket_feats"], tj["pocket_xyz"] + shift, ti["mask"], True,
            )
            p_swap = torch.sigmoid(ls[ti["mask"]]).mean(0)
            cos = F.cosine_similarity(p_true.unsqueeze(0), p_swap.unsqueeze(0)).item()
            vals.append(1.0 - cos)
        rows.append((samples[i]["target"], float(np.mean(vals))))
    per_target = defaultdict(list)
    for t, v in rows:
        per_target[t].append(v)
    return {t: float(np.mean(v)) for t, v in per_target.items()}


def run_shard(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} shard={args.shard}/{args.nshards}", flush=True)
    blind = load_or_build(args.data, args.csv, args.pdb_dir, aware=False)
    aware = None
    if "residue_aware" in args.arms:
        aware = load_or_build(args.data_aware, args.csv, args.pdb_dir, aware=True)
        if len(aware) != len(blind):
            raise RuntimeError("aware/blind dataset length mismatch")

    splits = args.splits
    combos = []
    for split in splits:
        if split == "target":
            folds = target_folds(blind, n_splits=args.folds)
        else:
            folds = scaffold_folds(blind, n_splits=args.folds)
        for seed in args.seeds:
            for fi, (tr, te) in enumerate(folds):
                combos.append((split, seed, fi, tr, te))
    mine = [c for k, c in enumerate(combos) if k % args.nshards == args.shard]
    print(f"combos total={len(combos)} mine={len(mine)}", flush=True)

    prepared = {}
    records = []
    for split, seed, fi, tr, te in mine:
        nn = nearest_train_tanimoto(blind, tr, te)
        matched = [int(te[j]) for j in range(len(te)) if nn[j] < 0.6]
        print(f"[{split} seed={seed} fold={fi}] train={len(tr)} test={len(te)} "
              f"matched={len(matched)}", flush=True)
        for arm in args.arms:
            cfg = ARMS[arm]
            samples = aware if cfg["aware"] else blind
            key = (id(samples), device)
            if key not in prepared:
                prepared[key] = prepare(samples, device)
            model, metrics, preds = run_fold(
                samples, tr, te, typed=cfg["typed"], use_pocket=cfg["use_pocket"],
                lambda_swap=cfg["lambda_swap"], epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, dim=args.dim, device=device,
                seed=seed, prepared=prepared[key], return_preds=True,
            )
            rec = {
                "split": split, "seed": int(seed), "fold": int(fi), "arm": arm,
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "matched_ids": matched,
                "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                "matched": subset_metrics(preds, matched, cfg["typed"]),
            }
            if arm == "main":
                per_target = {}
                for t in sorted({samples[i]["target"] for i in te}):
                    ids = [int(i) for i in te if samples[i]["target"] == t]
                    if len(ids) < 2:
                        continue
                    m = subset_metrics(preds, ids, True)
                    per_target[t] = {
                        "n": len(ids),
                        "macro_auc_cap": m.get("macro_auc_cap"),
                        "auc_cap_contact_any": m.get("auc_cap_contact_any"),
                        "auc_contact_any": m.get("auc_contact_any"),
                    }
                rec["per_target"] = per_target
                rec["swap"] = swap_sensitivity(
                    model, prepared[key], [int(i) for i in te], samples, seed=seed
                )
            records.append(rec)
            cap = metrics.get("macro_auc_cap", float("nan"))
            allm = metrics.get("macro_auc", metrics.get("auc_contact", float("nan")))
            print(f"  {arm}: capAUC={cap:.3f} allAUC={allm:.3f}", flush=True)

    out = os.path.join(args.outdir, f"shard_{args.shard}.json")
    os.makedirs(args.outdir, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(records, fh)
    print(f"saved {out} ({len(records)} records)", flush=True)


def _ci(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return None, None, None
    if len(vals) == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1))
    half = 1.96 * sd / np.sqrt(len(vals))
    return mean, mean - half, mean + half


def combine(args):
    files = sorted(
        os.path.join(args.outdir, f) for f in os.listdir(args.outdir)
        if f.startswith("shard_") and f.endswith(".json")
    )
    records = []
    for f in files:
        with open(f) as fh:
            records.extend(json.load(fh))
    if not records:
        raise SystemExit("no shard records")
    print(f"combined {len(records)} records from {len(files)} shards")

    rows = []
    for r in records:
        row = {"split": r["split"], "seed": r["seed"], "fold": r["fold"],
               "arm": r["arm"], "n_test": r["n_test"]}
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
            entry = {"n_runs": int(len(d))}
            for k in METRIC_KEYS:
                col = f"m_{k}"
                if col in d.columns:
                    mean, lo, hi = _ci(d[col].values)
                    if mean is not None:
                        entry[k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
                mcol = f"mm_{k}"
                if mcol in d.columns:
                    mean, lo, hi = _ci(d[mcol].values)
                    if mean is not None:
                        entry[f"matched_{k}"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
            summary[split][arm] = entry

    # paired comparisons (same split/seed/fold)
    def paired(a, b, key, split, matched=False, key_b=None):
        da = df[(df.split == split) & (df.arm == a)].set_index(["seed", "fold"])
        db = df[(df.split == split) & (df.arm == b)].set_index(["seed", "fold"])
        pre = "mm" if matched else "m"
        cola = f"{pre}_{key}"
        colb = f"{pre}_{key_b or key}"
        if cola not in da.columns or colb not in db.columns:
            return None
        idx = da.index.intersection(db.index)
        deltas = (da.loc[idx, cola] - db.loc[idx, colb]).values
        mean, lo, hi = _ci(deltas)
        return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_pairs": int(len(idx))}

    comparisons = {}
    for split in df.split.unique():
        comparisons[split] = {
            "H1_main_minus_untyped_contact_auc": paired(
                "main", "untyped", "auc_contact_any", split, key_b="auc_contact"),
            "H2_main_minus_aware_cap_auc": paired("main", "residue_aware", "macro_auc_cap", split),
            "H2_matched_main_minus_aware": paired("main", "residue_aware", "macro_auc_cap", split, matched=True),
            "main_minus_ligand_only_cap_auc": paired("main", "ligand_only", "macro_auc_cap", split),
            "matched_main_minus_ligand_only": paired("main", "ligand_only", "macro_auc_cap", split, matched=True),
        }

    # H3: per-target swap sensitivity vs per-target transfer
    h3 = {}
    for split in df.split.unique():
        main_recs = [r for r in records if r["split"] == split and r["arm"] == "main"]
        per_t = defaultdict(lambda: {"sens": [], "auc": []})
        for r in main_recs:
            for t, v in r.get("swap", {}).items():
                per_t[t]["sens"].append(v)
            for t, m in r.get("per_target", {}).items():
                if m.get("macro_auc_cap") is not None and np.isfinite(m["macro_auc_cap"]):
                    per_t[t]["auc"].append(m["macro_auc_cap"])
        pt = []
        for t, v in per_t.items():
            if v["sens"] and v["auc"]:
                pt.append({"target": t, "swap_sensitivity": float(np.mean(v["sens"])),
                           "macro_auc_cap": float(np.mean(v["auc"])),
                           "n_eval": len(v["auc"])})
        pt_df = pd.DataFrame(pt)
        h3[split] = {"n_targets": int(len(pt_df))}
        if len(pt_df) >= 5:
            rho, p = spearmanr(pt_df.swap_sensitivity, pt_df.macro_auc_cap)
            h3[split].update({"spearman_rho": float(rho), "p_value": float(p)})
        pt_df.to_csv(os.path.join(args.outdir, f"per_target_{split}.csv"), index=False)

    out = {"summary": summary, "comparisons": comparisons, "h3": h3,
           "config": {"arms": list(df.arm.unique()), "splits": list(df.split.unique()),
                      "n_records": len(records)}}
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "combine"])
    ap.add_argument("--data", default="results/dataset.pkl")
    ap.add_argument("--data-aware", default="results/dataset_aware.pkl")
    ap.add_argument("--csv", default="/home/xiao/workspace/drug-design/data/pdb_hops/ligands.csv")
    ap.add_argument("--pdb-dir", default="/home/xiao/workspace/drug-design/data/pdb_hops/pdb")
    ap.add_argument("--outdir", default="results/full")
    ap.add_argument("--splits", nargs="+", default=["target", "scaffold"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=16)
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
