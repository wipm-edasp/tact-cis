"""CIS experiment: split-conformal interaction sets on target-held-out folds.

For each fold: split training targets into fit/calibration (family-stratified),
train COD on fit, calibrate thresholds on cal, evaluate coverage/size on test.
Methods: global single threshold, per-type (marginal), per-(type, family)
Mondrian, and top-1/2/3 sets.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

from .cod import (CODModel, compute_pos_weight_cod, predict_cod, prepare_cod,
                  train_cod_epoch)
from .conformal import (evaluate_sets, family_of, fit_thresholds, predict_sets,
                        topk_sets)
from .config import SEED
from .dataset import load_dataset
from .splits import scaffold_folds, target_folds
from .train import set_seed

METHODS = ("global_single", "type", "family", "temp", "ensemble", "geo_add", "geo_partner", "crc_global", "crc_type", "crc_geo_add", "crc_geo_partner", "geo_l025", "geo_l100", "geo_s05", "crc_geo_l025", "crc_geo_l100", "crc_geo_s05", "top1", "top2", "top3")
ALPHAS = (0.1, 0.2)


def split_fit_cal(samples, tr, frac=0.2, seed=0):
    rng = random.Random(seed)
    by = defaultdict(list)
    for i in tr:
        by[family_of(samples[i]["target"])].append(int(i))
    cal = []
    for fam, idxs in by.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        k = max(1, int(round(frac * len(idxs)))) if len(idxs) >= 2 else 0
        cal += idxs[:k]
    calset = set(cal)
    fit = [int(i) for i in tr if int(i) not in calset]
    if not cal:
        cal = [int(tr[0])]
        fit = [int(i) for i in tr if int(i) != cal[0]]
    return fit, cal


def train_model(samples, prepared, fit_idx, device, epochs, seed, backbone="mlp",
                use_cap_gate=True, n_types=None):
    lig_dim = samples[0]["lig_feats"].shape[1]
    pocket_dim = samples[0]["pocket_feats"].shape[1]
    if n_types is None:
        n_types = samples[0]["labels"].shape[1]
    model = CODModel(lig_dim=lig_dim, pocket_dim=pocket_dim, dim=64,
                     use_decomp=True, backbone=backbone,
                     use_cap_gate=use_cap_gate, n_types=n_types).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pw = compute_pos_weight_cod(prepared, fit_idx, use_cap_gate=use_cap_gate)
    rng = random.Random(seed)
    for _ in range(epochs):
        train_cod_epoch(model, prepared, fit_idx, opt, batch_size=8,
                        lambda_perturb=0.1, pos_weight=pw, rng=rng,
                        use_cap_gate=use_cap_gate)
    return model


def atom_groups(preds, samples):
    return np.asarray([family_of(samples[int(i)]["target"])
                       for i in preds["sample_index"]], dtype=object)


def run_shard(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_dataset(args.data)
    prepared = prepare_cod(samples, device)
    folds = target_folds(samples, args.folds) if args.split == "target" else scaffold_folds(samples, args.folds)
    combos = []
    for seed in args.seeds:
        for fi, (tr, te) in enumerate(folds):
            combos.append((seed, fi, tr, te))
    mine = [c for k, c in enumerate(combos) if k % args.nshards == args.shard]
    print(f"device={device} n={len(samples)} combos={len(mine)}", flush=True)

    records = []
    no_mask = getattr(args, "no_mask", False)
    for seed, fi, tr, te in mine:
        set_seed(seed)
        fit, cal = split_fit_cal(samples, tr, frac=0.2, seed=seed)
        print(f"[seed={seed} fold={fi}] fit={len(fit)} cal={len(cal)} "
              f"test={len(te)}", flush=True)
        model = train_model(samples, prepared, fit, device, args.epochs, seed,
                            backbone=args.backbone, use_cap_gate=not no_mask)
        pc = predict_cod(model, prepared, cal)
        pt = predict_cod(model, prepared, te)
        g_cal = atom_groups(pc, samples)
        g_test = atom_groups(pt, samples)
        t_test = np.asarray([samples[int(i)]["target"] for i in pt["sample_index"]], dtype=object)
        probs_c = 1.0 / (1.0 + np.exp(-pc["logits"]))
        probs_t = 1.0 / (1.0 + np.exp(-pt["logits"]))
        cap_c = np.ones_like(pc["capability"]) if no_mask else pc["capability"]
        cap_t = np.ones_like(pt["capability"]) if no_mask else pt["capability"]
        for alpha in ALPHAS:
            q = fit_thresholds(probs_c, pc["labels"], pc["mask"], cap_c,
                               g_cal, alpha)
            for method in ("global_single", "type", "family"):
                sets = predict_sets(probs_t, cap_t, q, g_test, method)
                m = evaluate_sets(sets, pt["labels"], pt["mask"], pt["capability"], g_test, targets=t_test)
                records.append({"seed": int(seed), "fold": int(fi), "alpha": alpha,
                                "method": method, **m})
            for k in (1, 2, 3):
                sets = topk_sets(probs_t, cap_t, k)
                m = evaluate_sets(sets, pt["labels"], pt["mask"], pt["capability"], g_test, targets=t_test)
                records.append({"seed": int(seed), "fold": int(fi), "alpha": alpha,
                                "method": f"top{k}", **m})
        print("  done", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"shard_{args.shard}.json")
    with open(out, "w") as fh:
        json.dump(records, fh)
    print("saved", out, len(records), flush=True)


def combine(args):
    files = sorted(os.path.join(args.outdir, f) for f in os.listdir(args.outdir)
                   if f.startswith("shard_") and f.endswith(".json"))
    records = []
    for f in files:
        records.extend(json.load(open(f)))
    summary = {}
    for alpha in ALPHAS:
        summary[str(alpha)] = {}
        for method in METHODS:
            rs = [r for r in records if r["alpha"] == alpha and r["method"] == method]
            if not rs:
                continue
            cov = np.asarray([r["coverage"] for r in rs])
            size = np.asarray([r["size"] for r in rs])
            imp = np.asarray([r.get("impossible_rate", 0.0) for r in rs])
            fam = defaultdict(list)
            for r in rs:
                for g, v in r["per_family"].items():
                    fam[g].append(v["coverage"])
            types = defaultdict(list)
            for r in rs:
                for k, v in r.get("per_type", {}).items():
                    types[k].append(v["coverage"])
            counts = defaultdict(lambda: [0, 0])
            for r in rs:
                for t, v in r.get("per_target", {}).items():
                    counts[t][0] += v["covered"]
                    counts[t][1] += v["n_true"]
            cluster_ci = None
            if len(counts) >= 5:
                rng = np.random.default_rng(0)
                keys = list(counts)
                boot = []
                for _ in range(1000):
                    sample = rng.choice(keys, len(keys), replace=True)
                    c = sum(counts[t][0] for t in sample)
                    n = sum(counts[t][1] for t in sample)
                    if n > 0:
                        boot.append(c / n)
                cluster_ci = [float(np.percentile(boot, 2.5)),
                              float(np.percentile(boot, 97.5))]
            summary[str(alpha)][method] = {
                "n_runs": len(rs),
                "coverage_mean": float(cov.mean()),
                "coverage_ci": [float(cov.mean() - 1.96 * cov.std(ddof=1) / np.sqrt(len(rs))),
                                float(cov.mean() + 1.96 * cov.std(ddof=1) / np.sqrt(len(rs)))],
                "size_mean": float(size.mean()),
                "impossible_rate_mean": float(imp.mean()),
                "coverage_gap": float(abs(cov.mean() - (1 - alpha))),
                "per_family": {g: float(np.mean(v)) for g, v in fam.items()},
                "per_type_coverage": {k: float(np.mean(v)) for k, v in types.items()},
                "cluster_coverage_ci": cluster_ci,
            }
    model_records = [r for r in records
                     if r.get("method") == "model" and "model" in r]
    if model_records:
        keys = sorted({k for r in model_records for k in r["model"]})
        model_summary = {"n_runs": len(model_records)}
        for k in keys:
            vals = [float(r["model"][k]) for r in model_records
                    if isinstance(r["model"].get(k), (int, float))
                    and np.isfinite(r["model"][k])]
            if vals:
                model_summary[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
        summary["model"] = model_summary
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "combine"])
    ap.add_argument("--data", default="results/dataset_cod.pkl")
    ap.add_argument("--outdir", default="results/cis")
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--backbone", default="mlp", choices=["mlp", "egnn"])
    ap.add_argument("--split", default="target", choices=["target", "scaffold"])
    ap.add_argument("--no-mask", action="store_true")
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
