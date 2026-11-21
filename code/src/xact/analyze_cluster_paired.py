"""Paired target-cluster bootstrap for CIS method comparisons.

Reads shard JSON files that contain per-target size (and covered/n_true) and
computes paired method differences with the target as the resampling unit.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

PAIRS = (
    ("geo_partner", "type"),
    ("crc_geo_partner", "top1"),
    ("crc_geo_partner", "crc_type"),
    ("crc_geo_partner", "crc_global"),
    ("type", "top1"),
)


def load_records(outdir):
    records = []
    for path in sorted(glob.glob(os.path.join(outdir, "shard_*.json"))):
        with open(path) as fh:
            records.extend(json.load(fh))
    return records


def _value(rec, target, metric):
    v = rec.get("per_target", {}).get(target)
    if v is None:
        return None
    if metric == "size":
        return v.get("size")
    if metric == "coverage":
        return v["covered"] / v["n_true"] if v["n_true"] else None
    raise ValueError(metric)


def paired_cluster(records, alpha, method_a, method_b, metric, n_boot=2000,
                   seed=0):
    by = {(r["seed"], r["fold"], r["method"]): r for r in records
          if abs(r["alpha"] - alpha) < 1e-9}
    target_num = defaultdict(float)
    target_den = defaultdict(float)
    for (s, f, method), ra in by.items():
        if method != method_a:
            continue
        rb = by.get((s, f, method_b))
        if rb is None:
            continue
        for t in ra.get("per_target", {}):
            va = ra["per_target"][t]
            vb = rb.get("per_target", {}).get(t)
            if vb is None:
                continue
            if metric == "size":
                w = va.get("n_atoms")
                if w is None or va.get("size") is None or vb.get("size") is None:
                    continue
                target_num[t] += w * (vb["size"] - va["size"])
                target_den[t] += w
            elif metric == "coverage":
                if not va["n_true"]:
                    continue
                target_num[t] += vb["covered"] - va["covered"]
                target_den[t] += va["n_true"]
            else:
                raise ValueError(metric)
    keys = sorted(t for t in target_num if target_den[t] > 0)
    if not keys:
        return None
    obs = float(sum(target_num[t] for t in keys) / sum(target_den[t] for t in keys))
    target_mean = {t: target_num[t] / target_den[t] for t in keys}
    equal_mean = float(np.mean([target_mean[t] for t in keys]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(keys, len(keys), replace=True)
        num = sum(target_num[t] for t in sample)
        den = sum(target_den[t] for t in sample)
        boot.append(float(num / den))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    run_means = []
    for (s, f, method), ra in by.items():
        if method != method_a:
            continue
        rb = by.get((s, f, method_b))
        if rb is None:
            continue
        diffs = []
        for t in ra.get("per_target", {}):
            va = ra["per_target"][t]
            vb = rb.get("per_target", {}).get(t)
            if vb is None:
                continue
            if metric == "size" and va.get("size") is not None and vb.get("size") is not None:
                diffs.append(vb["size"] - va["size"])
            elif metric == "coverage" and va["n_true"]:
                diffs.append(vb["covered"] / vb["n_true"]
                             - va["covered"] / va["n_true"])
        if diffs:
            run_means.append(float(np.mean(diffs)))
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "n_targets": len(keys),
        "n_runs": len(run_means),
        "mean": obs,
        "ci95": [float(lo), float(hi)],
        "target_equal_mean": equal_mean,
        "run_level_mean": float(np.mean(run_means)),
        "run_level_ci95": [
            float(np.mean(run_means) - 1.96 * np.std(run_means, ddof=1)
                  / np.sqrt(len(run_means))),
            float(np.mean(run_means) + 1.96 * np.std(run_means, ddof=1)
                  / np.sqrt(len(run_means))),
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    records = load_records(args.outdir)
    result = {}
    for alpha in (0.1, 0.2):
        result[str(alpha)] = {}
        for a, b in PAIRS:
            for metric in ("size", "coverage"):
                key = f"{b}-{a}:{metric}"
                res = paired_cluster(records, alpha, a, b, metric,
                                     n_boot=args.n_boot)
                if res is not None:
                    result[str(alpha)][key] = res
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    for alpha in ("0.1", "0.2"):
        for key, r in result[alpha].items():
            print(f"{alpha} {key:45s} {r['mean']:+.4f} "
                  f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
                  f"targets={r['n_targets']} runs={r['n_runs']}")


if __name__ == "__main__":
    main()
