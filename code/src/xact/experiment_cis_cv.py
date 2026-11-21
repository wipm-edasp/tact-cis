"""CIS with cross-conformal (CV+) calibration.

For each outer target fold:
  - split the training targets into `inner_folds` target-disjoint folds;
  - train a model on the other folds, predict the held-out inner fold;
  - pool all training-target scores (each target scored by a model that did not
    see it) as the calibration set;
  - train a final model on all training targets and score the test targets;
  - fit conformal thresholds on the pooled calibration set.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from .cod import CUTOFFS, predict_cod, prepare_cod
from .conformal import (chemistry_score, crc_threshold, evaluate_sets,
                        family_of, fit_temperature, fit_thresholds, predict_sets,
                        sigmoid, topk_sets)
from .config import INTERACTION_TYPES, SEED, WATER_TYPES
from .dataset import load_dataset
from .experiment_cis import ALPHAS, METHODS, atom_groups, train_model
from .metrics import typed_metrics
from .splits import (scaffold_folds, strict_temporal_folds, target_folds,
                     temporal_folds)
from .train import set_seed


def split_kfold(samples, tr, k, seed):
    targets = sorted({samples[i]["target"] for i in tr})
    rng = random.Random(seed)
    rng.shuffle(targets)
    folds = [[] for _ in range(k)]
    for j, t in enumerate(targets):
        folds[j % k].append(t)
    out = []
    for fs in folds:
        fs = set(fs)
        out.append([int(i) for i in tr if samples[i]["target"] in fs])
    return out



def score_preds(preds, samples, lam=0.5, mode="geo_add", cutoff_shift=0.0):
    probs = sigmoid(preds["logits"])
    ids = preds["sample_index"]
    order, seen = [], set()
    for i in ids:
        j = int(i)
        if j not in seen:
            seen.add(j)
            order.append(j)
    cutoffs = tuple(c + cutoff_shift for c in CUTOFFS)
    out = []
    for i in order:
        sel = ids == i
        s = samples[i]
        out.append(chemistry_score(probs[sel], np.asarray(s["lig_xyz"]),
                                   np.asarray(s["pocket_xyz"]),
                                   np.asarray(s["pocket_roles"]), cutoffs, lam, mode,
                                   water_xyz=s.get("water_xyz")))
    return np.concatenate(out)
def run_shard(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    no_mask = getattr(args, "no_mask", False)
    samples = load_dataset(args.data)
    n_types = samples[0]["labels"].shape[1]
    prepared = prepare_cod(samples, device)
    if args.split == "temporal_strict":
        folds = strict_temporal_folds(samples, args.cutoff)
    elif args.split == "temporal":
        folds = temporal_folds(samples, args.cutoff)
    elif args.split == "target":
        folds = target_folds(samples, args.folds)
    else:
        folds = scaffold_folds(samples, args.folds)
    combos = []
    for seed in args.seeds:
        for fi, (tr, te) in enumerate(folds):
            combos.append((seed, fi, tr, te))
    mine = [c for k, c in enumerate(combos) if k % args.nshards == args.shard]
    print(f"device={device} n={len(samples)} combos={len(mine)} no_mask={no_mask}", flush=True)

    records = []
    for seed, fi, tr, te in mine:
        set_seed(seed)
        inner = split_kfold(samples, tr, args.inner_folds, seed)
        pooled = {k: [] for k in ("logits", "labels", "mask", "capability", "sample_index")}
        inner_test_logits = []
        for j, cal_idx in enumerate(inner):
            if not cal_idx:
                continue
            cal_set = set(cal_idx)
            fit_idx = [int(i) for i in tr if int(i) not in cal_set]
            set_seed(seed * 100 + j)
            model = train_model(samples, prepared, fit_idx, device, args.epochs,
                                seed + j, backbone=args.backbone,
                                use_cap_gate=not no_mask, n_types=n_types)
            pc = predict_cod(model, prepared, cal_idx)
            pt_in = predict_cod(model, prepared, te)
            inner_test_logits.append(pt_in["logits"])
            for key in pooled:
                pooled[key].append(pc[key])
        cal = {key: np.concatenate(v) for key, v in pooled.items()}
        g_cal = atom_groups(cal, samples)
        probs_c = 1.0 / (1.0 + np.exp(-cal["logits"]))

        set_seed(seed * 1000 + 7)
        model_full = train_model(samples, prepared, [int(i) for i in tr],
                                 device, args.epochs, seed + 7,
                                 backbone=args.backbone,
                                 use_cap_gate=not no_mask, n_types=n_types)
        pt = predict_cod(model_full, prepared, te)
        g_test = atom_groups(pt, samples)
        probs_t = 1.0 / (1.0 + np.exp(-pt["logits"]))
        t_test = np.asarray([samples[int(i)]["target"] for i in pt["sample_index"]], dtype=object)
        model_metrics = typed_metrics(
            pt["logits"], pt["labels"], pt["mask"], pt["capability"],
            types=WATER_TYPES if n_types == 6 else INTERACTION_TYPES)
        records.append({"seed": int(seed), "fold": int(fi), "alpha": 0.0,
                        "method": "model", "model": model_metrics})
        cap_c = np.ones_like(cal["capability"]) if no_mask else cal["capability"]
        cap_t = np.ones_like(pt["capability"]) if no_mask else pt["capability"]
        temp = fit_temperature(cal["logits"], cal["labels"], cal["mask"], cap_c)
        probs_c_T = sigmoid(cal["logits"] / temp)
        probs_t_T = sigmoid(pt["logits"] / temp)
        ens_test = np.mean([sigmoid(x) for x in inner_test_logits], axis=0)
        print(f"[seed={seed} fold={fi}] inner={args.inner_folds} "
              f"cal_atoms={len(cal['labels'])} test={len(te)} T={temp:.2f}", flush=True)

        for alpha in ALPHAS:
            q = fit_thresholds(probs_c, cal["labels"], cal["mask"],
                               cap_c, g_cal, alpha)

            # conformal risk control: FNR <= alpha with a Hoeffding bound
            tp_cal = cal["mask"][:, None] & (cap_c > 0.5) & (cal["labels"] > 0.5)
            tau_g = crc_threshold((1.0 - probs_c)[tp_cal], alpha)
            sets = ((1.0 - probs_t) <= tau_g) & (cap_t > 0.5)
            m = evaluate_sets(sets, pt["labels"], pt["mask"], pt["capability"], g_test,
                              targets=t_test)
            records.append({"seed": int(seed), "fold": int(fi), "alpha": alpha,
                            "method": "crc_global", **m})
            q_crc = {}
            for k in range(n_types):
                sel = cal["mask"] & (cap_c[:, k] > 0.5) & (cal["labels"][:, k] > 0.5)
                q_crc[("type", k)] = (crc_threshold((1.0 - probs_c)[sel, k], alpha)
                                      if sel.sum() else 1.0)
            sets = predict_sets(probs_t, cap_t, q_crc, g_test, "type")
            m = evaluate_sets(sets, pt["labels"], pt["mask"], pt["capability"], g_test,
                              targets=t_test)
            records.append({"seed": int(seed), "fold": int(fi), "alpha": alpha,
                            "method": "crc_type", **m})
            for method in ("global_single", "type", "family"):
                sets = predict_sets(probs_t, cap_t, q, g_test, method)
                m = evaluate_sets(sets, pt["labels"], pt["mask"], pt["capability"], g_test,
                                  targets=t_test)
                records.append({"seed": int(seed), "fold": int(fi), "alpha": alpha,
                                "method": method, **m})
            q_t = fit_thresholds(probs_c_T, cal["labels"], cal["mask"],
                                 cap_c, g_cal, alpha)
            sets = predict_sets(probs_t_T, cap_t, q_t, g_test, "type")
            m = evaluate_sets(sets, pt["labels"], pt["mask"], pt["capability"], g_test,
                              targets=t_test)
            records.append({"seed": int(seed), "fold": int(fi), "alpha": alpha,
                            "method": "temp", **m})
            sets = predict_sets(ens_test, cap_t, q, g_test, "type")
            m = evaluate_sets(sets, pt["labels"], pt["mask"], pt["capability"], g_test,
                              targets=t_test)
            records.append({"seed": int(seed), "fold": int(fi), "alpha": alpha,
                            "method": "ensemble", **m})
            for mode in ("geo_add", "geo_partner"):
                sc_c = score_preds(cal, samples, 0.5, mode)
                sc_t = score_preds(pt, samples, 0.5, mode)
                q_c = fit_thresholds(1.0 - sc_c, cal["labels"], cal["mask"],
                                     cap_c, g_cal, alpha)
                sets = predict_sets(1.0 - sc_t, cap_t, q_c, g_test, "type")
                m = evaluate_sets(sets, pt["labels"], pt["mask"], pt["capability"], g_test,
                                  targets=t_test)
                records.append({"seed": int(seed), "fold": int(fi), "alpha": alpha,
                                "method": mode, **m})
                tp_cal_b = (cal["mask"][:, None] & (cap_c > 0.5)
                            & (cal["labels"] > 0.5))
                tau = crc_threshold(sc_c[tp_cal_b], alpha)
                sets = (sc_t <= tau) & (cap_t > 0.5)
                m = evaluate_sets(sets, pt["labels"], pt["mask"], pt["capability"], g_test,
                                  targets=t_test)
                records.append({"seed": int(seed), "fold": int(fi), "alpha": alpha,
                                "method": f"crc_{mode}", **m})
            def _eval_geo(method, lam, shift):
                sc_c = score_preds(cal, samples, lam, "geo_partner", shift)
                sc_t = score_preds(pt, samples, lam, "geo_partner", shift)
                q_c = fit_thresholds(1.0 - sc_c, cal["labels"], cal["mask"],
                                     cap_c, g_cal, alpha)
                sets = predict_sets(1.0 - sc_t, cap_t, q_c, g_test, "type")
                m = evaluate_sets(sets, pt["labels"], pt["mask"], pt["capability"],
                                  g_test, targets=t_test)
                records.append({"seed": int(seed), "fold": int(fi), "alpha": alpha,
                                "method": method, **m})
                tp_cal_b = (cal["mask"][:, None] & (cap_c > 0.5)
                            & (cal["labels"] > 0.5))
                tau = crc_threshold(sc_c[tp_cal_b], alpha)
                sets = (sc_t <= tau) & (cap_t > 0.5)
                m = evaluate_sets(sets, pt["labels"], pt["mask"], pt["capability"],
                                  g_test, targets=t_test)
                records.append({"seed": int(seed), "fold": int(fi), "alpha": alpha,
                                "method": f"crc_{method}", **m})
            for lam, tag in ((0.25, "l025"), (1.0, "l100")):
                _eval_geo(f"geo_{tag}", lam, 0.0)
            _eval_geo("geo_s05", 0.5, 0.5)
            for k in (1, 2, 3):
                sets = topk_sets(probs_t, cap_t, k)
                m = evaluate_sets(sets, pt["labels"], pt["mask"], pt["capability"], g_test,
                                  targets=t_test)
                records.append({"seed": int(seed), "fold": int(fi), "alpha": alpha,
                                "method": f"top{k}", **m})
        print("  done", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"shard_{args.shard}.json")
    with open(out, "w") as fh:
        json.dump(records, fh)
    print("saved", out, len(records), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "combine"])
    ap.add_argument("--data", default="results/dataset_cod.pkl")
    ap.add_argument("--outdir", default="results/cis_cv")
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--backbone", default="mlp", choices=["mlp", "egnn"])
    ap.add_argument("--split", default="target", choices=["target", "scaffold", "temporal", "temporal_strict"])
    ap.add_argument("--cutoff", default="2020-01-01")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args(argv)
    set_seed(SEED)
    if args.mode == "run":
        run_shard(args)
    else:
        from .experiment_cis import combine
        combine(args)


if __name__ == "__main__":
    main()
