"""Audit water-bridge labels against PLIP, one subprocess per complex."""
from __future__ import annotations

import json
import os
import random
import subprocess
import tempfile

import numpy as np

from xact.dataset import load_dataset

PDB_DIR = "/home/xiao/workspace/drug-design/data/pdb_hops/pdb"
PY = "/home/xiao/.conda/envs/scaffold_vae/bin/python"
ONE = "/home/xiao/workspace/tact/plip_one.py"


def main():
    samples = load_dataset("data/dataset_cod.pkl")
    rows = json.load(open("data/water_bridge_labels.json"))
    rng = random.Random(0)
    by = {}
    for i, s in enumerate(samples):
        by.setdefault(s["target"], []).append(i)
    chosen = []
    for t in sorted(by):
        idxs = by[t][:]
        rng.shuffle(idxs)
        chosen.extend(idxs[:2])
    out = []
    tp = fp = fn = 0
    both_complex = 0
    for n, i in enumerate(chosen, 1):
        s = samples[i]
        pdb = f"{PDB_DIR}/{s['pdb']}.pdb"
        tmp = tempfile.mktemp(suffix=".json")
        try:
            r = subprocess.run([PY, ONE, pdb, s["ccd"], tmp],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                print("skip", s["pdb"], "rc", r.returncode, flush=True)
                continue
            d = json.load(open(tmp))
        except Exception as exc:  # noqa: BLE001
            print("skip", s["pdb"], exc, flush=True)
            continue
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        lig_xyz = np.asarray(s["lig_xyz"])
        plip = set()
        for xyz in d["atoms"]:
            dist = np.linalg.norm(lig_xyz - np.asarray(xyz)[None, :], axis=1)
            j = int(np.argmin(dist))
            if dist[j] < 0.5:
                plip.add(j)
        ours = set(rows[i].get("bridged_atoms", []))
        tp += len(plip & ours)
        fp += len(plip - ours)
        fn += len(ours - plip)
        both_complex += int(bool(plip) and bool(ours))
        out.append({"target": s["target"], "pdb": s["pdb"], "ccd": s["ccd"],
                    "plip_atoms": sorted(plip), "our_atoms": sorted(ours),
                    "plip_bridges": d["n_bridges"]})
        print(n, s["target"], s["pdb"], "plip", len(plip), "ours", len(ours),
              "intersect", len(plip & ours), flush=True)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    summary = {"n": len(out), "tp": tp, "fp": fp, "fn": fn,
               "precision": prec, "recall": rec, "f1": f1,
               "both_complex": both_complex}
    print(json.dumps(summary, indent=2))
    with open("results/plip_water_audit_60.json", "w") as fh:
        json.dump({"summary": summary, "rows": out}, fh, indent=2)


if __name__ == "__main__":
    main()
