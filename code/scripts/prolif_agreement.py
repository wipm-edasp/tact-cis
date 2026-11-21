"""Compare distance-based TACT labels with ProLIF on 60 complexes.

PDBs are hydrogenated with OpenBabel before ProLIF because the benchmark PDBs
lack explicit hydrogens/bonds. Heavy ligand atoms are matched by coordinates.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile

import numpy as np
import MDAnalysis as mda
import prolif as plf
import prolif.exceptions as pe

from xact.dataset import load_dataset

pe.FragmentedResidueError.on_error = "warn"
OBABEL = "/home/xiao/.conda/envs/scaffold_vae/bin/obabel"
PDB_DIR = "/home/xiao/workspace/drug-design/data/pdb_hops/pdb"
TYPES = ("hbond", "hydrophobic", "pi", "salt", "halogen")
TYPE_MAP = {
    "HBDonor": "hbond", "HBAcceptor": "hbond",
    "Hydrophobic": "hydrophobic",
    "PiStacking": "pi",
    "Cationic": "salt", "Anionic": "salt",
    "XBDonor": "halogen", "XBAcceptor": "halogen",
}


def pick_samples(samples, n_per_target=2, total=60, seed=0):
    rng = random.Random(seed)
    by = {}
    for i, s in enumerate(samples):
        by.setdefault(s["target"], []).append(i)
    chosen = []
    for t in sorted(by):
        idxs = by[t][:]
        rng.shuffle(idxs)
        chosen.extend(idxs[:n_per_target])
    rest = [i for i in range(len(samples)) if i not in set(chosen)]
    rng.shuffle(rest)
    chosen.extend(rest[:max(0, total - len(chosen))])
    return chosen[:total]


def prep_pdb(pdb_path, tmpdir):
    out = os.path.join(tmpdir, "prepared.pdb")
    subprocess.run([OBABEL, pdb_path, "-O", out, "-h"],
                   check=True, capture_output=True, text=True)
    return out


def prolif_labels(pdb_path, ccd, lig_xyz, tmpdir):
    prepared = prep_pdb(pdb_path, tmpdir)
    u = mda.Universe(prepared)
    res = u.select_atoms(f"resname {ccd}")
    if len(res) == 0:
        raise RuntimeError("no ligand")
    groups = {}
    for a in res:
        groups.setdefault((a.segid, a.resid), []).append(a.index)
    cent = lig_xyz.mean(axis=0)
    best = min(((np.linalg.norm(u.atoms[idx].positions.mean(axis=0) - cent), k)
                for k, idx in groups.items()))
    lig = u.atoms[groups[best[1]]]
    prot = u.select_atoms("protein")
    fp = plf.Fingerprint(["HBDonor", "HBAcceptor", "Hydrophobic", "PiStacking",
                          "Cationic", "Anionic", "XBDonor", "XBAcceptor"])
    fp.run(u.trajectory, lig, prot, progress=False, n_jobs=1,
           converter_kwargs=({"inferrer": None}, {"inferrer": None}))
    ifp = fp.ifp[0]
    # Map ProLIF ligand group-local indices to heavy-atom order.
    heavy = [j for j, a in enumerate(lig) if not a.name.startswith("H")]
    heavy_xyz = np.asarray([a.position for a in lig if not a.name.startswith("H")])
    # Match to the sample's heavy-atom order by nearest coordinate.
    order = []
    for x in lig_xyz:
        d = np.linalg.norm(heavy_xyz - x, axis=1)
        j = int(np.argmin(d))
        order.append(j if d[j] < 0.5 else -1)
    out = np.zeros((len(lig_xyz), len(TYPES)), dtype=bool)
    for (lig_res, prot_res), inter_dict in ifp.items():
        for iname, entries in inter_dict.items():
            typ = TYPE_MAP.get(iname)
            if typ is None:
                continue
            k = TYPES.index(typ)
            for e in entries:
                for local_idx in e["parent_indices"]["ligand"]:
                    pos = heavy.index(local_idx) if local_idx in heavy else None
                    if pos is None:
                        continue
                    for atom_pos, mapped in enumerate(order):
                        if mapped == pos:
                            out[atom_pos, k] = True
    return out


def main():
    samples = load_dataset("data/dataset_cod.pkl")
    chosen = pick_samples(samples)
    tp = np.zeros(len(TYPES), dtype=int)
    fp = np.zeros(len(TYPES), dtype=int)
    fn = np.zeros(len(TYPES), dtype=int)
    both = np.zeros(len(TYPES), dtype=int)
    ours_only = np.zeros(len(TYPES), dtype=int)
    plf_only = np.zeros(len(TYPES), dtype=int)
    neither = np.zeros(len(TYPES), dtype=int)
    rows = []
    n_ok = 0
    tmpdir = tempfile.mkdtemp()
    try:
        for n, i in enumerate(chosen, 1):
            s = samples[i]
            pdb = os.path.join(PDB_DIR, f"{s['pdb']}.pdb")
            try:
                pl = prolif_labels(pdb, s["ccd"], np.asarray(s["lig_xyz"]), tmpdir)
            except Exception as exc:  # noqa: BLE001
                print(f"[{n}/{len(chosen)}] skip {s['target']} {s['pdb']}: {exc}",
                      flush=True)
                continue
            ours = s["labels"] > 0.5
            if pl.shape != ours.shape:
                print(f"[{n}] shape mismatch {pl.shape} vs {ours.shape}", flush=True)
                continue
            n_ok += 1
            for k, t in enumerate(TYPES):
                tp[k] += int((pl[:, k] & ours[:, k]).sum())
                fp[k] += int((pl[:, k] & ~ours[:, k]).sum())
                fn[k] += int((~pl[:, k] & ours[:, k]).sum())
                a, b = bool(ours[:, k].any()), bool(pl[:, k].any())
                if a and b:
                    both[k] += 1
                elif a:
                    ours_only[k] += 1
                elif b:
                    plf_only[k] += 1
                else:
                    neither[k] += 1
            rows.append({"target": s["target"], "pdb": s["pdb"], "ccd": s["ccd"],
                         "ours": int(ours.sum()), "prolif": int(pl.sum())})
            if n % 10 == 0:
                print(f"[{n}/{len(chosen)}] ok={n_ok}", flush=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    per_type = {}
    for k, t in enumerate(TYPES):
        denom = max(1, tp[k] + fp[k] + fn[k])
        prec = tp[k] / max(1, tp[k] + fp[k])
        rec = tp[k] / max(1, tp[k] + fn[k])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        per_type[t] = {
            "tp": int(tp[k]), "fp": int(fp[k]), "fn": int(fn[k]),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "jaccard": float(tp[k] / denom),
            "complex_both": int(both[k]), "complex_ours_only": int(ours_only[k]),
            "complex_prolif_only": int(plf_only[k]), "complex_neither": int(neither[k]),
        }
    micro_tp, micro_fp, micro_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
    micro_prec = micro_tp / max(1, micro_tp + micro_fp)
    micro_rec = micro_tp / max(1, micro_tp + micro_fn)
    micro_f1 = 2 * micro_prec * micro_rec / max(1e-9, micro_prec + micro_rec)
    result = {
        "n_complexes": n_ok,
        "n_requested": len(chosen),
        "micro": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1},
        "per_type": per_type,
        "per_complex": rows,
    }
    with open("results/prolif_agreement.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({k: result[k] for k in ("n_complexes", "n_requested", "micro")},
                     indent=2))
    for t, v in per_type.items():
        print(t, "F1=%.3f" % v["f1"], "prec=%.3f" % v["precision"],
              "rec=%.3f" % v["recall"], "complex_both=%d" % v["complex_both"])


if __name__ == "__main__":
    main()
