"""Recurring water-bridged residues in high- vs low-water targets."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from xact.chemistry import ligand_roles, protein_roles
from xact.dataset import load_dataset
from xact.pdbio import parse_pdb, select_ligand_copy
from xact.water import parse_waters

PDB_DIR = "/home/xiao/workspace/drug-design/data/pdb_hops/pdb"
TARGETS = ("HSP90", "DHFR", "VEGFR2", "ERa")
CUT = 3.5


def bridges(parsed, mol, pdb):
    lr = ligand_roles(mol)
    pr = protein_roles(parsed["protein"])
    pxyz = np.asarray([a["xyz"] for a in parsed["protein"]], dtype=np.float32)
    conf = mol.GetConformer()
    lxyz = np.asarray([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])
    lig_polar = lr["donor"] | lr["acceptor"]
    prot_polar = pr["donor"] | pr["acceptor"]
    out = []
    for w in parse_waters(pdb):
        dl = np.linalg.norm(lxyz - w["xyz"], axis=1)
        dp = np.linalg.norm(pxyz - w["xyz"], axis=1)
        li = np.where((dl <= CUT) & lig_polar)[0]
        pj = np.where((dp <= CUT) & prot_polar)[0]
        for i in li:
            for j in pj:
                if np.linalg.norm(lxyz[i] - pxyz[j]) <= CUT:
                    continue
                vl = lxyz[i] - w["xyz"]
                vp = pxyz[j] - w["xyz"]
                cos = float(np.dot(vl, vp) /
                            (np.linalg.norm(vl) * np.linalg.norm(vp) + 1e-9))
                ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
                if not (60 <= ang <= 180):
                    continue
                comp = ((lr["donor"][i] and pr["acceptor"][j])
                        or (lr["acceptor"][i] and pr["donor"][j]))
                if comp:
                    a = parsed["protein"][j]
                    out.append((a["chain"], a["resseq"], a["resname"]))
    return out


def main():
    samples = load_dataset("data/dataset_cod.pkl")
    res = {}
    for target in TARGETS:
        idxs = [i for i, s in enumerate(samples) if s["target"] == target]
        by_res = defaultdict(set)
        n_with = 0
        total = 0
        for i in idxs:
            s = samples[i]
            pdb = f"{PDB_DIR}/{s['pdb']}.pdb"
            try:
                parsed = parse_pdb(pdb)
                sel = select_ligand_copy(pdb, s["ccd"], s["smiles"], parsed=parsed)
                if sel is None:
                    continue
                br = bridges(parsed, sel["mol"], pdb)
            except Exception:
                continue
            if br:
                n_with += 1
            total += len(br)
            for r in set(br):
                by_res[r].add(i)
        top = sorted(((f"{r[2]}{r[1]}.{r[0]}", len(v) / max(1, len(idxs)))
                      for r, v in by_res.items()), key=lambda x: -x[1])[:12]
        res[target] = {"n_complexes": len(idxs), "with_bridge": n_with,
                       "total_bridges": total,
                       "top_residues": [{"residue": r, "fraction": f}
                                        for r, f in top]}
        print(target, res[target]["n_complexes"], n_with, total,
              [t["residue"] for t in res[target]["top_residues"][:5]], flush=True)
    with open("data/water_network_case.json", "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
