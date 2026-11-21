"""PDB parsing and per-copy crystal-ligand extraction (dependency-light).

Protein atoms are parsed from fixed-width ATOM records. Ligand copies are
extracted residue-by-residue from the raw text (RDKit's SplitMolByPDBResidues
merges copies that share a residue name), then bond orders are assigned from the
benchmark SMILES. No residue metadata is kept for model input.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from .config import CONTACT_RADIUS

RDLogger.DisableLog("rdApp.*")


def _element_from_line(line: str) -> str:
    elem = line[76:78].strip()
    if elem:
        return elem.upper()
    name = line[12:16].strip()
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return ""


def parse_pdb(path: str) -> dict:
    protein = []
    hetero = defaultdict(list)
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            rec = line[:6]
            if rec not in ("ATOM  ", "HETATM"):
                continue
            altloc = line[16]
            if altloc not in (" ", "", "A"):
                continue
            elem = _element_from_line(line)
            if elem in ("H", "D", ""):
                continue
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            resname = line[17:20].strip().upper()
            atomname = line[12:16].strip()
            chain = line[21]
            resseq = int(line[22:26])
            if rec == "ATOM  ":
                protein.append(
                    {"element": elem, "name": atomname, "resname": resname,
                     "resseq": resseq, "chain": chain, "xyz": xyz}
                )
            else:
                hetero[(chain, resseq, resname)].append(
                    {"element": elem, "name": atomname, "xyz": xyz,
                     "serial": int(line[6:11])}
                )
    return {"protein": protein, "hetero": dict(hetero)}


def protein_coords(parsed: dict) -> np.ndarray:
    return np.asarray([a["xyz"] for a in parsed["protein"]], dtype=np.float32)


def _read_lines(path: str) -> list[str]:
    with open(path, "r", errors="ignore") as fh:
        return fh.readlines()


def _conect_map(lines: list[str]) -> dict[int, set[int]]:
    bonds = defaultdict(set)
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        vals = []
        for start in (6, 11, 16, 21, 26):
            chunk = line[start:start + 5].strip()
            if chunk:
                try:
                    vals.append(int(chunk))
                except ValueError:
                    pass
        if len(vals) >= 2:
            a = vals[0]
            for b in vals[1:]:
                bonds[a].add(b)
                bonds[b].add(a)
    return bonds


def _residue_mol(lines: list[str], conect: dict[int, set[int]],
                  group: list[dict]) -> Optional[Chem.Mol]:
    nums = [a["serial"] for a in group]
    remap = {old: i + 1 for i, old in enumerate(nums)}
    keep = set(nums)

    out = []
    for a in group:
        line = a["_line"]
        out.append(f"{line[:6]}{remap[a['serial']]:5d}{line[11:]}")
    for a in group:
        for b in conect.get(a["serial"], ()):
            if b in keep and a["serial"] < b:
                out.append(
                    f"CONECT{remap[a['serial']]:5d}{remap[b]:5d}\n"
                )
    block = "".join(out) + "END\n"
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    return mol


def select_ligand_copy(path: str, ccd: str, smiles: str,
                       parsed: Optional[dict] = None) -> Optional[dict]:
    """Extract every copy of `ccd`, assign bond orders, keep the best contact copy."""
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return None
    if parsed is None:
        parsed = parse_pdb(path)
    prot = protein_coords(parsed)
    if len(prot) == 0:
        return None

    lines = _read_lines(path)
    conect = _conect_map(lines)
    # attach each hetero atom to its original line
    groups = defaultdict(list)
    for line in lines:
        if not line.startswith("HETATM"):
            continue
        if line[16] not in (" ", "", "A"):
            continue
        if line[76:78].strip() in ("H", "D"):
            continue
        key = (line[21], int(line[22:26]), line[17:20].strip().upper())
        groups[key].append({
            "serial": int(line[6:11]),
            "element": _element_from_line(line),
            "name": line[12:16].strip(),
            "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            "_line": line,
        })

    best = None
    for (chain, resseq, resname), group in groups.items():
        if resname != ccd.upper() or len(group) < 6:
            continue
        mol = _residue_mol(lines, conect, group)
        if mol is None:
            continue
        mol = Chem.RemoveHs(mol)
        if mol.GetNumAtoms() != template.GetNumAtoms():
            continue
        mol = assign_bond_orders(mol, smiles)
        if mol is None:
            continue
        conf = mol.GetConformer()
        xyz = np.asarray(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
             for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
        d = np.linalg.norm(xyz[:, None, :] - prot[None, :, :], axis=2)
        n_contacts = int((d.min(axis=1) <= CONTACT_RADIUS).sum())
        cand = {"mol": mol, "chain": chain, "resseq": resseq, "n_contacts": n_contacts}
        if best is None or n_contacts > best["n_contacts"]:
            best = cand
    return best


def assign_bond_orders(mol: Chem.Mol, smiles: str) -> Optional[Chem.Mol]:
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return None
    heavy = Chem.RemoveHs(mol)
    if template.GetNumAtoms() != heavy.GetNumAtoms():
        return None
    try:
        out = AllChem.AssignBondOrdersFromTemplate(template, heavy)
        Chem.SanitizeMol(out)
        return out
    except Exception:
        return None


def ligand_coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.asarray(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )
