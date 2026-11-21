"""Atom featurisation for the residue-blind pocket encoder and ligand GNN."""
from __future__ import annotations

import numpy as np
from rdkit import Chem

from .config import AROMATIC_RING_ATOMS, BACKBONE_ATOMS, ELEMENTS, RESIDUE_TYPES

ATOM_NAME_CATEGORIES = (
    "N", "CA", "C", "O", "CB", "CG", "CD", "CE", "CZ", "NZ", "SIDE_N", "OTHER",
)


def _onehot(value, table) -> np.ndarray:
    out = np.zeros(len(table), dtype=np.float32)
    try:
        out[table.index(value)] = 1.0
    except ValueError:
        out[-1] = 1.0
    return out


def _element_onehot(sym: str) -> np.ndarray:
    return _onehot(sym if sym in ELEMENTS else "other", ELEMENTS)


def _atom_name_category(name: str) -> str:
    if name in ("N", "CA", "C", "O", "CB"):
        return name
    for pref in ("NZ",):
        if name.startswith(pref):
            return "NZ"
    for pref in ("CG", "CD", "CE", "CZ"):
        if name.startswith(pref):
            return pref
    if name.startswith("S"):
        return "SIDE_N"
    if name in ("ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2"):
        return "SIDE_N"
    return "OTHER"


def ligand_features(mol: Chem.Mol, roles: dict) -> np.ndarray:
    rows = []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        fc = int(np.clip(atom.GetFormalCharge(), -2, 2))
        nh = int(np.clip(atom.GetTotalNumHs(), 0, 4))
        deg = int(np.clip(atom.GetDegree(), 0, 4))
        hyb = str(atom.GetHybridization())
        hyb_map = {"S": 0, "SP": 1, "SP2": 2, "SP3": 3}
        h = np.zeros(5, dtype=np.float32)
        h[hyb_map.get(hyb, 4)] = 1.0
        f = np.concatenate([
            _element_onehot(atom.GetSymbol()),
            np.asarray([
                atom.GetIsAromatic(), atom.IsInRing(), roles["donor"][i],
                roles["acceptor"][i], roles["hydrophobic"][i],
                roles["halogen"][i], roles["charged"][i],
            ], dtype=np.float32),
            np.eye(5, dtype=np.float32)[deg],
            np.eye(5, dtype=np.float32)[fc + 2],
            np.eye(5, dtype=np.float32)[nh],
            h,
        ])
        rows.append(f)
    return np.asarray(rows, dtype=np.float32)


def bond_features(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    """Return symmetric edge_index (2, 2E) and edge_attr (2E, 6)."""
    src, dst, feats = [], [], []
    bt_map = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = np.zeros(6, dtype=np.float32)
        f[bt_map.get(b.GetBondType(), 0)] = 1.0
        f[4] = float(b.GetIsConjugated())
        f[5] = float(b.IsInRing())
        for a, c in ((i, j), (j, i)):
            src.append(a)
            dst.append(c)
            feats.append(f)
    if not src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 6), dtype=np.float32)
    return (np.asarray([src, dst], dtype=np.int64),
            np.asarray(feats, dtype=np.float32))


def pocket_features(protein: list[dict], aware: bool = False) -> np.ndarray:
    """Residue-blind (default) or residue-aware pocket atom features."""
    coords = np.asarray([a["xyz"] for a in protein], dtype=np.float32)
    m = len(protein)
    if m == 0:
        return np.zeros((0, 23 + (21 + 12 if aware else 0)), dtype=np.float32)

    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)

    syms = [a["element"] for a in protein]
    rows = []
    for i, a in enumerate(protein):
        near = np.where(d[i] <= 4.0)[0]
        if len(near):
            elems = [syms[k] for k in near]
            tot = float(len(elems))
            frac = [
                sum(e == "C" for e in elems) / tot,
                sum(e == "N" for e in elems) / tot,
                sum(e == "O" for e in elems) / tot,
                sum(e == "S" for e in elems) / tot,
                sum(e not in ("C", "N", "O", "S") for e in elems) / tot,
            ]
        else:
            frac = [0.0, 0.0, 0.0, 0.0, 0.0]

        cov = np.where(d[i] <= 2.5)[0]
        if len(cov) >= 3:
            xyz = coords[cov] - coords[cov].mean(axis=0)
            eig = np.linalg.eigvalsh(xyz.T @ xyz)
            planarity = float(1.0 - eig[0] / (eig[-1] + 1e-6))
        else:
            planarity = 0.0
        density = float(np.log1p((d[i] <= 5.0).sum()))

        f = np.concatenate([
            _element_onehot(a["element"]),
            np.eye(5, dtype=np.float32)[min(int(np.clip(len(near), 0, 4)), 4)],
            np.asarray(frac, dtype=np.float32),
            np.asarray([
                np.clip(planarity, 0.0, 1.0),
                density / 5.0,
                float(a["name"] in BACKBONE_ATOMS),
                float(a["resname"] in ("MSE", "SEP", "TPO", "PTR")),
            ], dtype=np.float32),
        ])
        if aware:
            f = np.concatenate([
                f,
                _onehot(a["resname"] if a["resname"] in RESIDUE_TYPES else "OTHER", RESIDUE_TYPES),
                _onehot(_atom_name_category(a["name"]), ATOM_NAME_CATEGORIES),
            ])
        rows.append(f)
    return np.asarray(rows, dtype=np.float32)
