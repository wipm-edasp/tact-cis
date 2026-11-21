"""Build typed-contact samples from the SV-Hop crystal benchmark."""
from __future__ import annotations

import os
import pickle
from collections import Counter
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .chemistry import ligand_roles, protein_roles, typed_contacts
from .config import POCKET_RADIUS
from .features import bond_features, ligand_features, pocket_features
from .pdbio import ligand_coords, parse_pdb, select_ligand_copy
from .water import (ligand_water_features, parse_waters, pocket_water_features,
                    water_bridge_mask)


def murcko(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core) if core is not None else ""


def build_sample(
    pdb_path: str,
    target: str,
    pdb: str,
    ccd: str,
    smiles: str,
    aware: bool = False,
    cutoff_shift: float = 0.0,
    include_water: bool = False,
) -> Optional[dict]:
    parsed = parse_pdb(pdb_path)
    if not parsed["protein"]:
        return None
    sel = select_ligand_copy(pdb_path, ccd, smiles, parsed=parsed)
    if sel is None:
        return None
    mol = sel["mol"]
    xyz = ligand_coords(mol)
    roles = ligand_roles(mol)

    all_prot = parsed["protein"]
    prot_xyz = np.asarray([a["xyz"] for a in all_prot], dtype=np.float32)
    dmin = np.linalg.norm(xyz[:, None, :] - prot_xyz[None, :, :], axis=2).min(axis=0)
    pocket_atoms = [a for a, keep in zip(all_prot, dmin <= POCKET_RADIUS) if keep]
    if not pocket_atoms:
        return None

    tc = typed_contacts(mol, xyz, pocket_atoms, cutoff_shift=cutoff_shift)
    ei, ea = bond_features(mol)
    capability = np.stack([
        roles["donor"] | roles["acceptor"],
        roles["hydrophobic"],
        roles["aromatic"],
        roles["charged"],
        roles["halogen"],
    ], axis=1).astype(np.float32)
    pocket_xyz = np.asarray([a["xyz"] for a in pocket_atoms], dtype=np.float32)
    lig_feats = ligand_features(mol, roles)
    pocket_feats = pocket_features(pocket_atoms, aware=aware)
    labels = tc["labels"]
    pocket_roles = tc["pocket_roles"]
    water_xyz = np.zeros((0, 3), dtype=np.float32)
    if include_water:
        waters = parse_waters(pdb_path)
        wmask, _ = water_bridge_mask(mol, xyz, pocket_atoms, waters)
        labels = np.concatenate(
            [labels, wmask[:, None].astype(np.float32)], axis=1)
        capability = np.concatenate(
            [capability,
             (roles["donor"] | roles["acceptor"])[:, None].astype(np.float32)],
            axis=1)
        pr = protein_roles(pocket_atoms)
        pocket_roles = np.concatenate(
            [pocket_roles,
             (pr["donor"] | pr["acceptor"])[:, None].astype(np.float32)],
            axis=1)
        water_xyz = np.asarray([w["xyz"] for w in waters], dtype=np.float32)
        lig_feats = np.concatenate(
            [lig_feats, ligand_water_features(xyz, waters)], axis=1)
        pocket_feats = np.concatenate(
            [pocket_feats, pocket_water_features(pocket_xyz, waters)], axis=1)
    return {
        "target": target,
        "pdb": pdb,
        "ccd": ccd,
        "smiles": smiles,
        "scaffold": murcko(smiles),
        "lig_feats": lig_feats,
        "lig_xyz": xyz,
        "edge_index": ei,
        "edge_attr": ea,
        "pocket_feats": pocket_feats,
        "pocket_xyz": pocket_xyz,
        "labels": labels,
        "mask": tc["mask"],
        "contact_any": tc["contact_any"],
        "capability": capability,
        "pocket_roles": pocket_roles,
        "water_xyz": water_xyz,
    }


def build_dataset(
    csv_path: str,
    pdb_dir: str,
    limit: Optional[int] = None,
    targets: Optional[Iterable[str]] = None,
    aware: bool = False,
    cutoff_shift: float = 0.0,
    include_water: bool = False,
) -> tuple[list[dict], Counter]:
    df = pd.read_csv(csv_path)
    if targets is not None:
        keep = set(targets)
        df = df[df.target.isin(keep)]
    if limit is not None:
        df = df.head(limit)
    samples, reasons = [], Counter()
    for r in df.itertuples():
        path = os.path.join(pdb_dir, f"{r.pdb}.pdb")
        if not os.path.exists(path):
            reasons["missing_pdb"] += 1
            continue
        try:
            s = build_sample(path, r.target, r.pdb, r.ccd, r.smiles, aware=aware,
                             cutoff_shift=cutoff_shift,
                             include_water=include_water)
        except Exception as exc:  # noqa: BLE001
            reasons[f"error:{type(exc).__name__}"] += 1
            continue
        if s is None:
            reasons["no_ligand_or_bonds"] += 1
            continue
        if s["labels"].sum() == 0:
            reasons["no_typed_contact"] += 1
            continue
        samples.append(s)
        reasons["ok"] += 1
    return samples, reasons


def save_dataset(samples: list[dict], path: str) -> None:
    with open(path, "wb") as fh:
        pickle.dump(samples, fh)


def load_dataset(path: str) -> list[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)
