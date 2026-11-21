"""Interaction chemistry: atom roles and typed contact labels.

Labels are built from experimental crystal poses. Protein residue identity is
used here (and only here) to define chemistry; it is never a model input.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import Chem

from .config import (
    AROMATIC_RING_ATOMS,
    BACKBONE_ATOMS,
    CHARGED_NEGATIVE,
    CHARGED_POSITIVE,
    HALOGEN_DIST,
    HALOGENS,
    HBOND_DIST,
    HYDROPHOBIC_DIST,
    MASK_RADIUS,
    N_TYPES,
    PI_DIST,
    PROTEIN_ACCEPTORS,
    PROTEIN_DONORS,
    SALT_DIST,
)


def _amide_nitrogen(atom: Chem.Atom) -> bool:
    for nb in atom.GetNeighbors():
        if nb.GetSymbol() == "C":
            for b in nb.GetBonds():
                other = b.GetOtherAtom(nb)
                if other.GetSymbol() == "O" and b.GetBondType() == Chem.BondType.DOUBLE:
                    return True
    return False


def ligand_roles(mol: Chem.Mol) -> dict:
    n = mol.GetNumAtoms()
    donor = np.zeros(n, dtype=bool)
    acceptor = np.zeros(n, dtype=bool)
    aromatic = np.zeros(n, dtype=bool)
    hydrophobic = np.zeros(n, dtype=bool)
    halogen = np.zeros(n, dtype=bool)
    charged = np.zeros(n, dtype=bool)
    ring = np.zeros(n, dtype=bool)

    ri = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        sym = atom.GetSymbol()
        fc = atom.GetFormalCharge()
        donor[i] = sym in ("N", "O", "S") and atom.GetTotalNumHs() > 0
        acceptor[i] = sym == "O" and fc <= 0
        if sym == "N" and fc <= 0 and not _amide_nitrogen(atom):
            acceptor[i] = True
        aromatic[i] = atom.GetIsAromatic()
        halogen[i] = sym in HALOGENS
        charged[i] = fc != 0
        ring[i] = atom.IsInRing()
        if sym == "C":
            hydrophobic[i] = not any(nb.GetSymbol() in ("N", "O", "S") for nb in atom.GetNeighbors())

    return {
        "donor": donor,
        "acceptor": acceptor,
        "aromatic": aromatic,
        "hydrophobic": hydrophobic,
        "halogen": halogen,
        "charged": charged,
        "ring": ring,
    }


def _protein_token(resname: str, atomname: str) -> str:
    if atomname in BACKBONE_ATOMS:
        return atomname
    return f"{resname}:{atomname}"


def protein_roles(protein: list[dict]) -> dict:
    m = len(protein)
    donor = np.zeros(m, dtype=bool)
    acceptor = np.zeros(m, dtype=bool)
    charged = np.zeros(m, dtype=bool)
    hydrophobic = np.zeros(m, dtype=bool)
    polar = np.zeros(m, dtype=bool)
    aromatic = np.zeros(m, dtype=bool)

    for i, a in enumerate(protein):
        tok = _protein_token(a["resname"], a["name"])
        donor[i] = tok in PROTEIN_DONORS
        acceptor[i] = tok in PROTEIN_ACCEPTORS
        charged[i] = a["resname"] in CHARGED_POSITIVE or a["resname"] in CHARGED_NEGATIVE
        hydrophobic[i] = a["element"] == "C" and a["resname"] not in (
            "ASP", "GLU", "ASN", "GLN", "ARG", "LYS", "HIS", "SER", "THR", "TYR",
        )
        polar[i] = a["element"] in ("N", "O")
        aromatic[i] = a["name"] in AROMATIC_RING_ATOMS.get(a["resname"], set())

    return {
        "donor": donor,
        "acceptor": acceptor,
        "charged": charged,
        "hydrophobic": hydrophobic,
        "polar": polar,
        "aromatic": aromatic,
    }


def _ligand_aromatic_rings(mol: Chem.Mol) -> list[np.ndarray]:
    rings = []
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) >= 5 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            rings.append(np.asarray(ring, dtype=int))
    return rings


def _protein_aromatic_rings(protein: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped = defaultdict(list)
    for i, a in enumerate(protein):
        ring_names = AROMATIC_RING_ATOMS.get(a["resname"], set())
        if a["name"] in ring_names:
            grouped[(a["chain"], a["resseq"], a["resname"])].append(i)
    out = []
    for _, idx in grouped.items():
        idx = np.asarray(idx, dtype=int)
        xyz = np.asarray([protein[i]["xyz"] for i in idx], dtype=np.float32)
        if len(idx) >= 5:
            out.append((idx, xyz.mean(axis=0)))
    return out


def typed_contacts(mol: Chem.Mol, lig_xyz: np.ndarray, protein: list[dict],
                   cutoff_shift: float = 0.0) -> dict:
    """Return per-ligand-atom multi-label contact chemistry.

    labels: (n_lig, 5) float32, mask: (n_lig,) bool, contact_any: (n_lig,) bool
    Order: hbond, hydrophobic, pi, salt, halogen.
    """
    hb_dist = HBOND_DIST + cutoff_shift
    hp_dist = HYDROPHOBIC_DIST + cutoff_shift
    pi_dist = PI_DIST + cutoff_shift
    salt_dist = SALT_DIST + cutoff_shift
    xb_dist = HALOGEN_DIST + cutoff_shift
    lr = ligand_roles(mol)
    pr = protein_roles(protein)
    prot_xyz = np.asarray([a["xyz"] for a in protein], dtype=np.float32)

    n = len(lig_xyz)
    m = len(protein)
    labels = np.zeros((n, N_TYPES), dtype=np.float32)
    if n == 0 or m == 0:
        return {"labels": labels, "mask": np.zeros(n, dtype=bool),
                "contact_any": np.zeros(n, dtype=bool),
                "pocket_roles": np.zeros((m, N_TYPES), dtype=np.float32)}

    d = np.linalg.norm(lig_xyz[:, None, :] - prot_xyz[None, :, :], axis=2)
    dmin = d.min(axis=1)
    mask = dmin <= MASK_RADIUS
    contact_any = dmin <= 4.5

    # hbond
    pair = d <= hb_dist
    hb = (lr["donor"][:, None] & pr["acceptor"][None, :]) | (
        lr["acceptor"][:, None] & pr["donor"][None, :]
    )
    labels[:, 0] = (pair & hb).any(axis=1)

    # hydrophobic
    pair = d <= hp_dist
    hp = lr["hydrophobic"][:, None] & pr["hydrophobic"][None, :]
    labels[:, 1] = (pair & hp).any(axis=1)

    # salt bridge
    pair = d <= salt_dist
    sb = lr["charged"][:, None] & pr["charged"][None, :]
    labels[:, 3] = (pair & sb).any(axis=1)

    # halogen bond
    pair = d <= xb_dist
    xb = lr["halogen"][:, None] & pr["polar"][None, :]
    labels[:, 4] = (pair & xb).any(axis=1)

    # pi-stacking (ring centroid to ring centroid)
    lig_rings = _ligand_aromatic_rings(mol)
    prot_rings = _protein_aromatic_rings(protein)
    if lig_rings and prot_rings:
        lc = np.asarray([lig_xyz[r].mean(axis=0) for r in lig_rings], dtype=np.float32)
        pc = np.asarray([c for _, c in prot_rings], dtype=np.float32)
        dd = np.linalg.norm(lc[:, None, :] - pc[None, :, :], axis=2)
        hit = dd <= pi_dist
        for k, ring in enumerate(lig_rings):
            if hit[k].any():
                labels[ring, 2] = 1.0

    return {"labels": labels, "mask": mask, "contact_any": contact_any,
            "pocket_roles": np.stack([
                pr["donor"] | pr["acceptor"],
                pr["hydrophobic"],
                pr["aromatic"],
                pr["charged"],
                pr["polar"],
            ], axis=1).astype(np.float32)}
