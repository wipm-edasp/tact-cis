"""Water-mediated interaction labels and features."""
from __future__ import annotations

import numpy as np

from .chemistry import ligand_roles, protein_roles

WATER_CUT = 3.5
MIN_OCC = 0.5
MAX_B = 80.0
MIN_ANGLE = 60.0
MAX_ANGLE = 180.0


def parse_waters(path: str) -> list[dict]:
    out = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            if line[16] not in (" ", "", "A"):
                continue
            if line[17:20].strip().upper() not in ("HOH", "WAT"):
                continue
            elem = line[76:78].strip().upper()
            if elem not in ("O", ""):
                continue
            try:
                occ = float(line[54:60]) if line[54:60].strip() else 1.0
                b = float(line[60:66]) if line[60:66].strip() else 0.0
                xyz = np.array([float(line[30:38]), float(line[38:46]),
                                float(line[46:54])], dtype=np.float32)
            except ValueError:
                continue
            if occ >= MIN_OCC and b <= MAX_B:
                out.append({"xyz": xyz, "occ": occ, "b": b})
    return out


def water_bridge_mask(mol, lig_xyz, protein, waters):
    """Return per-ligand-atom bool mask for classical water-mediated H-bonds."""
    lr = ligand_roles(mol)
    pr = protein_roles(protein)
    pxyz = np.asarray([a["xyz"] for a in protein], dtype=np.float32)
    lig_polar = lr["donor"] | lr["acceptor"]
    prot_polar = pr["donor"] | pr["acceptor"]
    mask = np.zeros(len(lig_xyz), dtype=bool)
    n_bridge = 0
    for w in waters:
        dl = np.linalg.norm(lig_xyz - w["xyz"], axis=1)
        dp = np.linalg.norm(pxyz - w["xyz"], axis=1)
        li = np.where((dl <= WATER_CUT) & lig_polar)[0]
        pj = np.where((dp <= WATER_CUT) & prot_polar)[0]
        if len(li) == 0 or len(pj) == 0:
            continue
        for i in li:
            for j in pj:
                if np.linalg.norm(lig_xyz[i] - pxyz[j]) <= WATER_CUT:
                    continue
                vl = lig_xyz[i] - w["xyz"]
                vp = pxyz[j] - w["xyz"]
                cos = float(np.dot(vl, vp) /
                            (np.linalg.norm(vl) * np.linalg.norm(vp) + 1e-9))
                angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
                if angle < MIN_ANGLE or angle > MAX_ANGLE:
                    continue
                comp = ((lr["donor"][i] and pr["acceptor"][j])
                        or (lr["acceptor"][i] and pr["donor"][j]))
                if comp:
                    mask[i] = True
                    n_bridge += 1
    return mask, n_bridge


def ligand_water_features(lig_xyz, waters):
    n = len(lig_xyz)
    if not waters:
        return np.zeros((n, 6), dtype=np.float32)
    wxyz = np.asarray([w["xyz"] for w in waters], dtype=np.float32)
    wocc = np.asarray([w["occ"] for w in waters], dtype=np.float32)
    d = np.linalg.norm(lig_xyz[:, None, :] - wxyz[None, :, :], axis=2)
    out = np.zeros((n, 6), dtype=np.float32)
    out[:, 0] = (d <= 3.5).sum(axis=1)
    out[:, 1] = (d <= 4.5).sum(axis=1)
    out[:, 2] = (d <= 6.0).sum(axis=1)
    m45 = d <= 4.5
    out[:, 3] = (wocc[None, :] * m45).sum(axis=1)
    out[:, 4] = d.min(axis=1)
    out[:, 5] = np.where(m45.any(axis=1),
                         (wocc[None, :] * m45).sum(axis=1) /
                         np.maximum(1, m45.sum(axis=1)), 0.0)
    return out


def pocket_water_features(pocket_xyz, waters):
    m = len(pocket_xyz)
    if not waters:
        return np.zeros((m, 4), dtype=np.float32)
    wxyz = np.asarray([w["xyz"] for w in waters], dtype=np.float32)
    wocc = np.asarray([w["occ"] for w in waters], dtype=np.float32)
    d = np.linalg.norm(pocket_xyz[:, None, :] - wxyz[None, :, :], axis=2)
    out = np.zeros((m, 4), dtype=np.float32)
    out[:, 0] = (d <= 3.5).sum(axis=1)
    out[:, 1] = (d <= 4.5).sum(axis=1)
    m45 = d <= 4.5
    out[:, 2] = (wocc[None, :] * m45).sum(axis=1)
    out[:, 3] = d.min(axis=1)
    return out
