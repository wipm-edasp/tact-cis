"""Run PLIP on one hydrogenated PDB and emit ligand water-bridge atom coords."""
import json
import subprocess
import sys
import tempfile

from plip.structure.preparation import PDBComplex

pdb, ccd, out = sys.argv[1], sys.argv[2], sys.argv[3]
tmp = tempfile.mktemp(suffix="_h.pdb")
subprocess.run(["/home/xiao/.conda/envs/scaffold_vae/bin/obabel", pdb, "-O",
                tmp, "-h", "-p", "7.4"], check=True, capture_output=True)
c = PDBComplex()
c.load_pdb(tmp)
c.analyze()
key = next((k for k in c.interaction_sets if k.split(":")[0] == ccd), None)
atoms = []
n = 0
if key is not None:
    site = c.interaction_sets[key]
    n = len(site.water_bridges)
    for w in site.water_bridges:
        if w.restype_l == ccd:
            a = w.a
        elif w.restype == ccd:
            a = w.d
        else:
            continue
        atoms.append([float(a.coords[0]), float(a.coords[1]), float(a.coords[2])])
with open(out, "w") as fh:
    json.dump({"atoms": atoms, "n_bridges": n}, fh)
