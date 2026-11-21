# Code

`src/xact` is the package; `scripts/` contains the run scripts and audits.
Install dependencies with `pip install -r requirements.txt` and set
`PYTHONPATH=code/src` from the repository root.

The scripts were written for a cluster layout and use `data/` for the two
dataset pickles; rebuild those first (see `../README.md`). Raw PDB
structures and `ligands.csv` are external inputs.

