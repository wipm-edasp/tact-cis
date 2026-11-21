#!/bin/bash
# Rebuild the two CIS datasets. Set BENCH_CSV / PDB_DIR to the crystal benchmark
# manifest and PDB directory. The default paths are the workstation used here.
set -euo pipefail
cd "$(dirname "$0")/../.."
BENCH_CSV=${BENCH_CSV:-/home/xiao/workspace/drug-design/data/pdb_hops/ligands.csv}
PDB_DIR=${PDB_DIR:-/home/xiao/workspace/drug-design/data/pdb_hops/pdb}
PY=${PY:-python}
export PYTHONPATH=code/src

"$PY" -m xact.build --csv "$BENCH_CSV" --pdb-dir "$PDB_DIR" \
  --out data/dataset_cod.pkl
"$PY" -m xact.build --csv "$BENCH_CSV" --pdb-dir "$PDB_DIR" \
  --out data/dataset_cod_shift05.pkl --cutoff-shift 0.5
"$PY" -m xact.build --csv "$BENCH_CSV" --pdb-dir "$PDB_DIR" \
  --out data/dataset_cod_water.pkl --include-water
"$PY" -m xact.build --csv "$BENCH_CSV" --pdb-dir "$PDB_DIR" \
  --out data/dataset_cod_water_w1.pkl --include-water
