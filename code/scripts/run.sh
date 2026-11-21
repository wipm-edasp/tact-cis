#!/usr/bin/env bash
# TACT runner. Resolution order: $PYTHON, then the scaffold_vae conda env,
# then python on PATH.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${PYTHON:-}" ]; then
  if [ -x "/home/xiao/.conda/envs/scaffold_vae/bin/python" ]; then
    PYTHON="/home/xiao/.conda/envs/scaffold_vae/bin/python"
  else
    PYTHON="python"
  fi
fi
export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

DATA_CSV="${TACT_CSV:-/home/xiao/workspace/drug-design/data/pdb_hops/ligands.csv}"
DATA_PDB="${TACT_PDB:-/home/xiao/workspace/drug-design/data/pdb_hops/pdb}"

cmd="${1:-help}"
case "$cmd" in
  test)
    if "$PYTHON" -c "import pytest" >/dev/null 2>&1; then
      "$PYTHON" -m pytest -q "$HERE/tests"
    else
      "$PYTHON" "$HERE/tests/run_tests.py"
    fi
    ;;
  build)
    shift || true
    "$PYTHON" -m xact.build --csv "$DATA_CSV" --pdb-dir "$DATA_PDB" \
      --out "$HERE/results/dataset.pkl" "$@"
    ;;
  smoke)
    shift || true
    mkdir -p "$HERE/results"
    "$PYTHON" -m xact.smoke --csv "$DATA_CSV" --pdb-dir "$DATA_PDB" \
      --out "$HERE/results/smoke.json" "$@"
    ;;
  full)
    shift || true
    mkdir -p "$HERE/results/full"
    "$PYTHON" -m xact.experiment run \
      --data "$HERE/results/dataset.pkl" \
      --data-aware "$HERE/results/dataset_aware.pkl" \
      --csv "$DATA_CSV" --pdb-dir "$DATA_PDB" \
      --outdir "$HERE/results/full" "$@"
    ;;
  combine)
    "$PYTHON" -m xact.experiment combine --outdir "$HERE/results/full"
    ;;
  *)
    echo "usage: bash run.sh {test|build|smoke|full|combine} [extra args]"
    echo "  PYTHON=/path/to/python can override the interpreter"
    exit 1
    ;;
esac
