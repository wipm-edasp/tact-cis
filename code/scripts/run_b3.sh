#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=code/src
PY="${PY:-python}"

for spec in "data/dataset_cod.pkl:cis_b3_target" "data/dataset_cod_shift05.pkl:cis_b3_shift_target"; do
  data="${spec%%:*}"
  out="${spec##*:}"
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k "$PY" -m xact.experiment_cis_cv run \
      --data "$data" --outdir "results/$out" \
      --split target --shard "$k" --nshards 4 --inner-folds 4 --epochs 20 \
      > "results/${out}_${k}.log" 2>&1 &
  done
  wait
  "$PY" -m xact.experiment_cis_cv combine --outdir "results/$out" \
    > "results/${out}_combine.log" 2>&1
done
echo ALLDONE
