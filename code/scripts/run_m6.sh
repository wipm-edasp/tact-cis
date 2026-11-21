#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=code/src
PY="${PY:-python}"

for split in target scaffold; do
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k "$PY" -m xact.experiment_cis_cv run \
      --data data/dataset_cod.pkl --outdir "results/cis_final_${split}" \
      --split "$split" --shard "$k" --nshards 4 --inner-folds 4 --epochs 20 \
      > "results/cis_m6_${split}_${k}.log" 2>&1 &
  done
  wait
  "$PY" -m xact.experiment_cis_cv combine --outdir "results/cis_final_${split}" \
    > "results/cis_m6_${split}_combine.log" 2>&1
done
echo ALLDONE
