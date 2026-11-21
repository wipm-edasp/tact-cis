#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=code/src
PY="${PY:-python}"

for split in temporal temporal_strict; do
  if [ "$split" = "temporal_strict" ]; then CUT=2018-01-01; else CUT=2020-01-01; fi
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k "$PY" -m xact.experiment_cis_cv run \
      --data data/dataset_cod_water_w1.pkl --outdir "results/cis_water_w1_${split}" \
      --split "$split" --cutoff "$CUT" --shard "$k" --nshards 4 --inner-folds 4 --epochs 20 \
      > "results/cis_water_w1_${split}_${k}.log" 2>&1 &
  done
  wait
  "$PY" -m xact.experiment_cis_cv combine --outdir "results/cis_water_w1_${split}" \
    > "results/cis_water_w1_${split}_combine.log" 2>&1
done
echo ALLDONE
