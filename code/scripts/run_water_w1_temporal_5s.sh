#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=code/src
PY="${PY:-python}"

for split in temporal temporal_strict; do
  if [ "$split" = "temporal_strict" ]; then CUT=2018-01-01; else CUT=2020-01-01; fi
  for k in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$k "$PY" -m xact.experiment_cis_cv run \
      --data data/dataset_cod_water_w1.pkl --outdir "results/cis_water_w1_${split}_5s" \
      --split "$split" --cutoff "$CUT" --seeds 11 22 33 44 55 \
      --shard "$k" --nshards 4 --inner-folds 4 --epochs 20 \
      > "results/cis_water_w1_${split}_5s_${k}.log" 2>&1 &
  done
  wait
  "$PY" -m xact.experiment_cis_cv combine --outdir "results/cis_water_w1_${split}_5s" \
    > "results/cis_water_w1_${split}_5s_combine.log" 2>&1
done
echo ALLDONE
