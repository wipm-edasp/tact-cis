#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=code/src
PY="${PY:-python}"

for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k "$PY" -m xact.experiment_cis_cv run \
    --data data/dataset_cod_water.pkl --outdir results/cis_water_temporal \
    --split temporal --cutoff 2020-01-01 --seeds 11 22 33 44 55 \
    --shard "$k" --nshards 4 --inner-folds 4 --epochs 20 \
    > "results/cis_water_temporal_${k}.log" 2>&1 &
done
wait
"$PY" -m xact.experiment_cis_cv combine --outdir results/cis_water_temporal \
  > results/cis_water_temporal_combine.log 2>&1
echo ALLDONE
