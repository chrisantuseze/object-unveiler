#!/usr/bin/env bash
# Repeat the twin expert-iteration run (scripts/train_twin_iter.sh) with other seeds, one after the other, to see
# how much the real-frame scores move between runs.   scripts/train_twin_seeds.sh "1 2"
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/miniforge3/envs/unveiler/bin/python}
export SETS="no_sheet offline_check offline_check_"
for SEED in ${1:-"1 2"}; do
  OUT=save/sre_twin2_s$SEED
  mkdir -p $OUT
  $PY -m twin.unveil --out $OUT --states 4000 --iterations 3 --workers 3 --seed $SEED >> $OUT/run.out 2>&1 &
  scripts/replay_new_ckpts.sh $OUT "0 1 2"
  wait
done
