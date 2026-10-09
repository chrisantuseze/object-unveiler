#!/usr/bin/env bash
# Real2sim fine-tune: covered-target scenes in the DOFBOT twin (twin/unveil.py), labelled by search, then the IL SRE
# fine-tuned on them, then the new checkpoint replayed on both labelled real scene sets (CPU).
#   scripts/train_twin_chain.sh [OUT (default save/sre_twin)] [STATES (default 6000)]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/miniforge3/envs/unveiler/bin/python}
OUT=${1:-save/sre_twin}
STATES=${2:-6000}
mkdir -p $OUT
$PY -m twin.unveil --out $OUT --states $STATES --workers 3 >> $OUT/run.out 2>&1
scripts/replay_new_ckpts.sh $OUT "0"
