#!/usr/bin/env bash
# Expert iteration (AlphaZero style) in the sheet-less twin: iteration 0 rolls out the search, later iterations roll
# out the SRE and relabel with the search (twin/unveil.py --iterations). Each checkpoint is replayed on the real
# sets. Waits for scripts/train_twin_chain.sh (the first, single-round run) and scores its checkpoint on the
# sheet-less real scenes too.
#   scripts/train_twin_iter.sh [OUT (default save/sre_twin2)] [STATES per iteration (default 4000)] [ITERATIONS (3)]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/miniforge3/envs/unveiler/bin/python}
OUT=${1:-save/sre_twin2}
STATES=${2:-4000}
ITERS=${3:-3}
export SETS="no_sheet offline_check offline_check_"
while pgrep -f "[t]rain_twin_chain.sh" > /dev/null; do sleep 60; done
[ -f save/sre_twin/sre_exit_it0.pt ] && SETS="no_sheet" scripts/replay_new_ckpts.sh save/sre_twin "0"
mkdir -p $OUT
$PY -m twin.unveil --out $OUT --states $STATES --iterations $ITERS --workers 3 >> $OUT/run.out 2>&1 &
scripts/replay_new_ckpts.sh $OUT "$(seq -s ' ' 0 $((ITERS - 1)))"
wait
