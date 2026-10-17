#!/usr/bin/env bash
# As each expert-iteration checkpoint of a run appears, replay it (CPU) on both labelled real scene sets and save the
# scores: save/real_eval/<set>/replay_<run>_it<N>/score.txt
#   scripts/replay_new_ckpts.sh [RUN_DIR (default save/sre_exit_top)] [ITERATIONS (default "1 2 3")]
#   SETS="no_sheet" picks the real sets under save/real_eval/ (default: the two sheet-and-tag sets)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/miniforge3/envs/unveiler/bin/python}
RUN=${1:-save/sre_exit_top}
ITERS=${2:-"1 2 3"}
NAME=$(basename "$RUN")
WARP="88 71 501 38 639 479 54 479"
for N in $ITERS; do
  CK=$RUN/sre_exit_it$N.pt
  until [ -f "$CK" ] && grep -q "iteration $N: trained" "$RUN/run.out" 2>/dev/null; do sleep 60; done
  for SET in ${SETS:-offline_check_ offline_check}; do
    S=save/real_eval/$SET
    O=$S/replay_${NAME}_it$N
    mkdir -p "$O" && cp "$S/replay/labels.csv" "$O/"
    $PY -m robot.replay_offline --session-dir "$S" --out "$O" --device cpu --warp $WARP \
        --methods heuristic --il-ckpts save/sre_exit_top/sre_exit_it0.pt "$CK" > "$O/replay.log" 2>&1
    $PY -m robot.replay_offline --session-dir "$S" --out "$O" --score > "$O/score.txt" 2>&1
  done
done
