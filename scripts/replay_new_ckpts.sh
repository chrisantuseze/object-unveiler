#!/usr/bin/env bash
# As each top-mode expert-iteration checkpoint appears, replay it (CPU) on both labelled real scene sets and save the
# scores: save/real_eval/<set>/replay_exit_it<N>/score.txt
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/miniforge3/envs/unveiler/bin/python}
WARP="88 71 501 38 639 479 54 479"
for N in 1 2 3; do
  CK=save/sre_exit_top/sre_exit_it$N.pt
  until [ -f "$CK" ] && grep -q "iteration $N: trained" save/sre_exit_top/run.out; do sleep 60; done
  for SET in offline_check_ offline_check; do
    S=save/real_eval/$SET
    O=$S/replay_exit_it$N
    mkdir -p "$O" && cp "$S/replay/labels.csv" "$O/"
    $PY -m robot.replay_offline --session-dir "$S" --out "$O" --device cpu --warp $WARP \
        --methods heuristic --il-ckpts save/sre_exit_top/sre_exit_it0.pt "$CK" > "$O/replay.log" 2>&1
    $PY -m robot.replay_offline --session-dir "$S" --out "$O" --score > "$O/score.txt" 2>&1
  done
done
