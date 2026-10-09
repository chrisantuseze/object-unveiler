#!/usr/bin/env bash
# Selection-only simulation table (eval_selectors.py --execution ideal) for one density bin: the search optimum, the
# baselines and the expert-iteration SRE, then the IL and PPO SREs on the same scenes (same seed).
#   scripts/sim_eval_ideal.sh "6 9"     results: save/selector_eval/6_9_ideal{,_il_ppo}/summary.txt
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/miniforge3/envs/unveiler/bin/python}
BIN=${1:-"6 9"}
TAG=${BIN/ /_}
O=save/selector_eval/${TAG}_ideal
mkdir -p $O ${O}_il_ppo
$PY eval_selectors.py --execution ideal --nr_objects $BIN --n_scenes 30 --render egl --out $O \
    --selectors search oracle sre_il heuristic planner nearest random --sre_model save/sre_exit/sre_exit_best.pt \
    >> $O/run.log 2>&1
$PY eval_selectors.py --execution ideal --nr_objects $BIN --n_scenes 30 --render egl --out ${O}_il_ppo \
    --selectors sre_il sre --sre_model save/sre/sre_model_best.pt --sre_rl save/sre_rl/sre_rl_best.pt \
    >> ${O}_il_ppo/run.log 2>&1
for d in $O ${O}_il_ppo; do $PY eval_selectors.py --summarize $d > $d/summary.txt 2>&1; done
