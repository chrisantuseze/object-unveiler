#!/usr/bin/env bash
# Waits for scripts/train_sre_exit_all.sh to finish, then runs the simulation selector table (eval_selectors.py)
# on the same scenes for every selector. Results: save/selector_eval/<bin>_<model>/episodes.jsonl + summaries.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/miniforge3/envs/unveiler/bin/python}
while pgrep -f "[t]rain_sre_exit_all.sh" > /dev/null; do sleep 60; done

for BIN in "6 9" "9 13"; do
  TAG=${BIN/ /_}
  # baselines + the expert-iteration SRE (loaded through the sre_il selector)
  $PY eval_selectors.py --nr_objects $BIN --n_scenes 30 --render egl --out save/selector_eval/${TAG}_exit \
      --selectors oracle sre_il heuristic planner nearest random --sre_model save/sre_exit/sre_exit_best.pt \
      >> save/selector_eval/overnight.log 2>&1
  # the IL SRE and the PPO SRE on the same 30 scenes (same seed)
  $PY eval_selectors.py --nr_objects $BIN --n_scenes 30 --render egl --out save/selector_eval/${TAG}_il_ppo \
      --selectors sre_il sre --sre_model save/sre/sre_model_best.pt --sre_rl save/sre_rl/sre_rl_best.pt \
      >> save/selector_eval/overnight.log 2>&1
done
for d in save/selector_eval/*_exit save/selector_eval/*_il_ppo; do
  $PY eval_selectors.py --summarize "$d" > "$d/summary.txt" 2>&1
done
echo "overnight sim eval done $(date)" >> save/selector_eval/overnight.log
