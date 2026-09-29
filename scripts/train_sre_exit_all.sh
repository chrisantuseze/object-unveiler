#!/usr/bin/env bash
# Expert-iteration SRE training: the top-down model (real DOFBOT) first, then the side model (sim push-grasp).
# Both resume from saved shards if restarted. Progress: tail -f save/sre_exit_top/run.out (then save/sre_exit/run.out)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/miniforge3/envs/unveiler/bin/python}
mkdir -p save/sre_exit_top save/sre_exit
$PY -m trainer.train_sre_exit --access top --out save/sre_exit_top --iterations 4 --states_per_iter 2500 \
    --workers 3 >> save/sre_exit_top/run.out 2>&1
$PY -m trainer.train_sre_exit --access side --out save/sre_exit --iterations 4 --states_per_iter 2500 \
    --workers 3 >> save/sre_exit/run.out 2>&1
