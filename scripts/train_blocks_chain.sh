#!/usr/bin/env bash
# Overnight chain: (1) top-down SRE trained on block scenes that look like the real ones (the robot model),
# (2) resume the side model for the sim table, (3) the simulation selector table.
# A separate watcher (scripts/replay_new_ckpts.sh save/sre_exit_blocks "0 1 2 3") scores each block checkpoint on the
# real frames as it appears.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/miniforge3/envs/unveiler/bin/python}
mkdir -p save/sre_exit_blocks save/sre_exit
$PY -m trainer.train_sre_exit --access top --objects_set blocks --densities 3-5,4-7,5-8 \
    --out save/sre_exit_blocks --iterations 4 --states_per_iter 2500 --workers 3 >> save/sre_exit_blocks/run.out 2>&1
$PY -m trainer.train_sre_exit --access side --out save/sre_exit --iterations 4 --states_per_iter 2500 \
    --workers 3 >> save/sre_exit/run.out 2>&1
scripts/overnight_sim_eval.sh
