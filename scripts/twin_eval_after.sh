#!/usr/bin/env bash
# After the twin training chains finish: the selection-only table in the twin (twin/eval.py) for every twin
# checkpoint next to the earlier models and the baselines.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-~/miniforge3/envs/unveiler/bin/python}
while pgrep -f "[t]rain_twin_(chain|iter).sh" > /dev/null; do sleep 60; done
O=save/twin_eval/main
mkdir -p $O
CK=""
for f in save/sre_twin/sre_exit_it0.pt save/sre_twin2/sre_exit_it0.pt save/sre_twin2/sre_exit_it1.pt \
         save/sre_twin2/sre_exit_it2.pt save/sre_exit_top/sre_exit_it0.pt; do [ -f $f ] && CK="$CK $f"; done
$PY -m twin.eval --out $O --n_scenes 200 --il-ckpts $CK > $O/run.log 2>&1
$PY -m twin.eval --summarize $O > $O/summary.txt 2>&1
