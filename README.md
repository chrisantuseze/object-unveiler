# Unveiler

Code for **Learning Object-Centric Spatial Reasoning for Sequential Manipulation in Cluttered Environments**.

Chrisantus Eze, Ryan C. Julian, Christopher Crick

[Project page](https://object-unveiler.github.io) · [arXiv](https://arxiv.org/abs/2603.02511)

![Unveiler retrieving a covered target on a Dofbot-Pro](https://object-unveiler.github.io/static/images/real_external_task1.png)

Unveiler retrieves a target object from clutter by deciding *which* object to remove separately from *how* to
manipulate it:

- a **Spatial Relationship Encoder (SRE)** looks at the segmented objects and picks the next one to remove (or the
  target itself, once it is free);
- an independent **Action Decoder** turns that choice into an orientation-discretized push-grasp action.

The two modules talk only through a discrete object index, so the SRE can be trained in simulation and paired with a
different execution policy on a real robot. The SRE is trained in two stages: imitation of a heuristic, then
search-based policy improvement (expert iteration) using simulated object removals. No real-robot training data is
used.

## Repository layout

| path | what it is |
|---|---|
| `main.py` | entry point for training (`--mode sre`, `sre-rl`, `ae`, `fcn`, `reg`) and simulation evaluation (`--mode eval`) |
| `collect_data.py` | collects heuristic demonstrations in PyBullet |
| `policy/` | SRE (`sre_model.py`, `sre_actor_critic.py`), Action Decoder and grasping models |
| `trainer/` | training loops, including expert iteration (`train_sre_exit.py`) |
| `env/` | PyBullet environment and cameras |
| `mask_rg/` | Mask R-CNN object segmentation |
| `eval_agent.py`, `eval_agent_target.py` | full-system simulation evaluation |
| `eval_selectors.py` | selection-only harness: same scenes, segmentation and Action Decoder for every selection method |
| `baseline/` | GPT-4o and CLIP selection baselines |
| `twin/` | MuJoCo digital twin of the real setup: training scenes, selection-only evaluation, printable evaluation layouts |
| `robot/` | real-robot server (segmentation + object selection over rosbridge); see [`robot/README.md`](robot/README.md) |
| `scripts/` | training/evaluation chains and the paper figure and video scripts |
| `docs/` | lab record and real-robot evaluation commands |
| `act/` | ACT policy experiments |

## Setup

```bash
conda create -n unveiler python=3.10
conda activate unveiler
pip install -r requirements.txt
```

The digital twin needs MuJoCo with EGL (`MUJOCO_GL=egl`). The GPT-4o baseline needs `OPENAI_API_KEY`.

Checkpoints and datasets are not in the repository. The code expects them under `save/` and `downloads/`
(for example `save/sre/sre_model_best.pt`, `save/ae/ae_model_best.pt`, `save/fcn/fcn_model_best.pt`,
`downloads/reg_model.pt`), both of which are gitignored.

## Simulation

Collect heuristic demonstrations:

```bash
python3 collect_data.py --singulation_condition --n_samples 30000 --chunk_size 5 --seed 1
```

Train the Action Decoder and the SRE by imitation:

```bash
python3 main.py --mode ae  --dataset_dir save/ou-dataset --epochs 100 --batch_size 2 --lr 0.001
python3 main.py --mode sre --dataset_dir save/ou-dataset
```

Improve the SRE with expert iteration (search over simulated removals relabels the states the policy visits):

```bash
scripts/train_sre_exit_all.sh                 # full run; resumes from saved shards
python -m trainer.train_sre_exit --probe 30   # check the graspability test on a few scenes
```

Evaluate the full system:

```bash
python3 main.py --mode eval \
    --reg_model downloads/reg_model.pt \
    --fcn_model save/fcn/fcn_model_best.pt \
    --ae_model save/ae/ae_model_best.pt \
    --sre_model save/sre/sre_model_best.pt \
    --n_scenes 30 --chunk_size 5 --temporal_agg --seed 1
```

Compare selection methods with everything else held fixed:

```bash
python eval_selectors.py --nr_objects 6 9 --n_scenes 30 --render egl --out save/selector_eval/6_9 \
    --selectors oracle sre_il heuristic planner nearest random --sre_model save/sre_exit/sre_exit_best.pt
```

## Digital twin

`twin/` renders the real setup (the blocks, the table and the arm camera) in MuJoCo, and sends each render through
the same preprocessing the real robot uses, so the SRE trains on the tensors it will see on the robot.

```bash
python -m twin.unveil --preview 12 --out save/sre_twin/preview          # look at scenes first
python -m twin.unveil --out save/sre_twin --states 4000 --workers 3    # collect, then fine-tune the SRE
python -m twin.eval --out save/twin_eval/run --n_scenes 200 --il-ckpts save/sre_twin2/sre_exit_it0.pt
python -m twin.eval --summarize save/twin_eval/run
```

The fixed layouts used for the real-robot evaluation, with printable sheets, are in `twin/real_eval_layouts/`.

## Real robot

The lab PC runs segmentation and object selection; the robot's onboard computer runs the episode loop and its own
pick-and-place skill. They communicate over rosbridge. One server process serves one method:

```bash
python -m robot.server --jetson-ip <ROBOT_IP> --warp <8 corner coordinates> --method ours
```

Methods: `ours`, `il`, `gpt4o`, `clip`, `heuristic`, `random`, `always_target`, `ppo`.

- [`robot/README.md`](robot/README.md): server options, wire protocol and offline testing
- [`docs/real_eval_commands.md`](docs/real_eval_commands.md): the commands used for the evaluation
- [`docs/sre_training_and_real_eval.md`](docs/sre_training_and_real_eval.md): the lab record, with training
  details, offline checks and per-task results

## Citation

```bibtex
@article{eze2026unveiler,
  title   = {Learning Object-Centric Spatial Reasoning for Sequential
             Manipulation in Cluttered Environments},
  author  = {Eze, Chrisantus and Julian, Ryan C. and Crick, Christopher},
  journal = {arXiv preprint arXiv:2603.02511},
  year    = {2026}
}
```
