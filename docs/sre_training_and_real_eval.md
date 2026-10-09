# SRE training and real-robot evaluation plan

Status as of 2026-10-06. This covers why the PPO fine-tuning failed, what replaces it, which real-robot tasks to run, and
what the first offline check on real frames showed.

## 1. Why the PPO fine-tuning did not work

The PPO checkpoints drift towards a uniform policy over objects. That shows up in simulation and on real frames:

| Checkpoint | Mean top-1 probability, 20 real scenes (about 4 objects) | Mean top-1 minus top-2 |
| --- | --- | --- |
| `sre_il` (imitation only) | 0.73 | 0.53 |
| PPO episode 1000 | 0.41 | 0.16 |
| PPO episode 3000 | 0.34 | 0.08 |
| `sre_rl_best.pt` (about episode 6000) | 0.30 | 0.05 |
| PPO episode 9000 / 12000 | 0.25 | 0.01 |

Causes, in `trainer/train_sre_rl.py` and `utils/rl_rewards.py`:

1. **The return is mostly grasp noise.** Every sample is a full physics rollout with the Action Decoder, which fails
   50-70% of grasps whichever object was chosen. The selection signal is buried.
2. **The critic cannot learn a value.** It sees only the actor's own detached logits, not the scene, so the advantages
   are close to normalised noise.
3. **The entropy bonus is 0.05.** On top of near-zero advantages, that term dominates and flattens the policy.
4. **The shaped reward encodes the wrong notion of "blocked".** It uses top-down mask overlap and centroid lines,
   while the sim hand grasps from the side (and the real arm from the top, where overlap in a 2-D mask does not say
   what is on top).

Two related issues in the old evaluation path (`eval_agent.py`), worth checking before quoting Table II again:
success was labelled by hand at a `y/n` prompt, and `Policy.exploit_unveiler_rl` takes the aperture from `self.reg`,
whose weights are never loaded (`Policy.load` has that line commented out).

## 2. Replacement: expert iteration (`trainer/train_sre_exit.py`)

Expert iteration is the policy-improvement loop behind AlphaZero: a search over the true dynamics produces better
action targets than the current policy, the network is trained to reproduce them, and the network then drives data
collection for the next round.

**MDP.** State = the scene. Action = which segmented object to remove next; picking the target means "grasp it now".
Cost = 1 per action. Removal is ideal (the body is lifted out and physics settles), because selection is the SRE's job
and execution is the Action Decoder's. This is the same split as the paper's error decomposition (ε_SRE vs ε_exec).

**Expert = exact search in PyBullet.** `p.saveState`/`restoreState` lets the search try every candidate removal, let
the pile settle (so knock-on collapses and a toppled or displaced target count as failure) and test whether the
target is graspable. Q(s, i) = 1 + cost-to-go after removing i, by depth-limited search (depth 3) with memoisation.
Candidates are the objects that block the grasp or touch the target; removing anything else cannot help directly,
so it costs a wasted step.

**Graspability has two modes**, one per embodiment:

- `--access side` (the sim Barrett push-grasp): at least 40% of the 16 horizontal approach corridors are free. I
  checked this with `--probe` against real Action-Decoder grasps on 20 scenes. Targets with 7 or more of 16
  corridors free were grasped 45% of the time, those with fewer never. With "any corridor free" the test was useless
  (19/20 counted as free, 26% grasped).
- `--access top` (the real DOFBOT's top-down parallel grasp): no other object covers more than 10% of the target from
  above (vertical rays), and there is room for both fingers on at least one grasp axis. 60% of training scenes stack
  1-2 of the target's neighbours on top of it. Once the target is hidden, the reference mask from before stacking
  stands in for it, which matches the real protocol.

**Student = the SRE**, same architecture as `sre_model_best.pt` and initialised from it. It is trained with soft
cross-entropy to softmax(-Q / 0.3) on the Mask R-CNN masks it sees at test time. Iteration 0 rolls out the expert.
Iterations 1-3 roll out the student with probability 1 - β (β = 0.5, 0.25, 0.125), label every visited state with
the search, aggregate everything, and retrain (DAgger). Two thirds of the "target already graspable" states are
dropped, because they would otherwise be half the data.

**What changes for the paper.** The labels come from simulated consequences, not a hand-written ranking rule. That
answers the ICRA "it just imitates your heuristic" critique and gives the accessibility and stability claims
something concrete: the search models both. Present PPO as a negative result in one sentence, or leave it out.

### Running it

The script `scripts/train_sre_exit_all.sh` is already running: the top model first (4 iterations x 2500 states, 3
workers, roughly 2 h per iteration), then the side model. Both resume from saved shards after a restart. Only
states in saved shards survive a stop; each worker saves every 200 states.

```bash
tail -f save/sre_exit_top/run.out          # watch "optimal-choice" and "regret" on the validation split
tail -f save/sre_exit/run.out              # the side model, after the top model finishes
pkill -f train_sre_exit_all.sh; pkill -f "trainer.train_sre_exit"    # stop; rerun the script to continue
python -m trainer.train_sre_exit --probe 30                          # re-check the side graspability test
```

Outputs: `save/sre_exit_top/sre_exit_best.pt` (real robot) and `save/sre_exit/sre_exit_best.pt` (sim), plus one
checkpoint per iteration. Both load wherever an IL SRE checkpoint loads:

```bash
# real frames, offline (no arm time)
python -m robot.replay_offline --session-dir save/real_eval/<session> --warp <corners> \
    --sre-il-ckpt save/sre_exit_top/sre_exit_best.pt --out save/real_eval/<session>/replay_exit
# real robot: the server's `sre_il` method with the new weights
python -m robot.server --jetson-ip <IP> --warp <corners> --sre-il-ckpt save/sre_exit_top/sre_exit_best.pt
# simulation selector table
python eval_selectors.py --nr_objects 6 9 --n_scenes 30 --render egl --out save/selector_eval/6_9_exit \
    --selectors oracle sre_il heuristic planner nearest random --sre_model save/sre_exit/sre_exit_best.pt
```

Known limits. Sim objects are household meshes, while the real scenes use blocks. The finger-slot and coverage
thresholds are first guesses for the DOFBOT gripper and have not been measured.

## 3. Real-robot tasks

The real robot is the main evaluation. Keep one compact simulation table (the selector comparison at 6-9 and 9-12
objects), since training happens in simulation and the IROS AE asked for controlled baselines.

### Task 1 (default): retrieve a target occluded from above

This is the real-world counterpart of the simulation task. In simulation the hand grasps from the side, so occlusion
is lateral. The DOFBOT grasps from the top, so real occlusion means objects lying on the target.

- **Conditions.** *Partially covered* (the target is visible, but a block lies across it) and *fully covered* (the
  target is hidden; its mask comes from the reference photo taken before the occluders were placed). Use 3-4 blocks
  per scene, and 5-6 with the pixel-driven pick (Option B in `robot/CLIENT_INTEGRATION.md`).
- **The target must actually be covered in every scene.** In the first offline set, the target was already
  graspable in 15 of 20 scenes. On that set, "always grasp the target" scores 15/20 and beats every learned
  method (section 4).
- **Always report the "grasp the target now" baseline.** The heuristic (Algorithm 1) behaves almost exactly like it
  on 3-5 object scenes: for 3 or fewer objects it returns the target by construction.
- **Cover the AprilTags on the blocks.** Mask R-CNN segments them as separate objects. In the offline set, tags cost
  the SRE 4 of 20 scenes (episodes 9, 13, 16 and 17).
- **Methods:** `sre_il` loaded with the expert-iteration weights (main), IL SRE, heuristic, GPT-4o, random, always
  grasp the target.
- **Metrics:** success (k/n), steps, and failure cause (selection, segmentation, grasp), as in the integration guide.

### Task 2 (new): retrieve two targets in a given order

Example: "the green block, then the red block", where both are covered.

- **Why it is cheap.** The server and the models do not change. The SRE is simply called with the second target's
  mask once the first target is in the bin.
- **Jetson runner change (about 30 lines in `unveiler_session.py`).**
  - `scenes.yaml` takes `targets: [green, red]` instead of `target`.
  - A reference mask is captured for each target before the occluders go on.
  - The step loop runs once per target and carries over the step budget.
  - An episode succeeds only if both targets are retrieved in order.
- **What it adds.** The first retrieval changes the pile for the second, so a method that disturbs the scene pays for
  it on the second target.
- **Metrics:** success for both targets, success for the first target, and total steps.

## 4. First offline check on real frames (`save/real_eval/offline_check`)

These are 20 scenes captured with `--executor none` and replayed through every method with `robot/replay_offline.py`.
The captured episodes record what each method chose, not whether the choice was right. So `replay/labels.csv` holds
one label per scene: the indices (from `replay/scenes/*.jpg`) that should be removed next, or `t` when the target can
be grasped now. Blocking was judged top-down.

The labels were drafted from the images and checked by Chris; only episode 3 changed. Label with the numbers in
`replay/scenes/*.jpg`, not the capture's `overlay.jpg`: Mask R-CNN can order objects differently on the replay pass
(episode 3: red is 1 in the replay, 0 in the capture). Score with:

```bash
python -m robot.replay_offline --session-dir save/real_eval/offline_check --score
```

Accuracy against the draft labels (counted by hand from `replay.csv`; `--score` prints the same with 95% intervals):

| Method | Correct |
| --- | --- |
| always grasp the target | 15/20 |
| heuristic | 14/20 (0/5 on the scenes that need a removal) |
| PPO episode 12000 | 10/20 |
| PPO episode 9000 | 8/20 |
| `sre_rl_best.pt` | 8/20 |
| PPO episode 1000 | 6/20 |
| IL SRE | 5/20 |
| PPO episode 6000 | 5/20 |
| PPO episode 3000 | 4/20 |
| random | 2/20 |

Takeaways: this scene set cannot separate the methods (see Task 1), the tags hurt, and none of the current SRE
checkpoints is usable on the robot as is. The expert-iteration top model is the next thing to replay on these frames
and on a proper Task 1 set.

### Covered-scene set (`save/real_eval/offline_check`, episodes 21-45)

This is the Task 1 capture: in 21 of 25 scenes a block covers the target from above, and 4 are free-target
controls. The labels were drafted from `replay/scenes/*.jpg` and checked by Chris on 2026-10-05. Two changed:
episode 22 accepts either red 0 or blue 3, and in episode 27 the red cube is the obstacle to remove, not a free
target. Mask R-CNN gave that cube no mask, so episode 27 counts as wrong for every method (`correct = none`). The
file as Chris left it is kept as `replay/labels_as_checked_2026-10-05.csv`. Expert iteration is iteration 0 of the
top model after the loss-weighting fix.

| Method | Correct | 95% interval | 2-4 objects | 5-8 objects |
| --- | --- | --- | --- | --- |
| **SRE, expert iteration (it0)** | **15/25** | **40-76%** | 7/12 | **8/13** |
| GPT-4o | 7/25 | 12-48% | 4/12 | 3/13 |
| SRE, PPO (`sre_rl_best.pt`) | 6/25 | 8-40% | 5/12 | 1/13 |
| heuristic (Alg. 1) | 5/25 | 4-36% | 4/12 | 1/13 |
| SRE, imitation only | 5/25 | 4-36% | 4/12 | 1/13 |
| always grasp the target | 4/25 | 4-32% | 3/12 | 1/13 |
| random | 4/25 | 4-32% | 4/12 | 0/13 |

The expert-iteration SRE is the only method well above the trivial baselines, and it keeps its accuracy in denser
scenes. On the easy set, the same checkpoint scores 11/20 against 15/20 for "always grasp". Later iterations should
be checked on both sets, so a gain on covered scenes does not hide a loss on free targets.

### Later iterations on real frames

| Top checkpoint | Sim states | Easy set (20) | Covered set (25) |
| --- | --- | --- | --- |
| it0 | 2,505 | 11/20 | **15/25** |
| it1 | 3,809 | 12/20 | 12/25 |
| it2 | 5,877 | 11/20 | 7/25 |
| it3 | 8,380 | 12/20 | 7/25 |

(Covered-set counts use the checked labels.)

The DAgger iterations kept improving the fit to simulated states but made covered real scenes worse. That is a
sim-to-real gap: training scenes use household meshes, while the real scenes use blocks. **Use `sre_exit_it0.pt`
on the robot.** `sre_exit_best.pt` is only a copy of the last iteration (it3). The 15/25 is the best of eight
checkpoints scored on these 25 scenes, so part of its lead over it1 may be selection luck; the robot run is the
real test.

### Block-scene model (`save/sre_exit_blocks`, trained 2026-09-29)

The same top-down training on box meshes the size of the real blocks, in the real block colours on a white table
(`--objects_set blocks`). It did not help:

| Blocks checkpoint | Sim states | Easy set (20) | Covered set (25) |
| --- | --- | --- | --- |
| it0 | 1,666 | 4/20 | 5/25 |
| it1 | 3,500 | 8/20 | 8/25 |
| it2 | 5,567 | 7/20 | 6/25 |
| it3 | 6,601 | 10/20 | 6/25 |

On covered scenes the usual error is removing the wrong obstacle (12-14 of 25), not grasping the covered target
(3-4 of 25). The run was not a fair test of the block-scene idea, for three reasons found in its data:

1. **Uninformative labels.** In 17% of the states that need a removal (22% in it0), no segmented mask is a useful
   choice, so the soft label is uniform. The household-mesh run has 4%. Mask R-CNN misses some simulated blocks,
   and the target is hidden twice as often (18% against 9%).
2. **Sparse scenes.** States hold 3.8 blocks on average against 6.1 objects in the household-mesh run, although
   the densities asked for 3-8. Thirteen of the 25 covered real scenes have 5-8 objects.
3. **Lost data.** Collection workers crashed on a shared directory (`general_utils.recreate_train`, fixed on
   2026-10-05 with `exist_ok`). Iteration 0 lost one of three workers and iteration 3 most of two.

A rerun worth its 2.5 hours would drop the states in point 1, raise the densities to match the real scenes, and
use the fixed workers.

### Running overnight (started 2026-09-29, about 15:30)

- `scripts/train_sre_exit_all.sh`: the top model (iterations 1-3), then the side model (4 iterations).
- `scripts/replay_new_ckpts.sh`: replays each new top checkpoint on both real sets on CPU. Scores are written to
  `save/real_eval/<set>/replay_exit_it<N>/score.txt`.
- `scripts/overnight_sim_eval.sh`: once training ends, runs the simulation selector table (30 scenes per bin, 6-9
  and 9-12 objects). It covers the expert-iteration SRE, the IL and PPO SREs, the heuristic, the planner, nearest,
  random and the oracle. Summaries go to `save/selector_eval/*/summary.txt`.

### Simulation results (finished 2026-09-30; selection-only rerun 2026-10-05)

**With Action-Decoder grasps** (`save/selector_eval/{6_9,9_13}_{exit,il_ppo}`, 29 scenes per bin) the table cannot
separate the selectors. The Action Decoder fails 61-92% of valid grasps and the target often topples, so the
touch oracle scores the same as random at 6-9 objects (7/29 each).

| Selector | 6-9 objects | 9-13 objects |
| --- | --- | --- |
| oracle (highest object touching the target) | 7/29 | 7/29 |
| SRE, expert iteration (side model) | 9/29 | 7/29 |
| SRE, imitation only | 12/29 | 6/29 |
| SRE, PPO | 8/29 | 2/29 |
| heuristic | 8/29 | 4/29 |
| planner | 13/29 | 5/29 |
| nearest | 8/29 | 3/29 |
| random | 7/29 | 3/29 |

**Selection only** (`eval_selectors.py --execution ideal`, `scripts/sim_eval_ideal.sh`). The chosen object is
lifted out and the pile settles; the target is retrieved when the search's side-graspability test passes. This is
the MDP the expert-iteration SRE was trained on, so the comparison favours it by construction. `search` is that
MDP's optimum. Six steps allowed; 27 of 30 scenes at 6-9 objects are usable and all 27 need 1-3 removals.

| Selector (6-9 objects) | Success | 95% interval | First choice optimal | Grasped a blocked target |
| --- | --- | --- | --- | --- |
| search (optimum) | 27/27 | - | 100% | 0/27 |
| planner | 22/27 | 67-93% | 52% | 4/27 |
| oracle (touching) | 18/27 | 48-81% | 56% | 9/27 |
| SRE, imitation only | 14/27 | 33-70% | 67% | 7/27 |
| SRE, expert iteration (side model) | 13/27 | 30-67% | 56% | 14/27 |
| SRE, PPO | 10/27 | 19-56% | 52% | 8/27 |
| random | 15/27 (10/27 strict) | 37-74% | 44% | 13/27 |
| nearest | 7/27 | 11-44% | 48% | 3/27 |
| heuristic (Alg. 1) | 1/27 | 0-11% | 7% | 26/27 |

What it shows:

- **The side expert-iteration SRE is no better than the imitation-only SRE in simulation**, and the classical
  planner beats both. Only the heuristic and the search optimum are clearly separated from the rest at n = 27.
- **Its failures are all one kind.** In all 14 it grasps at the target while the search still counts it blocked.
  That step changes nothing, so a deterministic selector repeats it to the step limit; random escapes by chance.
  "Strict" ends the episode at such a grasp, which only changes random (15 to 10).
- **The heuristic almost always grasps the target at once** (26/27), as on the real covered set.
- **The 9-13 bin is unusable as run:** the search finds no solution within 3 removals in 22 of 30 scenes, leaving 7.
  It needs `--max_depth` raised in the search (slower) or a 6-9-only table.

## 5. Real2sim training in the DOFBOT twin (started 2026-10-05)

The sim-to-real gap above is in the scenes, not the labels, so the training scenes now come from a twin of the
real setup. `twin/` holds the MuJoCo twin from verify2act (`main-wm`): the real blocks (30x30x60 mm, four colours,
textures cut from real frames) on the wooden table, seen from the arm camera at the home pose. Changes for this
project (Chris, 2026-10-05): no AprilTags on the blocks, no white sheet, at most four blocks of different colours,
and no test-time real2sim bridge (the SRE sees real frames on the robot).

- **Camera.** Fitted to this project's `--warp` corners (`88 71 501 38 639 479 54 479`); the fit landed within 1 cm
  of verify2act's pose. Each render goes through the robot's preprocessing: the same fixed warp to the 400x400
  view, then Mask R-CNN at the robot's threshold.
- **Size check.** In that view a twin block's mask measures a median 146 x 70 px, a real one 155 x 73 px.
- **Scenes** (`twin/unveil.py`). 2-4 blocks; in 80% a block is dropped on the target (two blocks in 30% of
  those) and physics decides whether it stays on top, leans or slides off. The target mask is the reference view
  from before the occluders went on, or the current mask when the target is visible.
- **Labels.** The same search as section 2, in MuJoCo. Graspable from the top = no block covers more than 10% of
  the target and both fingers have about 12 mm of room beside it. States where the block to remove has no mask
  are dropped (the fault found in the block-scene run).
- **Table and light.** The table texture is cut from an empty patch of the real sheet-less frames and the
  brightness tuned to them: the SRE's view averages about 46 of 255 in the twin and 49 in the real frames.
- **Training** (expert iteration, the AlphaZero-style loop of section 2). Iteration 0 rolls out the search's
  choices. Iterations 1 and 2 roll out the SRE itself with probability 0.5 and 0.75, label every state it reaches
  with the search, add them to the data and retrain from the imitation-only SRE. 4,000 states per iteration:
  `scripts/train_twin_iter.sh save/sre_twin2 4000 3`. This is policy iteration with a search-based critic, so the
  paper can keep the imitation-then-RL structure; PPO's three faults (section 1) do not arise because the values
  are exact and the removal is ideal.

Limits. Mask R-CNN misses some dark blocks on the dark table, in the twin and on the real frames. The finger
clearance is a guess for the DOFBOT gripper; Chris's labels agree with it on the two real scenes that test it
(a block about 1 cm beside the target does not block it, a block touching it does).

### Real test set without sheet or tags (`save/real_eval/no_sheet`, 25 scenes)

Captured by Chris on 2026-10-05 (`save/real_eval/offline_check_no_white_sheet`, one arrangement per episode, four
blocks each). Labels drafted from `replay/scenes/*.jpg` and checked by Chris. 15 scenes need a removal and 10
have a free target. In two (episodes 5 and 25) the block to remove has no mask, so no method can be right; the
ceiling is 23/25.

| Method | Correct | 95% interval | Needs a removal (15) | Free target (10) |
| --- | --- | --- | --- | --- |
| **SRE, expert iteration in the twin, iteration 2** (`save/sre_twin2/sre_exit_it2.pt`) | **20/25** | 64-96% | 10 | 10 |
| SRE, expert iteration in the twin, iteration 0 | 19/25 | 60-92% | 9 | 10 |
| SRE, expert iteration on household meshes (`sre_exit_top` it0) | 16/25 | 44-84% | 9 | 7 |
| heuristic (Alg. 1) | 11/25 | 24-64% | 2 | 9 |
| always grasp the target | 10/25 | 24-60% | 0 | 10 |
| SRE, imitation only | 7/25 | 12-48% | 5 | 2 |
| SRE, PPO | 6/25 | 8-40% | 5 | 1 |
| GPT-4o | 5/25 | 8-36% | 3 | 2 |
| random | 3/25 | 0-28% | 3 | 0 |

Paired on the same scenes (exact sign test), the twin model beats the heuristic (9 scenes to 0, p = 0.004),
always-grasp (10 to 0, p = 0.002), the imitation-only SRE (14 to 1, p = 0.001), PPO and GPT-4o (p <= 0.001). Its
lead over the household-mesh model (5 to 1, p = 0.22) is not significant at this size. Iteration 2 is the
checkpoint to use because it is the last iteration, not because it scored highest: four twin checkpoints scored
19, 19, 20 and 20, which this set cannot tell apart.

The twin model's remaining errors on solvable scenes are all the same kind: it grasps at a target that a block
still partly covers (episodes 10, 15 and 22).

All twin checkpoints on the three real sets (the two older sets have the sheet and tags, which the twin no
longer models):

| Checkpoint | Twin look | Sheet-less (25) | Covered, sheet + tags (25) | Easy, sheet + tags (20) |
| --- | --- | --- | --- | --- |
| `sre_twin` it0 (6,000 states, one round) | first version, reddish table | 20 | 17 | 15 |
| `sre_twin2` it0 | matched to the real table | 19 | 21 | 13 |
| `sre_twin2` it1 | matched | 19 | 13 | 16 |
| `sre_twin2` it2 | matched | 20 | 17 | 15 |
| `sre_exit_top` it0 (household meshes) | - | 16 | 14 | 11 |

### Simulation table in the twin (`twin/eval.py`, `save/twin_eval/main`, 200 held-out scenes)

Each twin scene is rendered from the arm camera and handed to `robot/backend.py` as the Jetson would hand it over,
so every method runs through the robot's own code. The chosen block is lifted out and the pile settles; picking
the target succeeds when the search's top-grasp test passes, and a grasp at a still-blocked target fails the
episode. Four steps allowed. 155 scenes need a removal, 45 have a free target.

| Method | Success | 95% interval | Optimal episodes | First choice optimal |
| --- | --- | --- | --- | --- |
| search (the optimum, no perception) | 200/200 | - | 200 | 100% |
| **SRE, expert iteration in the twin, iteration 2** | **169/200** | 80-90% | **158** | 82% |
| SRE, expert iteration in the twin, iteration 1 | 166/200 | 78-88% | 154 | 80% |
| SRE, expert iteration in the twin, iteration 0 | 171/200 | 80-90% | 149 | 80% |
| SRE, twin, first version (6,000 states, one round) | 173/200 | 82-91% | 160 | 82% |
| SRE, imitation only | 128/200 | 57-70% | 52 | 37% |
| random | 118/200 | 52-66% | 41 | 33% |
| SRE, PPO | 115/200 | 50-64% | 49 | 34% |
| SRE, expert iteration on household meshes | 111/200 | 48-62% | 78 | 42% |
| heuristic (Alg. 1) | 70/200 | 28-42% | 69 | 37% |
| always grasp the target | 45/200 | 17-28% | 45 | 22% |

Read "optimal episodes" (success in the search's number of steps), not success alone: with at most four blocks,
removing blocks that are not in the way still ends in success, which is why random reaches 118. On optimal
episodes the twin models reach 149-160 of 200 and everything else 41-78.

- The twin checkpoints are within noise of one another here too: the later iterations do not add a measurable
  gain over iteration 0.
- Their failures are almost all grasps at a target that is still blocked (25-32 of 200), the same error as on
  the real frames.
- The twin models were trained in this twin, so this table favours them by construction; the real test set above
  is the independent check.

**Three seeds of the twin run** (`save/sre_twin2`, `save/sre_twin2_s1`, `save/sre_twin2_s2`; each collects its
own scenes). Correct choices on the real sets:

| Iteration | Sheet-less (25): seeds 0 / 1 / 2 | Mean | Covered, sheet + tags (25) | Easy, sheet + tags (20) |
| --- | --- | --- | --- | --- |
| 0 | 19 / 18 / 20 | 19.0 | 21 / 19 / 17 | 13 / 15 / 16 |
| 1 | 19 / 19 / 18 | 18.7 | 13 / 17 / 15 | 16 / 16 / 14 |
| 2 | 20 / 22 / 19 | 20.3 | 17 / 19 / 17 | 15 / 19 / 17 |

Every twin checkpoint scores 18-22 of 25 on the sheet-less set, above every other method (at most 16). The last
iteration averages 20.3 against 19.0 for the first; with 25 scenes and three seeds that difference is not
established. For the paper, report the final iteration as 20.3/25 (81%) averaged over three seeds, range 19-22.
`save/sre_twin2/sre_exit_it2.pt` (seed 0, 20/25) stays the robot checkpoint: it is the run the tables above
describe, and picking seed 1 for its 22 would be choosing on the test set.

**Tried: a confidence threshold on grasping the target** (`backend.target_min_prob`, off by default; replay
flag `--target-min-prob`). When the SRE's top choice is the target with probability below the threshold, the
backend takes its best other object. In the twin (`save/twin_eval/tau`, same 200 scenes) a threshold of 0.6
raised success from 169 to 177 and optimal episodes from 158 to 164; 0.9 gave 183 successes but only 135 optimal.
The threshold was fixed at 0.6 from the twin and then applied once to the 25 real scenes: 19/25, against 20/25
without it. No gain on real frames, so it stays off.

**Data repair, 2026-10-05.** The sheet-less capture session was also named `offline_check`, so the server wrote
its logs into `save/real_eval/offline_check/` and overwrote the frames of the covered set's episodes 21-25. The
new logs are now in `save/real_eval/no_sheet_server_logs/` and the five frames were restored from
`offline_check/replay/server_logs`. Since the restore the household-mesh model scores 14/25 on the covered set,
not 15: its choice in episode 23 changed, cause not found. Use a new session name for each capture.

## 6. Fixed scenes for the robot run (`twin/real_eval_layouts`, 2026-10-06)

As Verify2Act did, the robot scenes are built in the twin first and copied onto the table from a sheet, so every
method runs on the same arrangements. `python -m twin.make_eval_layouts` wrote 20 scenes and 5 spares: 10 with one
block on the target, 6 with two, and 4 with a free target. The protocol is in
`twin/real_eval_layouts/README.md` and the commands, one per baseline, in `docs/real_eval_commands.md`.

- A layout was kept only if the search's step count and best first choices held over 8 rebuilds with placement
  noise (4 mm, 5°). No model was run to choose the scenes, and the colours take turns as target and as cover.
- `--check <id> --frame <frame or session folder>` draws a layout's expected block edges on a real frame;
  `--check grid` draws the workspace rectangle, for marking the table now that the sheet is gone.
- `twin/eval.py --layouts` runs the same 20 scenes in the twin (`save/twin_eval/real_layouts`): twin SRE 18/20,
  imitation-only SRE 15, PPO 11, random 11, heuristic 9, always grasp 4. These are the per-scene predictions to
  set beside the robot results.
- The target is partly visible in all 16 covered eval scenes; from the arm camera a block lying on another rarely
  hides it fully.
- Task 2 has 10 scenes and 2 spares (`scenes_task2.yaml`): 6 with each target under its own block, 4 with one
  block across both targets. Not run in the twin.
- The method is now chosen on the lab PC: `python -m robot.server --method ours|il|gpt4o|clip|heuristic|always_target`
  serves that one method whatever the client asks for, and labels its replies and logs with it. `always_target`
  is a server method. Two changes are still to be made in the Jetson client: drop its method list, and end the
  episode at the first grasp at the target, with no retry (`docs/real_eval_commands.md`).

## 7. Robot results (`save/real_eval/results`, run 2026-10-07)

Four methods (`ours`, `il`, `gpt4o`, `heuristic`), each on the first 10 Task 1 scenes of `scenes.yaml` (5 C1, 3 C2,
2 free) and the 10 Task 2 scenes. CLIP and "always grasp the target" were dropped. Scenes C1-06 onward were not run.

**Success rule (strict, the same for every method):** the target ends in the bin, no block outside the twin
search's plan (`layouts.json`) was removed, and the target was not grasped while still covered. Task 1 was rescored
with this rule on 2026-10-07; the operator's original y/n is kept in `episodes.jsonl` (`real_success`) and shown as
"original label". **In bin** is the client's `success_auto`: the pick at the target succeeded, covered or not.

| Task 1 (n=10) | Success (strict) | 95% interval | Original label | In bin | First choice optimal | Mean steps |
|---|---|---|---|---|---|---|
| Ours (`task1_ours`) | 7 | 40-89% | 8 | 8 | 9 | 2.1 |
| IL | 4 | 17-69% | 6 | 8 | 5 | 2.6 |
| Heuristic | 4 | 17-69% | 4 | 9 | 7 | 1.7 |
| GPT-4o | 3 | 11-60% | 3 | 6 | 4 | 2.0 |

| Task 2 (n=10) | Both targets | 95% interval | First target | Mean steps |
|---|---|---|---|---|
| Ours | 9 | 60-98% | 9 | 3.4 |
| IL | 3 | 11-60% | 6 | 2.8 |
| Heuristic | 2 | 6-51% | 4 | 2.1 |
| GPT-4o | 2 | 6-51% | 4 | 2.7 |

**Table for the paper** (percentages, strict rule; the counts above are the lab record):

| Method | Task 1 success | Task 1 correct selections | Task 1 mean steps | Task 2 both targets | Task 2 correct selections | Task 2 mean steps |
|---|---|---|---|---|---|---|
| Ours | 70% | 85.7% (18/21) | 2.1 | 90% | 100% (34/34) | 3.4 |
| IL | 40% | 69.2% (18/26) | 2.6 | 30% | 75.0% (21/28) | 2.8 |
| Heuristic | 40% | 70.6% (12/17) | 1.7 | 20% | 71.4% (15/21) | 2.1 |
| GPT-4o | 30% | 45.0% (9/20) | 2.0 | 20% | 55.6% (15/27) | 2.7 |

**Correct selections** counts every selection a method made over the task's scenes. A selection is correct if it
removes a block that still covers a target (in Task 2, either target), or grasps the current target once nothing
covers it, judged by the twin search's plan in `layouts.json`. A request with no selection is wrong. It scores
the choice, not the grasp. Checked episode by episode on 2026-10-07. The paper must define it and say that
success is per episode over 10 scenes per task. On T2a-04 blue was not on red on the table (Chris, 2026-10-07), so taking red without removing blue
counts as correct for ours, IL and the heuristic. The edits the paper needs are listed in
`docs/paper_update_plan.md`.

- **Labels changed by the strict rule (Task 1):** ours C1-02 (blue, yellow, then green; yellow is not in the plan),
  IL C1-01 (blue removed, not in the plan) and IL C2-01 (yellow removed, and the client logged the target grasp as
  failed). Heuristic C1-04 follows the plan but stays a failure: the operator saw the grasp at the target fail.
- **Task 2 labels are unchanged.** No success there has an unneeded removal. On T2a-04 ours, IL and the heuristic
  took red without removing blue; blue was not on red on the table, so those are correct.
- **Task 2 separates the methods; Task 1 does not.** Fisher exact, two-sided: Task 2 ours against IL p = 0.020,
  against the heuristic and GPT-4o p = 0.006. Task 1 ours against IL and the heuristic p = 0.37, GPT-4o p = 0.18.
  Pooled over both tasks (a choice made after seeing the data): ours 16/20 (58-92%), IL 7/20 (p = 0.010),
  heuristic 6/20 (p = 0.004), GPT-4o 5/20 (p = 0.001).
- **"In bin" does not rank the methods.** The heuristic reaches 9/10 on Task 1 because the arm can often pull a
  covered block out from under another. The paper has to state the strict rule.
- **`task1_ours_old` is discarded.** Part of it ran before the fix to the robot camera's depth detection on
  stacked blocks, so ours was rerun on Task 1 as `task1_ours` (per Chris, 2026-10-07). The old run started first
  that night (23:05), before every other session.
- **Failures of ours.** Task 1: F-01 was a grasp failure at a free target; C1-02 had the unneeded removal; C2-03
  ended in `colour_ambiguous`, and failed for every method. Task 2: T2b-03 failed on the grasp at the first target.
- **Against the twin's prediction** for the same layouts (section 6: ours 18/20, IL 15, heuristic 9): the order
  holds on the robot.

## 8. Next steps

- [x] Real2sim training in the twin; expert iteration; a sheet-less real test set.
- [x] Fixed Task 1 scenes from the twin, with sheets and the scene list (section 6).
- [x] Robot Task 1 (first 10 scenes) and Task 2 (10 scenes) for ours, IL, GPT-4o and the heuristic (section 7).
- [ ] Update `paper/root.tex` with the robot results (`docs/paper_update_plan.md`). No more robot runs are
      planned; Task 1 scenes 11 to 20 stay unrun.
- [ ] Decide the paper's simulation table: the twin table (`twin/eval.py`, `save/twin_eval/main/summary.txt`) is
      the one that matches the real setup.
- [ ] Brighter light or a segmenter fine-tuned on twin renders, for the dark blocks Mask R-CNN misses.
