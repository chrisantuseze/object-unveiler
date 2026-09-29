# SRE training and real-robot evaluation plan

Status as of 2026-09-29. This covers why the PPO fine-tuning failed, what replaces it, which real-robot tasks to run, and
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

This is the Task 1 capture: in 20 of 25 scenes a block covers the target from above, and 5 are free-target
controls. The labels are drafts written from `replay/scenes/*.jpg` (`draft (Claude)` in the notes) and need
checking, especially episodes 22 and 27. Expert iteration is iteration 0 of the top model after the loss-weighting
fix.

| Method | Correct | 95% interval | 2-4 objects | 5-8 objects |
| --- | --- | --- | --- | --- |
| **SRE, expert iteration (it0)** | **15/25** | **40-76%** | 7/12 | **8/13** |
| GPT-4o | 7/25 | 12-48% | 4/12 | 3/13 |
| SRE, PPO (`sre_rl_best.pt`) | 6/25 | 8-40% | 5/12 | 1/13 |
| heuristic (Alg. 1) | 6/25 | 8-44% | 4/12 | 2/13 |
| SRE, imitation only | 5/25 | 4-36% | 4/12 | 1/13 |
| always grasp the target | 5/25 | 4-36% | 3/12 | 2/13 |
| random | 4/25 | 4-32% | 4/12 | 0/13 |

The expert-iteration SRE is the only method well above the trivial baselines, and it keeps its accuracy in denser
scenes. On the easy set, the same checkpoint scores 11/20 against 15/20 for "always grasp". Later iterations should
be checked on both sets, so a gain on covered scenes does not hide a loss on free targets.

### Running overnight (started 2026-09-29, about 15:30)

- `scripts/train_sre_exit_all.sh`: the top model (iterations 1-3), then the side model (4 iterations).
- `scripts/replay_new_ckpts.sh`: replays each new top checkpoint on both real sets on CPU. Scores are written to
  `save/real_eval/<set>/replay_exit_it<N>/score.txt`.
- `scripts/overnight_sim_eval.sh`: once training ends, runs the simulation selector table (30 scenes per bin, 6-9
  and 9-12 objects). It covers the expert-iteration SRE, the IL and PPO SREs, the heuristic, the planner, nearest,
  random and the oracle. Summaries go to `save/selector_eval/*/summary.txt`.

## 5. Next steps

- [x] Label the first set (checked by Chris) and capture a covered set (episodes 21-45).
- [ ] Check the draft labels for the covered set (`save/real_eval/offline_check/replay/labels.csv`), then run
      `python -m robot.replay_offline --session-dir save/real_eval/offline_check --score`.
- [ ] Read the overnight scores. Pick the top checkpoint that is best on the covered set without losing free
      targets, and use it on the robot (`--sre-il-ckpt`, method `sre_il`).
- [ ] Run the Task 1 robot protocol with the chosen checkpoint, GPT-4o, the heuristic and "always grasp the target".
      The AprilTags still need covering: they cost several scenes in both sets.
- [ ] Add the Task 2 changes to the Jetson runner; run Task 2 with the two or three best methods.
- [ ] Once the side model finishes, run the simulation selector table for the paper.
