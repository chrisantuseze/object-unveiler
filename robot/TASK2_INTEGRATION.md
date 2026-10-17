# Unveiler real-robot evaluation: Task 2 (ordered two-target retrieval) and an "always grasp the target" baseline

This guide is for a session working in **dofbot-controller** on the DOFBOT Pro's Jetson. It extends the existing
Unveiler episode runner, `unveiler/unveiler_session.py`, which was built from `CLIENT_INTEGRATION.md` (read that
first). **The lab-PC server and the wire protocol do not change.** Everything below lives on the Jetson.

It covers two changes:

1. **Task 2:** retrieve two targets in a given order, e.g. "the green block, then the red block". Both start covered.
2. **The `always_target` method**, for Task 1 and Task 2. It skips the server and grasps the current target
   straight away. It is the baseline every method has to beat, and it takes a few lines.

The Jetson runs Python 3.8, so avoid `X | None` annotations and other newer syntax. Keep Task 1 working exactly as
it does now: an episode with one target must behave identically.

## 1. Task 2 definition

- The scene has 3-6 blocks of distinct colours. Two of them are targets, given in order, `targets: [A, B]`.
- **Success:** A reaches the bin, then B reaches the bin, within the step budget. Picking B before A is an
  `order_violation` and ends the episode as a failure.
- Obstacles are anything that is not the current target. While working on A, B counts as an obstacle only if it
  lies on A (see scene rule 3).
- The step budget is shared across both targets: `max_steps = n_blocks + 2 * len(targets)`.
- The server answers the same question it answers for Task 1: which object to remove next to reach *this* target.
  The runner calls it with A's mask until A is retrieved, then with B's mask. Removing blocks while retrieving A
  changes the pile B sits in, which is what this task measures.

### Scene rules (put these in the paper)

1. Both targets are covered at the start. A is fully or partly covered from above, and so is B.
2. Every block has a unique colour among red, green, blue and yellow. Tape over the AprilTags: the segmenter treats
   them as separate objects.
3. Never place B on top of A. Otherwise the ordered task is impossible without violating the order. A may lie on B:
   that is the interesting case, because retrieving A then also uncovers B.
4. Build each scene from a layout photo, so every method sees the same layout.

## 2. `scenes.yaml`

Add an optional `targets` list. Old entries with `target:` keep working as Task 1.

```yaml
# Task 1 (unchanged)
- {id: S01, density: "3-4", occlusion: full, target: green, blocks: [green, red, blue, yellow]}
# Task 2
- {id: T01, task: ordered, targets: [green, red], blocks: [green, red, blue, yellow]}
- {id: T02, task: ordered, targets: [blue, yellow], blocks: [blue, yellow, red, green]}
```

In the runner:

```python
def scene_targets(scene):          # both formats -> list of colours
    return list(scene["targets"]) if "targets" in scene else [scene["target"]]
```

## 3. Runner changes (`unveiler/unveiler_session.py`)

### 3.1 Reference masks for every target

Task 1 captures one reference mask before the occluders go on. Task 2 needs one per target:

```
Operator: "Scene T01. Place ONLY the targets (green, red), uncovered, press Enter."
ref_frame = robot.capture_frame()
ref_masks = {c: target_mask_from_colour(ref_frame, c) for c in targets}   # all must be found; else ask again
Operator: "Build the rest of the scene (layout photo results/<session>/layouts/T01.jpg), press Enter."
```

### 3.2 The step loop, one pass per target

This replaces the single-target loop. With one target it reduces to the Task 1 loop.

```python
step = 0
per_target = []                                   # one record per target
for k, colour in enumerate(targets):
    later = targets[k + 1:]                       # targets still to come
    rec = {"target": colour, "retrieved": False, "steps": 0, "failure_cause": None}
    while step < max_steps:
        frame = robot.capture_frame()
        live = target_mask_from_colour(frame, colour)
        visible = live is not None
        if visible:
            ref_masks[colour] = live              # keep the latest view as the fallback reference

        # later targets must not be picked early: exclude them from the choice
        reachable = None
        if later:
            reachable = np.full(frame.shape[:2], 255, np.uint8)
            for c in later:
                m = target_mask_from_colour(frame, c)
                if m is not None:
                    reachable[m > 0] = 0          # objects whose centroid falls here are not selectable

        if method == "always_target":             # section 4
            r = {"chosen_index": 0, "is_target": True, "chosen_mask_np": live}
            if not visible:
                rec["failure_cause"] = "target_hidden"; break
        else:
            r = client.select(frame, live if visible else ref_masks[colour], method=method, step=step,
                              target_visible=visible, reachable_mask=reachable, return_overlay=True)
            save frame + overlay under ep_NNN/step_SS/
            if r["chosen_index"] < 0:
                rec["failure_cause"] = "no_selection: " + r["reason"]; break

        chosen = colour if r["is_target"] else match_colour(r["chosen_mask_np"], frame)
        if chosen is None:
            rec["failure_cause"] = "colour_mismatch"; break
        if chosen in later:                       # should not happen with the reachable mask; guard anyway
            rec["failure_cause"] = "order_violation"; break

        ok = execute(chosen)                      # robot / manual / none, exactly as in Task 1
        step += 1; rec["steps"] += 1
        record the step: {target: colour, chosen_colour: chosen, is_target: chosen == colour, exec_ok: ok,
                          probs, unfiltered_index, timing_ms, log_dir}
        if chosen == colour:
            rec["retrieved"] = ask_operator(f"Is the {colour} block in the bin? [y/n]") if executor != "none" else ok
            if rec["retrieved"]:
                break                             # on to the next target
            rec["failure_cause"] = "grasp_failed" # target still there: keep trying within the budget
    else:
        rec["failure_cause"] = rec["failure_cause"] or "step_limit"
    per_target.append(rec)
    if not rec["retrieved"]:
        break                                     # the order is broken: stop the episode

success = len(per_target) == len(targets) and all(t["retrieved"] for t in per_target)
```

Notes:

- `reachable_mask` is already part of the protocol. The server drops any object whose centroid lies outside the
  mask from every method's choice, including the SRE, the heuristic, GPT-4o and random. So masking the later
  targets keeps all methods from grasping them early, with no server change. If a later target is the only thing
  covering the current one, which scene rule 3 forbids, the server returns `chosen_index = -1` and the episode logs
  `no_selection`.
- `client.reset(session, episode)` is called once per episode, as now. The step numbers keep counting across both
  targets, so the server's logs stay in one folder per episode.

### 3.3 Episode record (`episodes.jsonl`)

This extends the Task 1 record. Task 1 episodes simply have one entry in `targets`.

```json
{"episode": 12, "scene": "T01", "task": "ordered", "method": "sre_il", "targets": ["green", "red"],
 "per_target": [{"target": "green", "retrieved": true, "steps": 2, "failure_cause": null},
                {"target": "red", "retrieved": false, "steps": 3, "failure_cause": "step_limit"}],
 "success": false, "first_target_success": true, "n_steps": 5, "steps": [...],
 "real_success": false, "note": "", "executor": "robot", "started": "...", "duration_s": 301.2}
```

`real_success` is still the operator's final label, as in Task 1: both targets in the bin, in order.

### 3.4 `summary.md`

For Task 2 scenes, report per method:

- both-in-order success (k/n)
- first-target success (k/n)
- mean steps over successful episodes
- failure causes, each attributed to the target it happened on (e.g. `red: step_limit x2`)

Keep the Task 1 table as it is, and add `always_target` as a row.

## 4. The `always_target` method

Add `always_target` to the accepted `--methods`. It never calls the server: the runner grasps the current target's
colour at every step, as the sketch in 3.2 shows. If the target is not visible, the step fails with `target_hidden`,
because a hidden target cannot be grasped directly. In Task 1 it answers "how often is the target directly
graspable anyway?", and every learned method must beat it on covered scenes. On the lab PC's offline replays,
covered scenes put it at 5/25.

## 5. Protocol

- **Methods:** `sre_il` (the server loads the expert-iteration weights, see below), `gpt4o`, `heuristic`,
  `always_target`. Add `random` if time allows.
- **Scenes:** 10 Task 2 scenes, 3-4 blocks each with both targets covered. Add 4-6 block scenes with the
  pixel-driven pick if it exists (Option B in `CLIENT_INTEGRATION.md`).
- **Order:** rotate the method order per scene, as in Task 1.
- **Server (lab PC), for reference.** Use `--device cpu` when the GPU is busy; it answers in about 2 s.
  ```
  python -m robot.server --jetson-ip <JETSON_IP> --warp 88 71 501 38 639 479 54 479 \
      --sre-il-ckpt save/sre_exit_top/sre_exit_it0.pt [--device cpu]
  ```

## 6. Bring-up checks

1. `--executor none` on one Task 2 scene with `heuristic`. Both targets are processed in order. Later targets
   never appear as the chosen object; confirm with the server overlays. `episodes.jsonl` has `per_target`.
2. `--executor none` with `always_target`: no server calls at all, and the episode ends at the first hidden target.
3. `--executor manual` on one scene: the operator prompts appear once per target.
4. `--executor robot` on one 3-block scene with `heuristic`.
5. Run one Task 1 scene again and diff its `episodes.jsonl` fields against an old record: nothing may change for
   single-target episodes except the added `targets` and `per_target` fields.
