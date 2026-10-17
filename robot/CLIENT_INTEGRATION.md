# Unveiler real-robot evaluation: client integration for dofbot-controller

This guide is for a session working in **dofbot-controller** on the DOFBOT Pro's Jetson. It specifies the client
(robot) side of the Unveiler real-robot evaluation. The server side is finished and tested. It lives in
**object-unveiler/robot/** on the lab PC (`python -m robot.server`); see `robot/README.md` there.

## 1. Goal and constraints

Unveiler's contribution is the **Spatial Relationship Encoder (SRE)**, which picks the next obstacle to remove so an
occluded target can be grasped. The RA-L revision needs **end-to-end real-robot results**: task success and steps,
comparing the SRE with other *object-ordering* methods (heuristic, GPT-4o, CLIP, random) **on the same executor**.

- The executor is **Verify2Act's pick-and-place skill** (`lang_color_detect.py` + `lang_color_grasp.py`,
  `pick_place` into `left_bin`), *not* Unveiler's Action Decoder. The DOFBOT is not the floating hand the Action
  Decoder was trained for: its reach is short and it only grasps forward. The paper will say so.
- Budget: **1–2 weeks in total**, robot runs included. Reuse Verify2Act code; do not build new infrastructure.
- The Jetson runs Python 3.8 (ROS Noetic). Avoid `X | None` annotations and other syntax newer than 3.8.

## 2. Architecture

```
Jetson                                                     Lab PC (GPU)
  roscore, arm_driver, camera, rosbridge_websocket :9090   python -m robot.server --jetson-ip <JETSON_IP> --crop ...
  lang_color_detect.py, lang_color_grasp.py                  (connects OUT to the Jetson's rosbridge)
  unveiler/unveiler_session.py  (NEW, this guide)
     ├── RemoteRobotClient("127.0.0.1")   camera frames + execute_subtask(color)   [verify2act/v2a_robot_client.py]
     └── UnveilerClient("127.0.0.1")      select(frame, target_mask, method)      [copied from object-unveiler]
```

Both clients connect to the Jetson's own rosbridge at 127.0.0.1, which is how `v2a_session.py --jetson_ip 127.0.0.1`
already works. The lab PC never talks to ROS directly.

## 3. Files to add or remove in dofbot-controller

```
unveiler/
  __init__.py
  protocol.py          COPY verbatim from object-unveiler/robot/protocol.py
  client.py            COPY verbatim from object-unveiler/robot/client.py (it falls back to `from protocol import ...`)
  color_match.py       NEW: HSV masks and chosen-object -> colour matching (section 5)
  unveiler_session.py  NEW: the episode runner (section 6)
  scenes.yaml          NEW: the evaluation scene list (section 8)
  results/             created by the runner
```

- **Delete** `unveiler_inference_server.py` at the repo root. It is the old Action-Decoder server, now replaced by
  `object-unveiler/robot/server.py`.
- Leave `dofbot_pro_ws/src/dofbot_pro_info/scripts/unveiler_grasp.py` alone. It is the old Action-Decoder executor
  and is not used here.
- `client.py` tries `robot.protocol`, then `unveiler.protocol`, then `protocol`, so the copy needs no edits. Import it
  as `from unveiler.client import UnveilerClient` with the repo root on `sys.path` (the way `v2a_session.py` does it).

## 4. The protocol in brief

The full spec is the docstring of `protocol.py`. Topics `/unveiler/request` and `/unveiler/response` are
`std_msgs/String` carrying JSON.

```python
client = UnveilerClient("127.0.0.1")
client.ping()                                   # {methods: [...], crop, seg_threshold, ...}; TimeoutError if no server
client.reset(session="sre_S03", episode=3)      # starts a log folder on the lab PC
r = client.select(frame_bgr,                    # 640x480 BGR from RemoteRobotClient.capture_frame()
                  target_mask,                  # uint8 HxW, nonzero = target (same size as frame)
                  method="sre",                 # sre | sre_il | heuristic | gpt4o | clip | random
                  step=k,
                  target_visible=True/False,    # False = full occlusion: target_mask is the reference mask
                  reachable_mask=None,          # optional uint8 HxW, nonzero = graspable area
                  return_overlay=True)          # JPEG with outlines, for your own logs
r["chosen_index"]      # index into r["objects"]; -1 = nothing selectable, see r["reason"]
r["is_target"]         # True -> the method wants the target grasped now
r["chosen_mask_np"]    # uint8 0/255 mask of the chosen object at frame size (None if chosen_index == -1)
r["chosen_centroid"]   # [x, y] in frame pixels
r["target_visible"], r["target_index"], r["num_objects"], r["objects"], r["unfiltered_index"], r["probs"]
r["timing_ms"]         # {decode, segment, select, total}: server-side latency, report in the paper
```

`client.select` raises `ServerError` when the server returns `ok: false` (bad method, size mismatch, ...) and
`TimeoutError` after 60 s. Warm replies take about 0.3 s round trip, or about 6 s for `gpt4o`.

## 5. Mapping the chosen object to something the arm can pick

The skill is colour-driven: `execute_subtask(color)` makes `lang_color_detect` find the **largest blob** of that
colour's HSV range and grasp it. Only four colours are calibrated: red, green, blue and yellow
(`dofbot_pro_voice_ctrl/scripts/Color/<colour>_colorHSV.text`, 6 comma-separated ints `h,s,v,H,S,V`).

**Option A: colour-driven (do this first; no change to robot nodes).**
- Every block in a scene has a **unique colour**, so a scene holds 2–4 blocks.
- `color_match.py`:
  - `hsv_mask(frame, colour)` reproduces `lang_color_detect._find_blob` exactly: `inRange` on the calibrated range;
    for red, OR it with hue 0–8 at the same S and V bounds; then `MORPH_CLOSE` with a 5×5 kernel.
  - `match_colour(chosen_mask, frame)` computes `|hsv_mask(c) ∩ chosen| / |chosen|` for each colour and returns the
    best one if it is ≥ 0.3, else `None`. Log `None` as the failure cause `colour_mismatch`: the segmenter's object
    does not correspond to a block the detector can find.
  - `target_mask_from_colour(frame, colour, min_area=150)` returns the largest blob of the target colour as a filled
    contour mask, or `None` when nothing is found (the target is hidden).
- Before executing, check the colour's largest blob against the chosen centroid. If the largest blob of that colour is
  not the chosen object (e.g. a second object with a similar hue), log `colour_ambiguous`.

**Option B: pixel-driven (only if Option A works and time remains).** This lifts the 4-object cap, so scenes can have
5–8 objects of any colour.
- Add an action to `lang_color_grasp.py`, e.g. `{"action": "pick_place_at", "px": x, "py": y, "target": "left_bin"}`.
  It takes the grasp pixel directly, reads depth there the way `lang_color_detect` does (median patch), and runs the
  same IK and grasp sequence as `pick_place`.
- `RemoteRobotClient.execute_subtask` already sends a JSON payload on the command topic, so add the new action there.
- Pixel-driven grasps need a grasp point and yaw. Use `chosen_centroid` plus `minAreaRect` of `chosen_mask_np` if the
  wrist joint (joint 5) is used, else the centroid alone.

Report which option was used. A cube-only, 2–4-object real benchmark is fine for RA-L if it is described honestly.

## 6. Episode runner: `unveiler/unveiler_session.py`

Model it on `verify2act/v2a_session.py`: interactive prompts, crash-safe `episodes.jsonl`, `summary.md`/`summary.json`
at the end. Flags: `--jetson_ip 127.0.0.1 --scenes unveiler/scenes.yaml --methods sre sre_il heuristic gpt4o random
--executor {robot,manual,none} --max_steps N --session_name ...`.

One episode (one scene × one method):

```
1. Operator: "Scene S03 (target=green, occlusion=full). Place ONLY the target block, press Enter."
   ref_frame  = robot.capture_frame();  ref_mask = target_mask_from_colour(ref_frame, "green")
   (For partial-occlusion scenes, still capture ref_mask; it is the fallback if the target gets hidden later.)
2. Operator: "Build the rest of the scene (see layout photo results/<session>/layouts/S03.jpg), press Enter."
   On the first method run for a scene, save the frame as its layout photo, so every method gets the same layout.
3. client.reset(session, episode)
4. for step in range(max_steps):             # max_steps = number of blocks + 2
       frame = robot.capture_frame()           # the arm is at the observation pose between subtasks
       live  = target_mask_from_colour(frame, target_colour)
       visible = live is not None
       r = client.select(frame, live if visible else ref_mask, method, step=step,
                         target_visible=visible, return_overlay=True)
       save frame + overlay under ep_NNN/step_SS/
       if r["chosen_index"] < 0:  cause = "no_selection: " + r["reason"]; break
       colour = match_colour(r["chosen_mask_np"], frame)            # Option A
       if colour is None:         cause = "colour_mismatch"; break
       ok = execute(colour)       # robot: robot.execute_subtask(colour, "left_bin", action="pick_place")
                                  # manual: print "remove the <colour> block by hand", wait for Enter, ok = True
                                  # none:   ok = True, no motion (dry run)
       record step {chosen_index, colour, is_target, target_visible, probs, unfiltered_index, exec_ok: ok,
                    timing_ms, server log_dir: r["log_dir"]}
       if r["is_target"]:
           success = ok; break    # target grasp attempted: episode ends either way
       if not ok: record cause "grasp_failed" (continue: the block is usually still there, as in the sim eval)
   else: cause = "step_limit"
5. Ask the operator: "Real outcome? [y] target retrieved  [n] failed  [s] skip" + optional note
   (same prompts as v2a_session._ask_outcome). The operator's label is the reported success; `success` above is the
   automatic one, kept for comparison.
6. Append the episode to episodes.jsonl immediately.
```

`episodes.jsonl` record:

```json
{"episode": 7, "scene": "S03", "method": "sre", "density": "3-4", "occlusion": "full", "target_colour": "green",
 "n_objects_initial": 4, "steps": [...], "n_steps": 3, "success_auto": true, "real_success": true,
 "failure_cause": null, "note": "", "executor": "robot", "started": "...", "duration_s": 212.4}
```

`summary.md`: success rate and mean steps per method × (density, occlusion) cell, with the numerator and denominator
of each (e.g. `7/10`); failure-cause counts per method; mean server `timing_ms.total` per method; and for `sre`, how
often `unfiltered_index != chosen_index` when a reachable mask was used.

## 7. Camera and server settings (do once, before any runs)

1. **Which camera.** Use the frame `RemoteRobotClient.capture_frame()` returns at the observation pose
   (`lang_color_grasp.init_joints`). The detector and IK also use that frame, so a chosen mask maps directly to what the
   arm grasps. Save one frame and look at it. The SRE was trained on straight-down 400×400 images. If the observation
   view is strongly oblique, note it for the paper; do not add a second camera within this time budget.
2. **Crop.** Find the table-workspace box in that frame, square if possible, and give it to the server:
   `python -m robot.server --jetson-ip <JETSON_IP> --crop X0 Y0 X1 Y1`. Check `ping()["crop"]`.
3. **Segmentation check.** Build one 4-block scene, call `select(..., method="sre")`, and open `overlay.jpg` from the
   reply (or the server's `log_dir`). Every block needs its own outline. If blocks are missed, restart the server with
   `--seg-threshold 0.9`. If one mask spans two blocks, that scene layout is too tight for the segmenter.
4. **Reachability (optional).** Place a block at each corner of the workspace and try a pick. Paint the reachable
   region as a polygon in frame pixels, save it as `unveiler/reachable_mask.png`, and pass it on every `select`.
   Alternatively, only build scenes inside the reachable region and skip the mask; say which in the paper.

## 8. Evaluation protocol

- **Methods:** `sre` (main), `heuristic`, `gpt4o`, `random`. Add `sre_il` as the real "w/o RL" ablation if time allows.
- **Cells:** density {2 blocks, 3–4 blocks} × occlusion {partial, full} for Option A (4 cells). With Option B, use
  {3–4, 5–8}.
- **Scenes:** 5 fixed scenes per cell = 20 scenes, each photographed (layout photo). Every method runs every scene
  once, so each method gets 20 episodes, 80 in total for four methods. At about 3–4 min each that is roughly 5 h.
- **Counterbalancing:** rotate the method order per scene (S01: sre, heuristic, gpt4o, random; S02: heuristic, gpt4o,
  random, sre; ...), so battery, lighting and calibration drift do not favour one method.
- **Occlusion definitions** (put them in the paper):
  - *Partial:* part of the target is visible from the camera, but a neighbouring block prevents a direct grasp.
  - *Full:* the target is not visible in the camera frame. It is under or fully behind other blocks, and its mask
    comes from the reference capture.
- `scenes.yaml`:

  ```yaml
  - {id: S01, density: "2", occlusion: partial, target: green, blocks: [green, red]}
  - {id: S02, density: "3-4", occlusion: full, target: blue, blocks: [blue, red, yellow, green]}
  ```

- **Failure causes:** `no_selection`, `colour_mismatch`, `colour_ambiguous`, `grasp_failed`, `step_limit`,
  `operator_abort`. The split between "wrong selection" and "execution failure" is what the paper's error
  decomposition (ε_SRE vs ε_exec) needs.
- **Manual-executor pass (cheap, optional):** `--executor manual` with the operator removing each chosen block by hand
  measures selection quality with zero hardware failures. It is a clean real-world ε_SRE measurement if the arm proves
  unreliable.

## 9. Bring-up order and acceptance checks

1. Lab PC: `python -m robot.server --jetson-ip <JETSON_IP> --crop ...`. The log shows `Ready: listening on /unveiler/request`.
2. Lab PC: `python -m robot.mock_client --jetson-ip <JETSON_IP>` → `ALL CHECKS PASSED`. This proves the network path
   through the Jetson's real rosbridge, with no arm motion.
3. Jetson: `python3 -c "from unveiler.client import UnveilerClient; c = UnveilerClient('127.0.0.1'); print(c.ping())"`.
4. Jetson: `unveiler_session.py --executor none --methods sre heuristic` on one scene. Selections, colours, overlays and
   `episodes.jsonl` are written, and the arm does not move.
5. Jetson: `--executor manual` on one scene: prompts and the operator loop work.
6. Jetson: `--executor robot` on one 2-block partial scene with `heuristic`. The arm picks the chosen colour into the
   bin, and the episode ends with an outcome prompt.
7. Run the protocol in section 8.

Done means: 80 labeled episodes in `unveiler/results/<session>/episodes.jsonl`, a `summary.md` with the per-cell
table and failure causes, per-step frames and overlays on the Jetson, and matching server logs under
`object-unveiler/save/real_eval/` on the lab PC.

## 10. Known limitations to report

- Real objects are coloured blocks; sim used household-object meshes. Real scenes are small (Option A: at most 4
  blocks).
- The executor is a scripted forward pick-and-place, not the Action Decoder, and has no push. Approach-direction
  blocking (a block in front of the target) is not something the SRE models, because it was trained with a floating
  top-down hand.
- The SRE sees a single fixed observation view. Full-occlusion targets are specified by a reference mask captured
  before the occluders were placed.
