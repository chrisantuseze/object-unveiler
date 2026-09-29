# Unveiler real-robot server (lab PC)

The lab PC runs the perception and object-selection models. The DOFBOT's Jetson runs the episode loop and the
Verify2Act pick-and-place skill. They talk over the Jetson's rosbridge, the same pattern as Verify2Act:

```
Jetson (dofbot-controller)                         Lab PC (object-unveiler, this folder)
  camera, arm_driver, lang_color_detect/grasp
  rosbridge_websocket :9090  <── ws ─────────────  python -m robot.server --jetson-ip <JETSON_IP>
  unveiler episode runner                            Mask R-CNN + SRE / SRE-IL / heuristic / GPT-4o / CLIP / random
     │  /unveiler/request   {op: select, image, target_mask, method}  ──>
     │  /unveiler/response  {chosen_index, chosen_mask, is_target, ...} <──
     └─> execute_subtask(color of the chosen object)
```

The server is stateless per request. It segments the frame and returns the one object the chosen method would
remove next, and it never moves the arm. The wire format is specified in [`protocol.py`](protocol.py), and the
client side is described in [`CLIENT_INTEGRATION.md`](CLIENT_INTEGRATION.md).

| file | role |
|---|---|
| `protocol.py` | topics, ops, image/mask encoding (no torch; the Jetson copies it) |
| `client.py` | `UnveilerClient`: ping / reset / select over rosbridge (no torch; the Jetson copies it) |
| `backend.py` | segmentation, SRE input construction (same tensors as the sim eval), the six selectors, step logs |
| `server.py` | rosbridge server + `--offline-image` mode |
| `pick_corners.py` | click the workspace corners on a frame, print `--warp` arguments, preview the rectified view |
| `fake_rosbridge.py` | minimal rosbridge stand-in for testing without a Jetson |
| `mock_client.py` | sends the requests the robot will send, and checks the replies |
| `testdata/` | a sim top-down scene placed in a 640x480 frame at (120, 40)-(520, 440), plus its target mask |

## Run

```bash
conda activate unveiler                    # needs roslibpy (pip install roslibpy; in requirements.txt)
cd object-unveiler
python -m robot.server --jetson-ip 192.168.0.8 --crop X0 Y0 X1 Y1
```

- `--crop X0 Y0 X1 Y1`: the table workspace in camera pixels. The frame is cropped to this box and resized to
  400x400, the sim camera resolution the segmenter and the SRE were trained on. Choose a square-ish box that tightly
  frames the workspace. The heuristic ranks objects by distance to the image border, so the crop border should be the
  workspace edge, as it was in sim. The default is the whole frame.
- `--warp TLx TLy TRx TRy BRx BRy BLx BLy` (use instead of `--crop` when the camera is oblique): the four workspace
  corners in camera pixels. The quad is rectified with a homography to the straight-down 400x400 view the sim camera
  had, and masks are warped back, so replies stay in camera pixels. Get the corners by clicking them on a saved frame:
  `python -m robot.pick_corners <frame.jpg>` prints the `--warp ...` arguments and writes a preview of the rectified
  view. Pick a square-ish region of the table (the sim workspace was square), so blocks keep their aspect ratio.
  Every step also saves `sim_view.jpg`, the rectified image the segmenter and the SRE actually saw.
- `--seg-threshold` (default 0.97, `ObjectSegmenter`'s real-image value). Lower it if blocks are missed.
- `--output-dir` (default `save/real_eval`): every `select` writes
  `<session>/episode_NNNN/step_SS_<method>/{frame.jpg, target_mask.png, overlay.jpg, sim_view.jpg, masks_sim.npz, reply.json}`.
  `overlay.jpg` outlines every object with its index: target red, chosen green, unreachable grey.
- Checkpoints: `--sre-rl-ckpt save/sre_rl/sre_rl_best.pt` (the one `main.py` evaluates) and
  `--sre-il-ckpt save/sre/sre_model_best.pt`.
- `gpt4o` needs `OPENAI_API_KEY`. `clip` loads on first use (about 5 s once).
- The first start builds Mask R-CNN from torchvision, which downloads the COCO weights once (170 MB) before loading
  `downloads/maskrcnn.pth`.

Warm latency on a T4, per `select`: segmentation 105–165 ms, SRE about 50 ms, 145–225 ms server total, about
300 ms round trip including JPEG transfer. GPT-4o takes about 6 s.

## Test without the robot

```bash
PY=python   # the unveiler env
$PY -m robot.fake_rosbridge --port 9090 &
$PY -m robot.server --jetson-ip 127.0.0.1 --crop 120 40 520 440 &
$PY -m robot.mock_client --jetson-ip 127.0.0.1 --save-dir /tmp/overlays   # add: --methods sre gpt4o ...
```

`mock_client` checks ping, reset, a `select` for every method, the reachability filter, the hidden-target path and
the error path, and ends with `ALL CHECKS PASSED`. To test the real network path once the Jetson's rosbridge is up,
run the server and `python -m robot.mock_client --jetson-ip <JETSON_IP>` from the lab PC. The arm does not move.

Single image, no rosbridge:

```bash
python -m robot.server --offline-image robot/testdata/frame.png --offline-target robot/testdata/target_mask.png \
    --crop 120 40 520 440 --method sre
```

## Method notes (for the paper)

- All methods see the same segmentation, and each returns one object index. Execution is identical across methods
  (the Jetson's pick-and-place skill), so differences come from selection alone.
- `sre` is the PPO-fine-tuned SRE (the full Unveiler reasoning module). `sre_il` is the imitation-only SRE, i.e. the
  "SRE w/o RL" ablation.
- `heuristic` is `grasping.find_obstacles_to_remove`, the supervisor the SRE was trained on. For three or fewer
  objects it returns the target itself, and the server then ranks the remaining objects by distance to the target.
- `gpt4o` and `clip` are the sim and Table IV baselines, unchanged (`baseline/gpt.py`, `baseline/clip_eval.py`).
  Both receive binary masks, as in sim.
- **Target identity.** The client sends the target mask on every step. The server matches it to a segmented object by
  IoU ≥ 0.3. If the client says `target_visible: false` (full occlusion), no object is treated as the target, and the
  mask the client sent (captured before the occluders were placed) is the SRE's target input.
- **Reachability** (optional `reachable_mask`). Objects whose centroid falls outside the mask are excluded from every
  method's choice: SRE logits are set to -1e4, the heuristic skips them, GPT-4o is told their indices, and CLIP and
  random choose among the reachable objects. `unfiltered_index` records the SRE's choice before filtering, so you can
  report how often the arm's reach changed the decision.
- `sre_rl_best.pt` (PPO episode 6060, entropy bonus 0.05) produces a much flatter distribution over objects than the
  IL checkpoint. On the test scene its top probability is 0.20, versus 0.85 for `sre_il`. It is the checkpoint behind
  the sim numbers, so it is the right one to report. Just don't read much into its probabilities as confidence.
