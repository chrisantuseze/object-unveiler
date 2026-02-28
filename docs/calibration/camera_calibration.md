```markdown
# Camera ↔ Robot Calibration (T_cam_base)

This document records the exact procedure used to measure the camera-to-robot-base
extrinsics (`T_cam_base`) for the Dofbot Pro + Orbbec DaBai camera. Keep this
file with the repo so you can repeat the procedure later.

## Quick summary
- Script used: `dofbot_pro_ws/src/dofbot_pro_info/scripts/calibration/record_cam_extrinsics.py`
- AprilTag publisher: `/unveiler/tag_detections` (type `apriltag_ros/AprilTagDetectionArray`)
- FK service: `get_kinemarics` (type `dofbot_pro_info/kinemarics`) — use `kin_name='fk'`
- Output YAML: `yaml/cam_extrinsics.yaml`
- Target reprojection error: < 0.01 m (achieved ~0.0036 m)

## Prerequisites
1. Camera driver running and publishing topics:
   - `/camera/color/image_raw` (Image)
   - `/camera/depth/image_raw` (Image, 16UC1 mm)
   - `/camera/depth/camera_info` (CameraInfo)
2. `apriltag_ros` node running in `/unveiler/` namespace and publishing
   `/unveiler/tag_detections`.
3. `get_kinemarics` service available on the Jetson (or lab machine) and
   `/joint_states` publishing current joint values.
4. Printed AprilTag attached flat to the table (calibration tag ID known).

## Full procedure (what we ran)

1. Launch camera and apriltag nodes

   - Camera:

     ```bash
     roslaunch orbbec_camera dabai_dcw2.launch
     ```

   - Apriltag (example with standalone tag size override):

     ```bash
     roslaunch apriltag_ros continuous_detection.launch \
         camera_name:=/camera/color image_topic:=image_raw \
         standalone_tags:="[ {id: 1, size: 0.029} ]" \
         namespace:=/unveiler
     ```

   Verify detections:

   ```bash
   rostopic echo /unveiler/tag_detections
   ```

2. Collect tag pose pairs (camera-frame + robot-base-frame)

   - We used the interactive script `record_cam_extrinsics.py`:
     - CLI: `--n` (number of samples), `--auto-fk` to call the FK service,
       `--fk-service` to set the service name (default `get_kinemarics`),
       `--tag-topic` (default `/unveiler/tag_detections`).
     - The script subscribes to the apriltag topic and waits for a non-empty
       detection then either:
         - uses `get_kinemarics(kin_name='fk')` + current joint states to obtain
           the tag pose in the robot base frame; or
         - prompts for manual FK XYZ entry if auto-FK fails.

   - Example run:

     ```bash
     rosrun dofbot_pro_info record_cam_extrinsics.py --n 6 --auto-fk --tag-topic /unveiler/tag_detections
     ```

   - The script computes one `T_cam_base` per pair as
     `T_tag_in_base @ inv(T_tag_in_cam)`, averages them (translation mean,
     quaternion rotation averaging), validates reprojection error, and writes
     `yaml/cam_extrinsics.yaml`.

3. Validate

   - The script reports per-sample reprojection L2 errors and prints the mean.
   - Our run produced reprojection errors:
     `0.0049, 0.0025, 0.0014, 0.0040, 0.0015, 0.0074 m` — mean `~0.0036 m`.
   - If mean > 0.01 m: collect more samples or check tag mounting, FK accuracy,
     and apriltag pose stability.

4. Save

   - The final transform was saved to `yaml/cam_extrinsics.yaml` as `T_cam_base` (4×4 list).

## Files & key locations
- Script: `dofbot_pro_ws/src/dofbot_pro_info/scripts/calibration/record_cam_extrinsics.py`
- Saved extrinsics: `yaml/cam_extrinsics.yaml`
- Unveiler Jetson code loads the transform via `robot_operations.load_T_cam_base()`.

## Troubleshooting notes & gotchas
- Do NOT use `rospy.AnyMsg` to receive `AprilTagDetectionArray`; use the typed
  message `apriltag_ros.msg.AprilTagDetectionArray` so `.detections` is available.
- If `/unveiler/tag_detections` is empty (`detections: []`) repeatedly,
  check the `standalone_tags` size param — tag size mismatch prevents detection.
- If FK service returns unexpected values, verify `/joint_states` is current and
  the `get_kinemarics` server is using the same joint ordering/units (degrees vs radians).
- When using `--auto-fk`, ensure the gripper is centered over the printed tag and
  robot is static when the FK reading is taken.

## Example final `T_cam_base` (saved in YAML)

```yaml
T_cam_base:
- [-0.9998, 0.0196, -0.0078, -0.0800]
- [ 0.0180, 0.9858,  0.1670, -0.0085]
- [ 0.0110, 0.1668, -0.9859,  0.0881]
- [ 0.0,    0.0,     0.0,     1.0   ]
```

## Next recommended actions (Step 3 / Step 4)
1. Measure real workspace bounds (Step 3) by moving the gripper to 4 table corners
   and recording FK `(x,y,z)` — see the steps below.
2. Run the heightmap smoke-test (Step 4) using `robot_operations.get_real_heightmap()`
   to ensure the top-down heightmap matches the physical scene.

---

Document moved into `docs/calibration` on Feb 28, 2026.

```
