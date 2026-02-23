# Real Robot Setup Plan — Dofbot Pro + Unveiler Policy

> Status: work-in-progress  
> Last updated: February 2026

---

## Network Architecture (updated Feb 2026)

All lab computers now run **ROS 2**.  The Dofbot Pro Jetson still runs **ROS 1
Noetic** for its driver packages.  Rather than migrating any driver code to ROS 2
or setting up a bidirectional ros1_bridge, the chosen approach requires zero code
changes: **run a ROS 1 Noetic Apptainer container on a lab machine**.

```
┌─────────────────────────────────────┐      Ethernet / WiFi
│  Lab machine  (ROS 2 host OS)       │ ◄────────────────────►  Dofbot Pro Jetson
│  ┌─────────────────────────────┐    │                         (ROS 1 Noetic native)
│  │  Apptainer container        │    │                         - arm_driver
│  │  (ROS 1 Noetic + CUDA)      │    │                         - orbbec_camera
│  │                             │    │                         - ik_service
│  │  • roscore  (ROS master)    │    │
│  │  • SRE + ActionDecoder      │    │
│  │  • unveiler_grasp node      │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

The Jetson connects to the container's roscore by setting at boot:
```bash
export ROS_MASTER_URI=http://<lab_machine_ip>:11311
export ROS_IP=<jetson_ip>
```

---

## Step 0 — Set Up the ROS 1 Inference Container (lab machine)

Do this **once** per lab machine that will run inference.

### 0.1  Build the container image

```bash
# From the object-unveiler repo root on the lab machine
apptainer build apptainerfile_ros1.sif apptainerfile_ros1.def
```

The definition file (`apptainerfile_ros1.def`) installs:
- Ubuntu 20.04 + CUDA 11.8 (required for ROS 1 Noetic + lab GPU)
- ROS 1 Noetic (ros-base + cv_bridge + sensor/geometry/std msgs)
- All Python inference deps (torch 2.1.2+cu118, open3d, pybullet, etc.)

> **Tip:** The build takes ~15 min.  Run it once and re-use the `.sif` file.
> The `.sif` is not tracked by git (add it to `.gitignore`).

### 0.2  Launch the inference stack

```bash
# Bind-mount the repo so the container can see your code and yaml/ configs.
LAB_IP=<lab_machine_ip> \
apptainer exec --nv \
    --bind /path/to/object-unveiler:/workspace \
    apptainerfile_ros1.sif \
    bash /workspace/run_inference_container.sh
```

This script (see `run_inference_container.sh`):
1. Sources ROS 1 Noetic.
2. Sets `ROS_MASTER_URI` and `ROS_IP` to the lab machine's IP.
3. Starts `roscore` in the background.
4. Launches the Unveiler inference node (`eval_agent.py`).

Edit the last `python3 eval_agent.py ...` line in the script to match your
actual launch command if it differs.

### 0.3  Configure the Jetson

On the Jetson, add the following to `~/.bashrc` (replace the IP):

```bash
export ROS_MASTER_URI=http://<lab_machine_ip>:11311
export ROS_IP=<jetson_ip>
```

Then restart the shell or run `source ~/.bashrc`.  Verify connectivity:

```bash
# On the Jetson — should list topics from the container
rostopic list
```

---

## The Dependency Chain

Everything flows from one root cause: **we don't have a calibrated camera-to-robot
transform (T_cam_base)**.  Fixing that single missing piece unblocks the entire
downstream stack:

```
T_cam_base (extrinsics)
    │
    ▼
depth image → point cloud in robot base frame
    │
    ▼
point cloud projected to top-down heightmap  (= policy "state")
    │
    ▼
SRE + ActionDecoder predict grasp  (pixel x,y  + theta + aperture)
    │
    ▼
pixel coords converted to 3-D XYZ in robot base frame  (real_action3d)
    │
    ▼
IK service → joint angles → arm_driver → robot moves
```

Each section below is self-contained and must be completed in order.

---

## Step 1 — Calibrate Camera Extrinsics (T_cam_base)

### What it is
A 4×4 rigid-body transform that maps a 3-D point expressed in the **camera frame**
into the **robot base frame**:

```
p_robot = T_cam_base @ p_camera
```

### How to measure it (Eye-to-Hand calibration with AprilTag)

The workspace already contains `apriltag_ros-master`.  Use it as follows.

**1.1  Print and attach a calibration tag**

- Print AprilTag ID 0 from the `tag36h11` family (≈ 15 cm side).
- Attach it flat on the table at a known, stable position where the robot FK can also
  reach it — e.g., directly in front of the arm at ~30 cm.

**1.2  Get the tag pose in the camera frame from ROS**

```bash
# terminal 1 — launch camera
roslaunch orbbec_camera dabai_dcw2.launch

# terminal 2 — launch apriltag detector
roslaunch apriltag_ros continuous_detection.launch \
    camera_name:=/camera/color \
    image_topic:=image_raw

# terminal 3 — check detections
rostopic echo /tag_detections
```

The topic publishes `geometry_msgs/PoseStamped` for each detected tag.  Record the
`(x, y, z, qx, qy, qz, qw)` — this is **T_tag_in_cam**.

**1.3  Get the tag pose in the robot base frame**

Use the robot FK service at the known joint configuration where the gripper tip is
directly above/touching the tag centre.  Record the FK (x, y, z) — this is
**T_tag_in_base** (set rotation = identity, since we only need translation for a planar
worksurface).

**1.4  Solve for T_cam_base**

```python
import numpy as np
from scipy.spatial.transform import Rotation

# -- fill these in from the measurements above --
# tag pose in camera frame  (from /tag_detections)
t_tag_cam = np.array([x_c, y_c, z_c])
q_tag_cam = np.array([qx, qy, qz, qw])   # xyzw

# tag pose in robot base frame  (from FK service)
t_tag_base = np.array([x_r, y_r, z_r])

# Build 4x4 T_tag_in_cam
R_tag_cam = Rotation.from_quat(q_tag_cam).as_matrix()  # 3x3
T_tag_in_cam = np.eye(4)
T_tag_in_cam[:3, :3] = R_tag_cam
T_tag_in_cam[:3,  3] = t_tag_cam

# Assume tag is flat on table: rotation in base frame = identity
T_tag_in_base = np.eye(4)
T_tag_in_base[:3, 3] = t_tag_base

# T_cam_base = T_tag_in_base @ inv(T_tag_in_cam)
T_cam_base = T_tag_in_base @ np.linalg.inv(T_tag_in_cam)
print("T_cam_base:\n", T_cam_base)
```

**Collect 6–8 tag positions** (move the tag around the table, repeat the measurement)
and average the resulting `T_cam_base` matrices to reduce noise.

**1.5  Validate**

Pick a new tag position, compute `T_cam_base @ p_tag_cam` and compare it to the FK
value.  Target reprojection error: **< 1 cm**.

**1.6  Save the result**

```bash
# Save to yaml/ so the node can load it at startup
cat > object-unveiler/yaml/cam_extrinsics.yaml << 'EOF'
T_cam_base:
  - [r00, r01, r02, tx]
  - [r10, r11, r12, ty]
  - [r20, r21, r22, tz]
  - [0.0, 0.0, 0.0, 1.0]
EOF
```

---

## Step 2 — Fix the Depth→Point Cloud→Heightmap Pipeline

### Root cause
`general_utils.get_fused_heightmap()` was designed for the **simulation** pipeline:

- It expects `obs['color']`, `obs['depth']`, `obs['seg']` as **lists** (one per
  simulated camera).
- It computes the camera-to-world transform from PyBullet-style
  `pos / target_pos / up_vector` vectors via `pybullet_utils.get_camera_pose()`.

Neither of these assumptions holds for the real robot.

### Fix: write `get_real_heightmap()` in `robot_operations.py`

This function takes a **single** RGB-D frame + the measured `T_cam_base` and produces
the same `(H, W)` float32 heightmap that the policy expects.

```python
# robot_operations.py  — add this function

import numpy as np
import cv2
import yaml, os

_EXTRINSICS_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', 'yaml', 'cam_extrinsics.yaml'
)

def load_T_cam_base():
    """Load the measured camera-to-robot-base transform."""
    with open(_EXTRINSICS_PATH, 'r') as f:
        data = yaml.safe_load(f)
    return np.array(data['T_cam_base'], dtype=np.float64)


def get_real_heightmap(color_bgr, depth_m, intrinsics_3x3, T_cam_base,
                       bounds, pix_size):
    """
    Build a top-down depth heightmap from a single RGB-D frame.

    Parameters
    ----------
    color_bgr    : (H, W, 3) uint8   — BGR image from the Orbbec camera
    depth_m      : (H, W)    float32 — depth in **metres** (divide raw 16UC1 by 1000)
    intrinsics_3x3 : (3, 3)  float64 — camera K matrix from /camera/depth/camera_info
    T_cam_base   : (4, 4)    float64 — camera→robot-base rigid transform
    bounds       : (3, 2)    float64 — [[xmin,xmax],[ymin,ymax],[zmin,zmax]] in metres
    pix_size     : float              — metres per pixel in the heightmap

    Returns
    -------
    height_grid : (H_map, W_map) float32 — top-down heightmap in robot base frame
    """
    H, W = depth_m.shape
    fx = intrinsics_3x3[0, 0]
    fy = intrinsics_3x3[1, 1]
    cx = intrinsics_3x3[0, 2]
    cy = intrinsics_3x3[1, 2]

    # --- 1. Back-project depth pixels to 3-D points in camera frame ---
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = (depth_m > 0.05) & (depth_m < 1.5)   # clip to plausible range (metres)

    z_c = depth_m[valid]
    x_c = (u[valid] - cx) * z_c / fx
    y_c = (v[valid] - cy) * z_c / fy

    pts_cam = np.vstack([x_c, y_c, z_c, np.ones_like(z_c)])   # 4 × N

    # --- 2. Transform to robot base frame ---
    pts_base = (T_cam_base @ pts_cam)[:3, :]    # 3 × N

    # --- 3. Project to top-down heightmap ---
    heightmap_size = (
        int(np.round((bounds[1][1] - bounds[1][0]) / pix_size)),
        int(np.round((bounds[0][1] - bounds[0][0]) / pix_size)),
    )
    height_grid = np.zeros(heightmap_size, dtype=np.float32)

    for i in range(pts_base.shape[1]):
        px = pts_base[0, i]
        py = pts_base[1, i]
        pz = pts_base[2, i]

        idx_x = int(np.floor((px - bounds[0][0]) / pix_size))
        idx_y = int(np.floor((py - bounds[1][0]) / pix_size))

        if (0 <= idx_x < heightmap_size[1] - 1 and
                0 <= idx_y < heightmap_size[0] - 1):
            if height_grid[idx_y, idx_x] < pz:
                height_grid[idx_y, idx_x] = pz

    # Flip to match the orientation convention used during training
    return cv2.flip(height_grid, 1)
```

### Wire it into `unveiler_grasp.py`

Replace the call to `policy.state_representation(obs)` in `run()` with:

```python
# In unveiler_grasp.py  run() loop, replace:
#   state = self.policy.state_representation(obs)
# with:

from robot_operations import get_real_heightmap, load_T_cam_base

# -- done once at startup (or in __init__) --
T_cam_base = load_T_cam_base()

# -- in the grasp loop --
depth_m = obs['depth'].astype(np.float32) / 1000.0   # 16UC1 mm → metres
state = get_real_heightmap(
    obs['color'], depth_m, self.intrinsics,
    T_cam_base,
    bounds=np.array(self.policy.bounds),
    pix_size=self.policy.pxl_size,
)
```

> **Note:** the `seg` image that `get_fused_heightmap` uses in simulation is only used
> for colouring the point cloud for debugging.  The policy only reads the **height
> values**, so we can safely omit it.

---

## Step 3 — Measure Real Workspace Bounds

`action3d()` converts pixel coordinates back to 3-D using `self.bounds`, which are the
**simulation** workspace bounds.  For the real robot we need the equivalent physical
bounds measured in the **robot base frame**.

### Procedure

1. Move the robot gripper to the four corners of the reachable table area.
2. At each corner, record the FK (x, y, z) from the IK service (`kin_name = "fk"`).
3. Compute:

```python
# After collecting 4+ FK corner measurements:
x_vals = [fk[0] for fk in corners]   # metres
y_vals = [fk[1] for fk in corners]
real_bounds = np.array([
    [min(x_vals), max(x_vals)],   # x (forward/back in base frame)
    [min(y_vals), max(y_vals)],   # y (left/right in base frame)
    [0.004, 0.30],                # z: table surface to max object height
])
```

4. Save in `yaml/bhand.yml` under a new key `real_workspace.bounds`, and load it
   in `unveiler_grasp.py`.

---

## Step 4 — Fix Pixel → 3-D → IK (`real_action3d`)

### Root cause
`Policy.action3d()` computes XYZ using sim bounds and takes no IK step.  For the real
robot we need to:
1. Convert pixel `(px, py)` in the heightmap → physical XYZ in robot base frame using
   the **real** workspace bounds.
2. Pass that XYZ to the IK service.

### Add `real_action3d()` to `unveiler_grasp.py`

```python
def real_action3d(self, action, real_bounds, pix_size):
    """
    Convert a policy action (pixel coords + theta + aperture) to a robot
    joint configuration via the IK service.

    Parameters
    ----------
    action      : np.array([px, py, theta, aperture])
    real_bounds : (3, 2) array — real workspace bounds in robot base frame (metres)
    pix_size    : float — metres per pixel (same value used to build the heightmap)

    Returns
    -------
    joint_angles : list of 6 floats, or None if IK fails
    aperture     : float (gripper opening, already in servo degrees)
    """
    px, py, theta, aperture = action

    # pixel → physical XYZ  (mirror of Policy.action3d() but with real bounds)
    x = real_bounds[0][0] + pix_size * px
    y = real_bounds[1][0] + pix_size * py
    z = real_bounds[2][0] + 0.03           # small standoff above surface

    print(f"real_action3d → robot XYZ: ({x:.3f}, {y:.3f}, {z:.3f})  theta={np.degrees(theta):.1f}°")

    # IK call
    request = kinemaricsRequest()
    request.kin_name = "ik"
    request.tar_x = x
    request.tar_y = y
    request.tar_z = z
    # Use wrist angle derived from grasp orientation
    request.Roll = theta    # adjust sign/offset if the convention differs

    try:
        self.ik_client.wait_for_service(timeout=3.0)
        response = self.ik_client.call(request)

        joints = [
            response.joint1,
            response.joint2,
            response.joint3,
            min(response.joint4, 90.0),   # clamp joint4
            90.0,                          # joint5 fixed
            30.0,                          # joint6 (gripper — set by gripper_control)
        ]

        # Sanity check
        if any(j < 0 or j > 180 for j in joints[:4]):
            rospy.logwarn(f"IK result out of range: {joints}")
            return None, aperture

        return joints, aperture

    except rospy.ServiceException as e:
        rospy.logerr(f"IK service failed: {e}")
        return None, aperture
```

### Update `grasp_object()` to use `real_action3d`

```python
def grasp_object(self, action_4dof):
    """
    action_4dof : raw np.array([px, py, theta, aperture]) from exploit_unveiler()
                  (NOT the dict from Policy.action3d())
    """
    real_bounds = np.array(self.args.real_bounds)   # loaded from yaml
    joint_angles, aperture = self.real_action3d(
        action_4dof, real_bounds, self.policy.pxl_size
    )

    if joint_angles is None:
        rospy.logerr("IK failed — skipping grasp step")
        return self.get_observation()

    # Set gripper width before moving to grasp position
    self.gripper_control(aperture)
    rospy.sleep(0.5)

    self.step(joint_angles)
    print("Grasp executed.")
    rospy.sleep(2)

    general_utils.delete_episodes_misc(self.TEST_EPISODES_DIR)
    return self.get_observation()
```

And update the call site in `run()`:

```python
# Replace:
#   action_dict = self.policy.action3d(action)
#   next_obs = self.grasp_object(action_dict)
# With:
next_obs = self.grasp_object(action)   # pass the raw 4-DOF array
```

---

## Step 5 — Incremental Validation Sequence

Work through these in order.  Stop and debug at any step that fails before moving on.

| # | Test command / check | Passes when... |
|---|---|---|
| 1 | `rostopic echo /camera/depth/camera_info` | K matrix prints non-zero values |
| 2 | `rostopic echo /camera/depth/image_raw` | 16UC1 depth image appears |
| 3 | Run the calibration script (Step 1.4) | Reprojection error < 1 cm on held-out tag points |
| 4 | `get_real_heightmap()` smoke-test in a Python script on the robot | Heightmap image shows non-zero blobs where objects are placed |
| 5 | `real_action3d()` with a manually specified pixel | IK response contains valid (0–180°) joint angles |
| 6 | Single grasp on an isolated object | Robot moves to correct position and closes gripper |
| 7 | Full `run()` with 2-object clutter | Policy selects and removes the obstacle before grasping target |

### Quick heightmap smoke-test script

```python
# run this on the Jetson after steps 1–3 are done
import rospy, cv2, numpy as np, yaml
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

rospy.init_node('hmap_test')
bridge = CvBridge()

color_msg = rospy.wait_for_message('/camera/color/image_raw', Image)
depth_msg = rospy.wait_for_message('/camera/depth/image_raw', Image)
info_msg  = rospy.wait_for_message('/camera/depth/camera_info', CameraInfo)

color = bridge.imgmsg_to_cv2(color_msg, 'bgr8')
depth = bridge.imgmsg_to_cv2(depth_msg, '16UC1').astype(np.float32) / 1000.0
K     = np.array(info_msg.K).reshape(3, 3)

# adjust these to your measured real bounds
bounds   = np.array([[-0.05, 0.35], [-0.20, 0.20], [0.004, 0.30]])
pix_size = 0.005

import sys; sys.path.insert(0, '/path/to/object-unveiler')
from dofbot_pro_ws.src.dofbot_pro_info.scripts.robot_operations import (
    get_real_heightmap, load_T_cam_base
)
T = load_T_cam_base()
hmap = get_real_heightmap(color, depth, K, T, bounds, pix_size)

cv2.imwrite('/tmp/hmap.png',
    cv2.normalize(hmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
print('saved /tmp/hmap.png  max_height=', hmap.max())
```

---

## Summary of Files to Change

| File | Change |
|---|---|
| `yaml/cam_extrinsics.yaml` | **Create** — store measured T_cam_base |
| `yaml/bhand.yml` | **Add** `real_workspace.bounds` key |
| `robot_operations.py` | **Add** `load_T_cam_base()` and `get_real_heightmap()` |
| `unveiler_grasp.py` | **Add** `real_action3d()`; update `grasp_object()` and `run()` |
