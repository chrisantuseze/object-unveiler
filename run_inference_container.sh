#!/usr/bin/env bash
# run_inference_container.sh
#
# Launch the Unveiler inference stack inside the ROS 1 Noetic Apptainer container.
# Run this ON THE LAB MACHINE (NOT on the Jetson).
#
# The lab machine becomes the ROS master.  The Jetson's robot-driver packages
# connect to it using:
#   export ROS_MASTER_URI=http://<LAB_MACHINE_IP>:11311
#   export ROS_IP=<JETSON_IP>
#
# Typical invocation from the host OS (lab machine):
#   apptainer exec --nv \
#       --bind /path/to/object-unveiler:/workspace \
#       apptainerfile_ros1.sif \
#       bash /workspace/run_inference_container.sh
#
# ─────────────────────────────────────────────────────────────────────────────
set -e

# ── 1.  Configuration ─────────────────────────────────────────────────────────
WORKSPACE_DIR="${UNVEILER_WS:-/workspace}"        # bind-mounted object-unveiler root
YAML_CONFIG="${WORKSPACE_DIR}/yaml/bhand.yml"

# The lab machine's IP that the Jetson can reach over the network.
# Override with:  LAB_IP=x.x.x.x bash run_inference_container.sh
LAB_IP="${LAB_IP:-$(hostname -I | awk '{print $1}')}"

# ── 2.  Source ROS ────────────────────────────────────────────────────────────
source /opt/ros/noetic/setup.bash

# ── 3.  Export ROS networking ─────────────────────────────────────────────────
export ROS_MASTER_URI="http://${LAB_IP}:11311"
export ROS_IP="${LAB_IP}"

echo "============================================================"
echo "  ROS master : ${ROS_MASTER_URI}"
echo "  ROS IP     : ${ROS_IP}"
echo "  Workspace  : ${WORKSPACE_DIR}"
echo "============================================================"

# ── 4.  Start roscore (if it isn't already running) ───────────────────────────
if ! rostopic list > /dev/null 2>&1; then
    echo "[run_inference_container] Starting roscore..."
    roscore &
    ROSCORE_PID=$!
    # Wait until roscore is ready
    for i in $(seq 1 20); do
        sleep 1
        if rostopic list > /dev/null 2>&1; then
            echo "[run_inference_container] roscore is up."
            break
        fi
        echo "[run_inference_container] Waiting for roscore... (${i}/20)"
    done
else
    echo "[run_inference_container] roscore already running — skipping."
    ROSCORE_PID=""
fi

# ── 5.  Launch the inference node ─────────────────────────────────────────────
# Adjust the path / script name to match your actual entry point.
# The node typically runs eval_agent.py or unveiler_grasp.py with a ROS spin loop.

cd "${WORKSPACE_DIR}"

# Example: eval_agent.py in ROS-service mode.
# Replace the flags below with whatever your normal launch command requires.
python3 eval_agent.py \
    --config "${YAML_CONFIG}" \
    "$@"

# ── 6.  Cleanup ───────────────────────────────────────────────────────────────
if [ -n "${ROSCORE_PID}" ]; then
    echo "[run_inference_container] Shutting down roscore (pid ${ROSCORE_PID})..."
    kill "${ROSCORE_PID}" 2>/dev/null || true
fi
