#!/usr/bin/env python3
"""
test_transport.py — quick end-to-end transport smoke-test.

Runs entirely on the LAB computer (no ROS install needed).

Tests:
  1. Lab → Jetson publish (ActionData)  — verifiable with `rostopic echo /action/data`
  2. Jetson → Lab subscribe (ObservData) — count observations received for 10 s

Usage:
    python robot_policies_ws/src/policies/scripts/test_transport.py \
        --jetson_ip 192.168.0.8 --jetson_port 9090

On the Jetson, to trigger test (2) run:
    rostopic pub /action/obs dofbot_pro_info/ObservData \
        "color_image: {}" "depth_image: {}" "cam_intrinsics: [1,0,0,0,1,0,0,0,1]"
"""

import argparse
import time
import sys

import roslibpy

ACTION_TOPIC    = "/action/data"
ACTION_MSG_TYPE = "dofbot_pro_info/ActionData"
OBS_TOPIC       = "/action/obs"
OBS_MSG_TYPE    = "dofbot_pro_info/ObservData"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jetson_ip",   default="192.168.0.8", type=str)
    parser.add_argument("--jetson_port", default=9090,          type=int)
    parser.add_argument("--duration",    default=10,            type=int,
                        help="Seconds to listen for observations")
    args = parser.parse_args()

    print(f"Connecting to rosbridge at ws://{args.jetson_ip}:{args.jetson_port} ...")
    client = roslibpy.Ros(host=args.jetson_ip, port=args.jetson_port)
    client.run()

    # Wait for real connection
    deadline = time.time() + 10.0
    while not client.is_connected and time.time() < deadline:
        time.sleep(0.2)

    if not client.is_connected:
        print("FAILED: could not connect to rosbridge. Check that:")
        print("  1. rosbridge_server is running on the Jetson (port 9090)")
        print("  2. you can ping the Jetson from this machine")
        sys.exit(1)

    print(f"OK: connected to rosbridge (is_connected={client.is_connected})")

    # ── Test 1: Lab → Jetson publish ─────────────────────────────────────────
    print("\n── Test 1: Publishing 5 dummy ActionData messages to /action/data ──")
    print("   On the Jetson run:  rostopic echo /action/data   to verify receipt.")

    action_pub = roslibpy.Topic(client, ACTION_TOPIC, ACTION_MSG_TYPE)
    action_pub.advertise()
    time.sleep(0.5)   # let rosbridge register the advertise

    for i in range(5):
        msg = roslibpy.Message({'values': [float(i), float(i), 0.0, 0.5]})
        action_pub.publish(msg)
        print(f"  Published action #{i+1}: values=[{i}, {i}, 0.0, 0.5]")
        time.sleep(0.3)

    print("   Done — check Jetson rostopic echo output.")

    # ── Test 2: Jetson → Lab subscribe ───────────────────────────────────────
    obs_count = [0]

    def on_obs(msg):
        obs_count[0] += 1
        intrinsics = msg.get('cam_intrinsics', [])
        print(f"  [Obs #{obs_count[0]}] Received ObservData "
              f"(color_image encoding={msg.get('color_image', {}).get('encoding', '?')}, "
              f"cam_intrinsics len={len(intrinsics)})")

    print(f"\n── Test 2: Subscribing to /action/obs for {args.duration} s ──")
    print("   Trigger from Jetson with:")
    print("     rostopic pub /action/obs dofbot_pro_info/ObservData \\")
    print('       "color_image: {}" "depth_image: {}" "cam_intrinsics: [1,0,0,0,1,0,0,0,1]"')

    obs_sub = roslibpy.Topic(client, OBS_TOPIC, OBS_MSG_TYPE)
    obs_sub.subscribe(on_obs)
    time.sleep(2.0)   # let rosbridge propagate subscription
    print("   Subscribed — waiting ...")

    time.sleep(args.duration)
    obs_sub.unsubscribe()

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n── Results ──────────────────────────────────────────────────────")
    print(f"  Test 1 (lab→Jetson): published 5 messages — verify manually on Jetson")
    print(f"  Test 2 (Jetson→lab): received {obs_count[0]} observation(s) in {args.duration} s")

    if obs_count[0] == 0:
        print("\nWARNING: No observations received. Possible causes:")
        print("  - dofbot_pro_ws/devel/setup.bash not sourced before rosbridge was started")
        print("  - /action/obs topic name mismatch")
        print("  - ObservData binary not compiled (run catkin_make in dofbot_pro_ws)")
    else:
        print("\nSUCCESS: bi-directional transport confirmed.")

    client.terminate()


if __name__ == '__main__':
    main()
