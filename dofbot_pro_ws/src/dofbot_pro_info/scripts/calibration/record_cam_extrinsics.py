#!/usr/bin/env python3
"""Interactive helper to collect tag poses + FK pairs, compute averaged T_cam_base,
validate reprojection error, and write yaml/cam_extrinsics.yaml.

Usage:
  python3 record_cam_extrinsics.py --n 6 --out yaml/cam_extrinsics.yaml

The script will try to read the latest message from `/tag_detections` automatically
and fall back to asking you to paste the pose. For each tag pose accepted you will
be prompted to move the robot and enter the gripper FK (x y z in metres).
"""
import argparse
import os
import sys
import time
import yaml

import rospy
import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from dofbot_pro_info.srv import kinemarics, kinemaricsRequest
from apriltag_ros.msg import AprilTagDetectionArray


def try_extract_tag_pose(msg):
    """Try a few common attribute paths used by apriltag_ros messages.
    Return (t_vec, quat) or None if extraction fails.
    """
    dets = getattr(msg, 'detections', None)
    if dets and len(dets) > 0:
        det = dets[0]
        for first in ['pose', 'poses']:
            p = getattr(det, first, None)
            if p is None:
                continue
            if isinstance(p, (list, tuple)) and len(p) > 0:
                p = p[0]
            for _ in range(3):
                if p is None:
                    break
                if hasattr(p, 'pose'):
                    p = p.pose
                    continue
                break
            pos = getattr(p, 'position', None)
            ori = getattr(p, 'orientation', None)
            if pos is not None and ori is not None:
                t = np.array([pos.x, pos.y, pos.z], dtype=float)
                q = np.array([ori.x, ori.y, ori.z, ori.w], dtype=float)
                return t, q
    return None


def collect_pairs(n, auto_fk=False, fk_service_name='get_kinemarics', tag_topic='/unveiler/tag_detections'):
    pairs = []
    rospy.loginfo(f"Waiting for tag detections on {tag_topic}; will collect {n} pairs")
    fk_client = None
    if auto_fk:
        rospy.loginfo(f"Auto-FK enabled: will call service {fk_service_name} to obtain FK from current joints")
        try:
            rospy.wait_for_service(fk_service_name, timeout=5.0)
            fk_client = rospy.ServiceProxy(fk_service_name, kinemarics)
        except Exception as e:
            rospy.logwarn(f"FK service not available: {e}; falling back to manual FK input")
            fk_client = None
    while len(pairs) < n and not rospy.is_shutdown():
        print('\n--- Sample %d of %d ---' % (len(pairs) + 1, n))
        posed = None
        poll_start = time.time()
        while time.time() - poll_start < 5.0 and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message(tag_topic, AprilTagDetectionArray, timeout=1.0)
            except Exception:
                msg = None
                continue
            posed = try_extract_tag_pose(msg)
            if posed is not None:
                break

        if posed is not None:
            t_cam, q_cam = posed
            print('Auto-read tag pose from /tag_detections:')
            print('  tag_in_cam position (m):', t_cam)
            print('  tag_in_cam quat (xyzw):', q_cam)
            ans = input('Accept this detection? [Y/n]: ').strip().lower()
            if ans == 'n':
                continue
        else:
            print('No parsable /tag_detections message available.')
            s = input('Paste tag pose as "x y z qx qy qz qw" or type r to retry: ').strip()
            if s.lower() == 'r':
                continue
            try:
                parts = [float(x) for x in s.split()]
                if len(parts) != 7:
                    print('Expected 7 numbers, got', len(parts))
                    continue
                t_cam = np.array(parts[:3], dtype=float)
                q_cam = np.array(parts[3:], dtype=float)
            except Exception as e:
                print('Parse error:', e)
                continue

        t_base = None
        if auto_fk and fk_client is not None:
            try:
                js = rospy.wait_for_message('/joint_states', JointState, timeout=2.0)
                joints = list(js.position)
            except Exception:
                joints = []
            req = kinemaricsRequest()
            req.tar_x = 0.0; req.tar_y = 0.0; req.tar_z = 0.0
            req.Roll = 0.0; req.Pitch = 0.0; req.Yaw = 0.0
            import math
            defaults = [90.0, 90.0, 90.0, 0.0, 90.0, 30.0]
            for i in range(6):
                val = math.degrees(joints[i]) if i < len(joints) else defaults[i]
                setattr(req, f'cur_joint{i+1}', float(val))
            req.kin_name = 'fk'
            try:
                resp = fk_client(req)
                t_base = np.array([resp.x, resp.y, resp.z], dtype=float)
                print('Auto FK result (m):', t_base, ' rpy:', resp.Roll, resp.Pitch, resp.Yaw)
            except Exception as e:
                print('FK service call failed:', e)
                t_base = None

        if t_base is None:
            fk_line = input('Move robot over the tag and enter FK gripper position (x y z in metres): ').strip()
            try:
                fk_parts = [float(x) for x in fk_line.split()]
                if len(fk_parts) != 3:
                    print('Expected 3 numbers for FK position')
                    continue
                t_base = np.array(fk_parts, dtype=float)
            except Exception as e:
                print('Parse error:', e)
                continue

        pairs.append({'t_cam': t_cam.tolist(), 'q_cam': q_cam.tolist(), 't_base': t_base.tolist()})
        print('Saved pair #%d' % len(pairs))

    return pairs


def compute_T_from_pair(t_cam, q_cam, t_base):
    R_tag_cam = Rotation.from_quat(q_cam).as_matrix()
    T_tag_in_cam = np.eye(4)
    T_tag_in_cam[:3, :3] = R_tag_cam
    T_tag_in_cam[:3, 3] = t_cam
    T_tag_in_base = np.eye(4)
    T_tag_in_base[:3, 3] = t_base
    T_cam_base = T_tag_in_base @ np.linalg.inv(T_tag_in_cam)
    return T_cam_base


def average_transforms(T_list):
    translations = np.array([T[:3, 3] for T in T_list])
    mean_t = translations.mean(axis=0)
    rots = [Rotation.from_matrix(T[:3, :3]).as_quat() for T in T_list]
    Q = np.array(rots)
    q_mean = Q.sum(axis=0)
    q_mean = q_mean / np.linalg.norm(q_mean)
    R_mean = Rotation.from_quat(q_mean).as_matrix()
    T_avg = np.eye(4)
    T_avg[:3, :3] = R_mean
    T_avg[:3, 3] = mean_t
    return T_avg


def validate(T_avg, pairs):
    errs = []
    for p in pairs:
        t_cam = np.array(p['t_cam'])
        t_base = np.array(p['t_base'])
        p_cam = np.concatenate([t_cam, [1.0]])
        pred = (T_avg @ p_cam)[:3]
        err = np.linalg.norm(pred - t_base)
        errs.append(err)
    return np.array(errs)


def write_yaml(T, out_path):
    mat = T.tolist()
    data = {'T_cam_base': mat}
    ddir = os.path.dirname(out_path)
    if ddir and not os.path.exists(ddir):
        os.makedirs(ddir)
    with open(out_path, 'w') as f:
        yaml.safe_dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=6, help='number of pose pairs to collect')
    parser.add_argument('--out', type=str, default='yaml/cam_extrinsics.yaml', help='output yaml path')
    parser.add_argument('--auto-fk', action='store_true', help='call FK service automatically to get tag pose in base frame')
    parser.add_argument('--fk-service', type=str, default='get_kinemarics', help='name of the kinematics service to call for FK')
    parser.add_argument('--tag-topic', type=str, default='/unveiler/tag_detections', help='ROS topic to read AprilTag detections from')
    args = parser.parse_args()

    rospy.init_node('record_cam_extrinsics', anonymous=True)

    pairs = collect_pairs(args.n, auto_fk=args.auto_fk, fk_service_name=args.fk_service, tag_topic=args.tag_topic)
    if len(pairs) == 0:
        print('No pairs collected. Exiting.')
        sys.exit(1)

    Ts = [compute_T_from_pair(np.array(p['t_cam']), np.array(p['q_cam']), np.array(p['t_base'])) for p in pairs]
    T_avg = average_transforms(Ts)

    errs = validate(T_avg, pairs)
    print('\nReprojection errors (m) for each pair:')
    for i, e in enumerate(errs):
        print(f'  #{i+1}: {e:.4f}')
    print('Mean error: %.4f m' % errs.mean())

    save = input('Save averaged T_cam_base to %s ? [Y/n]: ' % args.out).strip().lower()
    if save != 'n':
        write_yaml(T_avg, args.out)


if __name__ == '__main__':
    main()
