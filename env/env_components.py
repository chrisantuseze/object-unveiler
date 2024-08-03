import os
import numpy as np
import math
import time
import pybullet as p

from utils.orientation import Quaternion, rot_z, rot_y
import utils.pybullet_utils as p_utils
from utils import urdf_editor
from utils.robotics import Trajectory

MOUNT_URDF_PATH = 'mount.urdf'
UR5_URDF_PATH = 'ur5e_bhand.urdf'
UR5_WORKSPACE_URDF_PATH = 'table/table.urdf'
PLANE_URDF_PATH = "plane/plane.urdf"

class Objects:
    def __init__(self,
                 name='',
                 pos=[0.0, 0.0, 0.0],
                 quat=Quaternion(),
                 size=[],
                 color=[1.0, 1.0, 1.0, 1.0],
                 body_id=None) -> None:
        
        self.name = name
        self.pos = pos
        self.quat = quat
        self.size = size
        self.color = color
        self.body_id = body_id

class FloatingBHand:
    """
    A moving mount and a gripper. The mount has 4 joints:
            0: prismatic x
            1: prismatic y
            2: prismatic z
            3: revolute z
    """

    def __init__(self,
                 robot_hand_urdf,
                 home_position,
                 simulation = None) -> None:
        
        # Define the mount link position w.r.t. hand base link.
        pos_offset = np.array([0.0, 0, -0.065])
        orn_offset = p.getQuaternionFromEuler([0, 0.0, 0.0])


        #################################################
        self.home_position = home_position

        # If there is no urdf file, generate the mounted-gripper urdf.
        self.mount_urdf = os.path.join('assets', MOUNT_URDF_PATH)
        mounted_urdf_name = "../assets/mounted_" + robot_hand_urdf.split('/')[-1].split('.')[0] + ".urdf"
        if not os.path.exists(mounted_urdf_name):
            self.generate_mounted_urdf(robot_hand_urdf, pos_offset, orn_offset)

        # rotation w.r.t. inertia frame
        self.home_quat = Quaternion.from_rotation_matrix(rot_y(-np.pi / 2))

        # Load robot hand urdf.
        self.robot_hand_id = p_utils.load_urdf(
            p,
            mounted_urdf_name,
            useFixedBase=True,
            basePosition=self.home_position,
            baseOrientation=self.home_quat.as_vector("xyzw")
        )

        # Mount joints.
        self.joint_ids = [0, 1, 2, 3]
        self.simulation = simulation


        #################################################
        pose = p_utils.get_link_pose('mount_link')
        p_utils.draw_pose(pose[0], pose[1])

        # Define force and speed (movement of mount).
        self.force = 10000
        self.speed = 0.01

        # Bhand joints.
        self.joint_names = ['bh_j11_joint', 'bh_j21_joint', 'bh_j12_joint', 'bh_j22_joint',
                            'bh_j32_joint', 'bh_j13_joint', 'bh_j23_joint', 'bh_j33_joint']
        self.indices = p_utils.get_joint_indices(self.joint_names, self.robot_hand_id)

        # Bhand links (for contact check).
        self.link_names = ['bh_base_link',
                           'bh_finger_32_link', 'bh_finger_33_link',
                           'bh_finger_22_link', 'bh_finger_23_link',
                           'bh_finger_12_link', 'bh_finger_13_link']
        self.link_indices = p_utils.get_link_indices(self.link_names, body_unique_id=self.robot_hand_id)
        self.distals = ['bh_finger_33_link', 'bh_finger_23_link', 'bh_finger_13_link']
        self.distal_indices = p_utils.get_link_indices(self.distals, body_unique_id=self.robot_hand_id)

        # Move fingers to home position.
        home_aperture_value = 0.6
        self.move_fingers(agent_cams=None, final_joint_values=[0.0, home_aperture_value, home_aperture_value, home_aperture_value])
        self.configure(n_links_before=4)


    def generate_mounted_urdf(self,
                              robot_hand_urdf,
                              pos_offset,
                              orn_offset):
        """
        Generates the urdf with a moving mount attached to a gripper.
        """

        # load gripper.
        robot_id = p_utils.load_urdf(
            p,
            robot_hand_urdf,
            flags=p.URDF_USE_SELF_COLLISION
        )

        # load mount.
        mount_body_id = p_utils.load_urdf(
            p,
            self.mount_urdf,
            useFixedBase=True
        )

        # combine mount and gripper by a joint.
        ed_mount = urdf_editor.UrdfEditor()
        ed_mount.initializeFromBulletBody(mount_body_id, 0)
        ed_gripper = urdf_editor.UrdfEditor()
        ed_gripper.initializeFromBulletBody(robot_id, 0)

        self.gripper_parent_index = 4  # 4 joints of mount
        new_joint = ed_mount.joinUrdf(
            childEditor=ed_gripper,
            parentLinkIndex=self.gripper_parent_index,
            jointPivotXYZInParent=pos_offset,
            jointPivotRPYInParent=p.getEulerFromQuaternion(orn_offset),
            jointPivotXYZInChild=[0, 0, 0],
            jointPivotRPYInChild=[0, 0, 0],
            parentPhysicsClientId=0,
            childPhysicsClientId=0
        )
        new_joint.joint_type = p.JOINT_FIXED
        new_joint.joint_name = "joint_mount_gripper"
        urdfname = "assets/mounted_" + robot_hand_urdf.split('/')[-1].split('.')[0] + ".urdf"
        ed_mount.saveUrdf(urdfname)

        # remove mount and gripper bodies.
        p.removeBody(mount_body_id)
        p.removeBody(robot_id)

    def move(self, agent_cams, target_pos, target_quat, duration=2.0, stop_at_contact=False):
        # compute translation
        affine_trans = np.eye(4)
        affine_trans[0:3, 0:3] = self.home_quat.rotation_matrix()
        affine_trans[0:3, 3] = self.home_position
        target_pos = np.matmul(np.linalg.inv(affine_trans), np.append(target_pos, 1.0))[0:3]

        # compute angle
        relative_rot = np.matmul(self.home_quat.rotation_matrix().transpose(), target_quat.rotation_matrix())
        angle = np.arctan2(relative_rot[2, 1], relative_rot[1, 1])
        target_states = [target_pos[0], target_pos[1], target_pos[2], angle]

        current_pos = []
        for i in self.joint_ids:
            current_pos.append(p.getJointState(0, i)[0])

        trajectories = []
        for i in range(len(self.joint_ids)):
            trajectories.append(Trajectory([0, duration], [current_pos[i], target_states[i]]))

        t = 0
        dt = 0.001
        interval = 0
        is_in_contact = False
        commands = []
        while t < duration:
            command = []
            vels = []

            for i in range(len(self.joint_ids)):
                command.append(trajectories[i].pos(t))
                vels.append(trajectories[i].vel(t)) #@Chris

            # print("command", len(command), command)
            p.setJointMotorControlArray(
                self.robot_hand_id,
                self.joint_ids,
                p.POSITION_CONTROL,
                targetPositions=command,
                forces=[100 * self.force] * len(self.joint_ids),
                positionGains=[100 * self.speed] * len(self.joint_ids)
            )

            if stop_at_contact:
                points = p.getContactPoints(bodyA=self.robot_hand_id)
                if len(points) > 0:
                    for pnt in points:
                        if pnt[9] > 0:
                            is_in_contact = True
                            break

                if is_in_contact:
                    break

            t += dt
            self.simulation.step()
            time.sleep(dt)

            if interval % 10 == 0:
                #@Chris we save the images at the beginning of the trajectory
                images = {'color': []}
                for cam in agent_cams:
                    color, depth, seg = cam.get_data() 
                    images['color'].append(color)
                    # cv2.imwrite(os.path.join("save/misc", "color.png"), color)

                commands.append((command, vels, images))
            interval += 1

        return commands, is_in_contact

    def set_hand_joint_position(self, joint_position, force):
        for i in range(len(self.joint_names)):
            if self.joint_names[i] in ['bh_j32_joint', 'bh_j33_joint']:
                apply_force = 1.7 * force 
            else:
                apply_force = force

            p.setJointMotorControl2(bodyUniqueId=0,
                                    jointIndex=self.indices[i],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_position[i],
                                    force=apply_force)
            
    def move_fingers(self, agent_cams=None, final_joint_values=[], duration=1, force=2):
        """
        Move fingers while keeping the hand to the same pose
        """

        # get current joint positions
        current_pos = []
        for i in self.indices:
            current_pos.append(p.getJointState(0, i)[0])

        # print("Current joint pos:", current_pos) #@Chris

        hand_pos = []
        for i in self.joint_ids:
            hand_pos.append(p.getJointState(0, i)[0])

        # print("Hand mount joint pos:", hand_pos) #@Chris

        final = [final_joint_values[0], final_joint_values[0], final_joint_values[1],
                 final_joint_values[2], final_joint_values[3], final_joint_values[1]/3,
                 final_joint_values[2]/3, final_joint_values[3]/3]
        
        trajectories = []
        for i in range(len(self.indices)):
            trajectories.append(Trajectory([0, duration], [current_pos[i], final[i]]))

        t = 0
        dt = 0.001
        interval = 0
        commands = []
        while t < duration:
            command = []
            vels = []

            for i in range(len(self.joint_names)):
                command.append(trajectories[i].pos(t))
                vels.append(trajectories[i].vel(t))

            self.set_hand_joint_position(command, force)

            # keep hand in the same pose
            p.setJointMotorControlArray(
                self.robot_hand_id,
                self.joint_ids,
                p.POSITION_CONTROL,
                targetPositions=hand_pos,
                forces=[100 * self.force] * len(self.joint_ids),
                positionGains=[100 * self.speed] * len(self.joint_ids)
            )

            t += dt
            self.simulation.step()
            time.sleep(dt)

            if not agent_cams:
                return commands

            if interval % 10 == 0:
                #@Chris we save the images at the beginning of the trajectory
                images = {'color': []}
                for cam in agent_cams:
                    color, depth, seg = cam.get_data() 
                    images['color'].append(color)
                    # cv2.imwrite(os.path.join("save/misc", "color.png"), color)

                commands.append((command, vels, images))
            interval += 1

        return commands#, [current_pos, hand_pos] #@Chris

    def step_constraints(self):
        current_pos = []
        for i in self.indices:
            current_pos.append(p.getJointState(0, i)[0])

        p.setJointMotorControlArray(
            self.robot_hand_id,
            self.indices,
            p.POSITION_CONTROL,
            targetPositions=current_pos,
            forces=[100 * self.force] * len(self.indices),
            positionGains=[100 * self.speed] * len(self.indices)
        )

    def close(self, agent_cams, joint_vals=[0.0, 1.8, 1.8, 1.8], duration=2):
        return self.move_fingers(agent_cams=agent_cams, final_joint_values=joint_vals)

    def open(self, agent_cams, joint_vals=[0.0, 0.6, 0.6, 0.6]):
        return self.move_fingers(agent_cams=agent_cams, final_joint_values=joint_vals, duration=1)

    def configure(self, n_links_before):
        # set friction coefficients for gripper fingers

        for i in range(n_links_before, p.getNumJoints(self.robot_hand_id)):
            p.changeDynamics(self.robot_hand_id, i, lateralFriction=1.0,
                             spinningFriction=1.0, rollingFriction=0.0001,
                             frictionAnchor=True)
            
    def is_grasp_stable(self):
        distal_contacts = 0
        for link_id in self.distal_indices:
            contacts = p.getContactPoints(bodyA=self.robot_hand_id, linkIndexA=link_id)
            distal_contacts += len(contacts)

        body_b = []
        total_contacts = 0
        for link_id in self.link_indices:
            contacts = p.getContactPoints(bodyA=self.robot_hand_id, linkIndexA=link_id)
            if len(contacts) == 0:
                continue
            for pnt in contacts:
                body_b.append(pnt[2])
            total_contacts += len(contacts)

        if distal_contacts == total_contacts or len(np.unique(body_b)) != 1:
            return False, total_contacts
        elif distal_contacts > total_contacts:
            assert (distal_contacts > total_contacts)
        else:
            return True, total_contacts
        

    def move_robot(self, joint_positions):
        p.setJointMotorControlArray(
            self.robot_hand_id,
            self.joint_ids,
            p.POSITION_CONTROL,
            targetPositions=joint_positions,
            forces=[100 * self.force] * len(self.joint_ids),
            positionGains=[100 * self.speed] * len(self.joint_ids)
        )

    def calculate_joint_positions(self, action, current_state, duration, t):
        target_pos = action['pos']
        target_quat = action['quat']

        if current_state == ActionState.MOVE_ABOVE_PREGRASP:
            target_pos = target_pos.copy()
            target_pos[2] += 0.3

        if current_state == ActionState.POWER_PUSH:
            rot = target_quat.rotation_matrix()
            target_pos = target_pos + rot[0:3, 2] * action['push_distance']

        if current_state == ActionState.MOVE_UP:
            target_pos = target_pos.copy()
            target_pos[2] += 0.4

        if current_state == ActionState.MOVE_HOME:
            target_pos = self.home_position
            target_quat = self.home_quat

        # Compute translation
        affine_trans = np.eye(4)
        affine_trans[0:3, 0:3] = self.home_quat.rotation_matrix()
        affine_trans[0:3, 3] = self.home_position
        target_pos = np.matmul(np.linalg.inv(affine_trans), np.append(target_pos, 1.0))[0:3]

        # Compute angle
        relative_rot = np.matmul(self.home_quat.rotation_matrix().transpose(), target_quat.rotation_matrix())
        angle = np.arctan2(relative_rot[2, 1], relative_rot[1, 1])

        # Combine position and angle
        target_states = [target_pos[0], target_pos[1], target_pos[2], angle]

        current_pos = []
        for i in self.joint_ids:
            current_pos.append(p.getJointState(0, i)[0])

        trajectories = []
        for i in range(len(self.joint_ids)):
            trajectories.append(Trajectory([0, duration], [current_pos[i], target_states[i]]))


        joint_positions = []
        for i in range(len(self.joint_ids)):
            joint_positions.append(trajectories[i].pos(t))

        return joint_positions

    def calculate_finger_positions(self, action, current_state, duration, t, force=2):
        if current_state == ActionState.CLOSE_FINGERS:
            joint_vals = [0.0, 1.8, 1.8, 1.8]

        if current_state == ActionState.OPEN_FINGERS:
            joint_vals = [0.0, 0.6, 0.6, 0.6]
            
        if current_state == ActionState.SET_FINGER_CONFIG:
            theta = action['aperture']
            joint_vals = [0.0, theta, theta, theta]

         # get current joint positions
        current_pos = []
        for i in self.indices:
            current_pos.append(p.getJointState(0, i)[0])

        # print("Current joint pos:", current_pos) #@Chris

        hand_pos = []
        for i in self.joint_ids:
            hand_pos.append(p.getJointState(0, i)[0])

        final = [
            joint_vals[0], joint_vals[0],  # Base joint
            joint_vals[1], joint_vals[2], joint_vals[3],  # Main finger joints
            joint_vals[1]/3, joint_vals[2]/3, joint_vals[3]/3  # Secondary finger joints
        ]

        trajectories = []
        for i in range(len(self.indices)):
            trajectories.append(Trajectory([0, duration], [current_pos[i], final[i]]))

        finger_positions = []
        for i in range(len(self.joint_names)):
            finger_positions.append(trajectories[i].pos(t))

        self.set_hand_joint_position(finger_positions, force)

        return hand_pos

class ActionState:
    MOVE_ABOVE_PREGRASP = (0, 0.1)
    SET_FINGER_CONFIG = (1, 0.1)
    MOVE_TO_PREGRASP = (2, 0.5)
    POWER_PUSH = (3, 2.0)
    CLOSE_FINGERS = (4, 1.0)
    MOVE_UP = (5, 0.1)
    MOVE_HOME = (6, 0.1)
    OPEN_FINGERS = (7, 0.1)