import os
import math
import time
import cv2

import pybullet as p
import numpy as np
import random

from utils.constants import TRAIN_DIR
from utils.orientation import Affine3, Quaternion, rot_z
import utils.pybullet_utils as p_utils
import utils.general_utils as general_utils
import env.env_components as env_components
from env.env_components import ActionState, AdaptiveActionState
import env.cameras as cameras
import utils.logger as logging

UR5_URDF_PATH = 'ur5e_bhand.urdf'
UR5_WORKSPACE_URDF_PATH = 'table/table.urdf'
PLANE_URDF_PATH = "plane/plane.urdf"
  

class Simulation:
    def __init__(self, objects):
        self.names_button = None #NamesButton('Show names') TODO: Uncomment
        self.objects = objects

    def step(self):
        p.stepSimulation()
        # self.names_button.show_names(self.objects) TODO: Uncomment

class Environment:
    def __init__(self, params, assets_root = "assets/", objects_set="seen") -> None:
        self.objects = []

        self.assets_root = assets_root
        self.objects_set = objects_set
        self.workspace_pos = np.array([0.0, 0.0, 0.0])
        hz = 240

        self.rotations = params['agent']['fcn']['rotations']
        self.aperture_limits = params['agent']['regressor']['aperture_limits']
        self.pxl_size = params['env']['pixel_size']
        self.bounds = np.array(params['env']['workspace']['bounds'])

        # self.nr_objects = [11, 15] #[5, 8]
        # self.nr_objects = [2, 4]
        # self.nr_objects = [4, 6]
        # self.nr_objects = [4, 8]
        self.nr_objects = [6, 9]
        # self.nr_objects = [2, 9]
        # self.nr_objects = [9, 12]


        # Setup cameras.
        self.agent_cams = []
        for config in cameras.RealSense.CONFIG:
            config_world = config.copy()
            config_world['pos'] = self.workspace2world(config['pos'])[0]
            config_world['target_pos'] = self.workspace2world(config['target_pos'])[0]
            self.agent_cams.append(cameras.SimCamera(config_world))

        self.bhand = None

        self.obj_files = []

        objects_path = os.path.join('objects', self.objects_set)
        objects_files = os.listdir(os.path.join(self.assets_root, objects_path))
        for obj_file in objects_files:
            if not obj_file.endswith('.obj'):
                continue
            self.obj_files.append(os.path.join(self.assets_root, objects_path, obj_file))

        self.rng = np.random.RandomState()

        # p.connect(p.DIRECT)
        p.connect(p.GUI)
        # Move default camera closer to the scene.
        target = np.array(self.workspace_pos)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.75,
            cameraYaw=180,
            cameraPitch=-45,
            cameraTargetPosition=target)
        
        p.setAdditionalSearchPath(self.assets_root)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setTimeStep(1.0 / hz)
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setRealTimeSimulation(0)  # we want to be faster than real time : #@Chris


        self.simulation = Simulation(self.objects)
        self.singulation_condition = False

        # self.current_state = ActionState.MOVE_ABOVE_PREGRASP
        self.current_state = AdaptiveActionState.MOVE_ABOVE_PREGRASP
        self.state_start_time = time.time()
        self.elapsed_time = self.interval = 0
        self.trajectories = []

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # Load UR5 robot arm equipped with Barrett hand.
        self.bhand = env_components.FloatingBHand('assets/robot_hands/barrett/bh_282.urdf',
                                    home_position=np.array([0.7, 0.0, 0.2]),
                                    simulation=self.simulation)
        
        p_utils.load_urdf(p, PLANE_URDF_PATH, [0, 0, -0.7])
        table_id = p_utils.load_urdf(p, UR5_WORKSPACE_URDF_PATH, self.workspace_pos)
        p.changeDynamics(table_id, -1, lateralFriction=0.1)


        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Generate a scene with randomly placed objects.
        self.objects = []
        self.add_objects()

        self.simulation.objects = self.objects

        t = 0
        while t < 100: # TODO why is it 100?
            self.simulation.step()
            t += 1

        # remove flat objects
        self.remove_flat_objs()

        # pack objects closer to the middle
        self.centralize_objs(force_magnitude=1)#2)

        # remove flat objects
        self.remove_flat_objs()

        while not self.is_moving():
            time.sleep(0.001)
            self.simulation.step()

        logging.info(">>>>>>>>>> Scene building complete >>>>>>>>>>")
        general_utils.recreate_train()

        obs = self.get_observation()
        images = {'color': []}
        for cam in self.agent_cams:
            color, depth, seg = cam.get_data() 
            images['color'].append(color)
        obs['traj_data'] = [([random.random() for _ in range(8)], images)]
        return obs

    def get_observation(self):
        obs = {'color': [], 'depth': [], 'seg': [], 'full_state': [], 'traj_data': []}

        for cam in self.agent_cams:
            color, depth, seg = cam.get_data() 
            obs['color'].append(color)
            obs['depth'].append(depth)
            obs['seg'].append(seg)

        # update the objects' pos and orient
        tmp_objs = []
        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            obj.pos, obj.quat = self.workspace2world(np.array(pos), Quaternion(x=quat[0], 
                                                                               y=quat[1], z=quat[2], w=quat[3]), inv=True)
            
            if obj.pos[2] > 0:
                tmp_objs.append(obj)

        self.objects = tmp_objs.copy()
        obs['full_state'] = self.objects

        return obs
    
    def get_hand_finger_pose(self):
        finger_current_pos = []
        for i in self.bhand.indices:
            finger_current_pos.append(p.getJointState(0, i)[0])

        hand_current_pos = []
        for i in self.bhand.joint_ids_act:
            hand_current_pos.append(p.getJointState(0, i)[0])

        return hand_current_pos, finger_current_pos
    
    def calculate_position_error(self, current_positions, target_positions):
        """Calculate the euclidean distance between current and target positions"""
        # If dealing with joint positions
        if isinstance(current_positions, list) and isinstance(target_positions, list):
            return np.sqrt(np.sum([(current_positions[i] - target_positions[i])**2 
                                for i in range(len(current_positions))]))
        
        # If dealing with 3D cartesian positions
        return np.linalg.norm(np.array(current_positions) - np.array(target_positions))
    
    def step_act(self, action, eval=True):
        dt = 0.001
        
        if self.current_state == AdaptiveActionState.MOVE_ABOVE_PREGRASP:
            state_id, min_duration, threshold = AdaptiveActionState.MOVE_ABOVE_PREGRASP
            
            if self.elapsed_time == 0:
                self.hand_current_pos, _ = self.get_hand_finger_pose()
                self.target_pos = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=min_duration)

            # Move with position error check
            joint_positions = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=self.elapsed_time)
            self.bhand.move_robot(joint_positions)
            self.elapsed_time += dt
            
            # Check if position reached or minimum time elapsed
            pos_error = self.calculate_position_error(joint_positions, self.target_pos)
            min_time_reached = self.elapsed_time >= min_duration * 0.5  # At least 50% of min time
            
            if (min_time_reached and pos_error < threshold) or self.elapsed_time >= min_duration:
                self.current_state = AdaptiveActionState.SET_FINGER_CONFIG
                self.elapsed_time = 0
        
        elif self.current_state == AdaptiveActionState.SET_FINGER_CONFIG:
            state_id, min_duration, threshold = AdaptiveActionState.SET_FINGER_CONFIG
            
            if self.elapsed_time == 0:
                self.hand_current_pos, self.finger_current_pos = self.get_hand_finger_pose()
                self.target_finger_pos = self.bhand.calculate_finger_positions(action, self.current_state, self.finger_current_pos, t=min_duration, force=5)

            joint_positions = self.bhand.calculate_finger_positions(action, self.current_state, self.finger_current_pos, t=self.elapsed_time, force=5)
            self.bhand.move_robot(self.hand_current_pos)
            self.elapsed_time += dt
            
            finger_error = self.calculate_position_error(joint_positions, self.target_finger_pos)
            min_time_reached = self.elapsed_time >= min_duration * 0.5
            
            if (min_time_reached and finger_error < threshold) or self.elapsed_time >= min_duration:
                self.current_state = AdaptiveActionState.MOVE_TO_PREGRASP
                self.elapsed_time = 0
        
        elif self.current_state == AdaptiveActionState.MOVE_TO_PREGRASP:
            state_id, min_duration, threshold = AdaptiveActionState.MOVE_TO_PREGRASP
            
            if self.elapsed_time == 0:
                self.hand_current_pos, _ = self.get_hand_finger_pose()
                self.target_pos = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=min_duration)

            joint_positions = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=self.elapsed_time)
            self.bhand.move_robot(joint_positions)
            self.elapsed_time += dt

            self.is_in_contact = self.bhand.check_in_contact()
            if self.is_in_contact:
                print("Collision occurred!")
                self.current_state = AdaptiveActionState.MOVE_UP
                self.elapsed_time = 0

                obs = self.get_observation()
                obs['traj_data'] = [(joint_positions, {'color': []})]
                return obs, {'collision': self.is_in_contact, 'stable': None, 'num_contacts': None, 'eoe': False}

            pos_error = self.calculate_position_error(joint_positions, self.target_pos)
            min_time_reached = self.elapsed_time >= min_duration * 0.5
            
            if (min_time_reached and pos_error < threshold) or self.elapsed_time >= min_duration:
                self.current_state = AdaptiveActionState.POWER_PUSH
                self.elapsed_time = 0

        elif self.current_state == AdaptiveActionState.POWER_PUSH:
            state_id, min_duration, threshold = AdaptiveActionState.POWER_PUSH
            
            if self.elapsed_time == 0:
                self.hand_current_pos, _ = self.get_hand_finger_pose()
                self.target_pos = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=min_duration)

            joint_positions = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=self.elapsed_time)
            self.bhand.move_robot(joint_positions)
            self.elapsed_time += dt

            pos_error = self.calculate_position_error(joint_positions, self.target_pos)
            min_time_reached = self.elapsed_time >= min_duration * 0.5
            
            if (min_time_reached and pos_error < threshold) or self.elapsed_time >= min_duration:
                self.current_state = AdaptiveActionState.CLOSE_FINGERS
                self.elapsed_time = 0

        elif self.current_state == AdaptiveActionState.CLOSE_FINGERS:
            state_id, min_duration, threshold = AdaptiveActionState.CLOSE_FINGERS
            
            if self.elapsed_time == 0:
                self.hand_current_pos, self.finger_current_pos = self.get_hand_finger_pose()
                self.target_finger_pos = self.bhand.calculate_finger_positions(action, self.current_state, self.finger_current_pos, t=min_duration)

            joint_positions = self.bhand.calculate_finger_positions(action, self.current_state, self.finger_current_pos, t=self.elapsed_time)
            self.bhand.move_robot(self.hand_current_pos)
            self.elapsed_time += dt
            
            finger_error = self.calculate_position_error(joint_positions, self.target_finger_pos)
            min_time_reached = self.elapsed_time >= min_duration * 0.5
            
            if (min_time_reached and finger_error < threshold) or self.elapsed_time >= min_duration:
                self.current_state = AdaptiveActionState.MOVE_UP
                self.elapsed_time = 0

        elif self.current_state == AdaptiveActionState.MOVE_UP:
            state_id, min_duration, threshold = AdaptiveActionState.MOVE_UP
            
            if self.elapsed_time == 0:
                self.hand_current_pos, _ = self.get_hand_finger_pose()
                self.target_pos = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=min_duration)

            joint_positions = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=self.elapsed_time)
            self.bhand.move_robot(joint_positions)
            self.elapsed_time += dt

            pos_error = self.calculate_position_error(joint_positions, self.target_pos)
            min_time_reached = self.elapsed_time >= min_duration * 0.5
            
            if (min_time_reached and pos_error < threshold) or self.elapsed_time >= min_duration:
                self.current_state = AdaptiveActionState.GRASP_STABILITY
                self.elapsed_time = 0

        elif self.current_state == AdaptiveActionState.GRASP_STABILITY:
            state_id, min_duration, threshold = AdaptiveActionState.GRASP_STABILITY
            
            if self.elapsed_time == 0:
                self.hand_current_pos, _ = self.get_hand_finger_pose()
                self.target_pos = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=min_duration)

            joint_positions = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=self.elapsed_time)
            self.bhand.move_robot(joint_positions)
            self.elapsed_time += dt

            pos_error = self.calculate_position_error(joint_positions, self.target_pos)
            min_time_reached = self.elapsed_time >= min_duration * 0.5
            
            if (min_time_reached and pos_error < threshold) or self.elapsed_time >= min_duration:
                self.current_state = AdaptiveActionState.MOVE_HOME
                self.elapsed_time = 0

        elif self.current_state == AdaptiveActionState.MOVE_HOME:
            state_id, min_duration, threshold = AdaptiveActionState.MOVE_HOME
            
            if self.elapsed_time == 0:
                self.hand_current_pos, _ = self.get_hand_finger_pose()
                self.target_pos = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=min_duration)

            joint_positions = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=self.elapsed_time)
            self.bhand.move_robot(joint_positions)
            self.elapsed_time += dt

            pos_error = self.calculate_position_error(joint_positions, self.target_pos)
            min_time_reached = self.elapsed_time >= min_duration * 0.5
            
            if (min_time_reached and pos_error < threshold) or self.elapsed_time >= min_duration:
                self.current_state = AdaptiveActionState.OPEN_FINGERS
                self.elapsed_time = 0

        elif self.current_state == AdaptiveActionState.OPEN_FINGERS:
            state_id, min_duration, threshold = AdaptiveActionState.OPEN_FINGERS
            
            if self.elapsed_time == 0:
                self.hand_current_pos, self.finger_current_pos = self.get_hand_finger_pose()
                self.target_finger_pos = self.bhand.calculate_finger_positions(action, self.current_state, self.finger_current_pos, t=min_duration)

            joint_positions = self.bhand.calculate_finger_positions(action, self.current_state, self.finger_current_pos, t=self.elapsed_time)
            self.bhand.move_robot(self.hand_current_pos)
            self.elapsed_time += dt
            
            finger_error = self.calculate_position_error(joint_positions, self.target_finger_pos)
            min_time_reached = self.elapsed_time >= min_duration * 0.5
            
            if (min_time_reached and finger_error < threshold) or self.elapsed_time >= min_duration:
                self.current_state = AdaptiveActionState.MOVE_ABOVE_PREGRASP
                self.elapsed_time = 0

                images = {'color': []}
                for cam in self.agent_cams:
                    color, depth, seg = cam.get_data() 
                    images['color'].append(color)
                obs = self.get_observation()
                obs['traj_data'] = [(joint_positions, images)]

                stable_grasp, num_contacts = self.bhand.is_grasp_stable()

                return obs, {'collision': self.is_in_contact, 'stable': stable_grasp, 'num_contacts': num_contacts, 'eoe': True} # eoe = end of episode
            

        # Step the simulation
        self.bhand.simulation.step()
        time.sleep(dt)
        
        # Return observation and info
        obs = self.get_observation()
        
        # Add trajectory data if needed
        images = {'color': []}
        if not eval and self.interval % 20 == 0:
            for cam in self.agent_cams:
                color, depth, seg = cam.get_data() 
                images['color'].append(general_utils.resize_image(color))
        else:
            images['color'].append(None)
        
        obs['traj_data'] = [(joint_positions, images)]
        self.interval += 1
        
        return obs,  {'collision': None, 'stable': None, 'num_contacts': None, 'eoe': False}
            
    
    def step_act_old(self, action, eval=True):
        # print("Executing action...", self.current_state)
        dt = 0.001
        
        if self.current_state == ActionState.MOVE_ABOVE_PREGRASP:
            if self.elapsed_time == 0:
                self.hand_current_pos, _ = self.get_hand_finger_pose()

            length = ActionState.MOVE_ABOVE_PREGRASP[1]
            if self.elapsed_time < length:
                joint_positions = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=self.elapsed_time)
                self.bhand.move_robot(joint_positions)
                self.elapsed_time += dt

            if self.elapsed_time >= length:
                self.current_state = ActionState.SET_FINGER_CONFIG
                self.elapsed_time = 0
        
        elif self.current_state == ActionState.SET_FINGER_CONFIG:
            if self.elapsed_time == 0:
                self.hand_current_pos, self.finger_current_pos = self.get_hand_finger_pose()

            length = ActionState.SET_FINGER_CONFIG[1]
            if self.elapsed_time < length:
                joint_positions = self.bhand.calculate_finger_positions(action, self.current_state, self.finger_current_pos, t=self.elapsed_time, force=5)
                self.bhand.move_robot(self.hand_current_pos)
                self.elapsed_time += dt

            if self.elapsed_time >= length:
                self.current_state = ActionState.MOVE_TO_PREGRASP
                self.elapsed_time = 0
        
        elif self.current_state == ActionState.MOVE_TO_PREGRASP:
            if self.elapsed_time == 0:
                self.hand_current_pos, _ = self.get_hand_finger_pose()

            length = ActionState.MOVE_TO_PREGRASP[1]
            if self.elapsed_time < length:
                joint_positions = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=self.elapsed_time)
                self.bhand.move_robot(joint_positions)
                self.elapsed_time += dt

                self.is_in_contact = self.bhand.check_in_contact()
                if self.is_in_contact:
                    print("Collision occurred!")
                    self.current_state = ActionState.MOVE_UP
                    self.elapsed_time = 0

            if self.elapsed_time >= length and not self.is_in_contact:
                self.current_state = ActionState.POWER_PUSH
                self.elapsed_time = 0
        
        elif self.current_state == ActionState.POWER_PUSH:
            if self.elapsed_time == 0:
                self.hand_current_pos, _ = self.get_hand_finger_pose()

            length = ActionState.POWER_PUSH[1]
            if self.elapsed_time < length:
                joint_positions = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=self.elapsed_time)
                self.bhand.move_robot(joint_positions)
                self.elapsed_time += dt

            if self.elapsed_time >= length:
                self.current_state = ActionState.CLOSE_FINGERS
                self.elapsed_time = 0
        
        elif self.current_state == ActionState.CLOSE_FINGERS:
            if self.elapsed_time == 0:
                self.hand_current_pos, self.finger_current_pos = self.get_hand_finger_pose()

            length = ActionState.CLOSE_FINGERS[1]
            if self.elapsed_time < length:
                joint_positions = self.bhand.calculate_finger_positions(action, self.current_state, self.finger_current_pos, t=self.elapsed_time)
                self.bhand.move_robot(self.hand_current_pos)
                self.elapsed_time += dt

            if self.elapsed_time >= length:
                self.current_state = ActionState.MOVE_UP
                self.elapsed_time = 0
        
        elif self.current_state == ActionState.MOVE_UP:
            if self.elapsed_time == 0:
                self.hand_current_pos, _ = self.get_hand_finger_pose()

            length = ActionState.MOVE_UP[1]
            if self.elapsed_time < length:
                joint_positions = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=self.elapsed_time)
                self.bhand.move_robot(joint_positions)
                self.elapsed_time += dt

            if self.elapsed_time >= length:
                self.current_state = ActionState.GRASP_STABILITY
                self.elapsed_time = 0

        elif self.current_state == ActionState.GRASP_STABILITY:
            if self.elapsed_time == 0:
                self.hand_current_pos, _ = self.get_hand_finger_pose()

            length = ActionState.GRASP_STABILITY[1]
            if self.elapsed_time < length:
                joint_positions = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=self.elapsed_time)
                self.bhand.move_robot(joint_positions)
                self.elapsed_time += dt

            if self.elapsed_time >= length:
                self.current_state = ActionState.MOVE_HOME
                self.elapsed_time = 0
        
        elif self.current_state == ActionState.MOVE_HOME:
            if self.elapsed_time == 0:
                self.hand_current_pos, _ = self.get_hand_finger_pose()

            length = ActionState.MOVE_HOME[1]
            if self.elapsed_time < length:
                joint_positions = self.bhand.calculate_joint_positions(action, self.current_state, self.hand_current_pos, t=self.elapsed_time)
                self.bhand.move_robot(joint_positions)
                self.elapsed_time += dt

            if self.elapsed_time >= length:
                self.current_state = ActionState.OPEN_FINGERS
                self.elapsed_time = 0
        
        elif self.current_state == ActionState.OPEN_FINGERS:
            if self.elapsed_time == 0:
                self.hand_current_pos, self.finger_current_pos = self.get_hand_finger_pose()

            length = ActionState.OPEN_FINGERS[1]
            if self.elapsed_time < length:
                joint_positions = self.bhand.calculate_finger_positions(action, self.current_state, self.finger_current_pos, t=self.elapsed_time)
                self.bhand.move_robot(self.hand_current_pos)
                self.elapsed_time += dt

            if self.elapsed_time >= length:
                # Action sequence complete
                self.current_state = ActionState.MOVE_ABOVE_PREGRASP
                self.elapsed_time = 0

                images = {'color': []}
                for cam in self.agent_cams:
                    color, depth, seg = cam.get_data() 
                    images['color'].append(color)
                obs = self.get_observation()
                obs['traj_data'] = [(joint_positions, images)]

                stable_grasp, num_contacts = self.bhand.is_grasp_stable()

                return obs, {'collision': self.is_in_contact, 'stable': stable_grasp, 'num_contacts': num_contacts, 'eoe': True} # eoe = end of episode
            

        # Step the simulation
        self.bhand.simulation.step()
        time.sleep(dt)

        obs = self.get_observation()

        # if save_traj_data:
        #@Chris we save the images at the beginning of the trajectory
        images = {'color': []}
        # if eval or self.interval % 20 == 0:
        if not eval and self.interval % 20 == 0:
            for cam in self.agent_cams:
                color, depth, seg = cam.get_data() 
                images['color'].append(general_utils.resize_image(color))
        else:
            images['color'].append(None)
        
        obs['traj_data'] = [(joint_positions, images)]
        self.interval += 1
        
        # Return intermediate observation and info
        return obs,  {'collision': None, 'stable': None, 'num_contacts': None, 'eoe': False}

    
    def step(self, action):
        trajectories = []
            
        # move hand above the pre-grasp position
        pre_grasp_pos = action['pos'].copy()
        pre_grasp_pos[2] += 0.3

        _ = self.bhand.move(pre_grasp_pos, action['quat'], duration=0.1)

        # set finger configuration
        theta = action['aperture']
        self.bhand.move_fingers([0.0, theta, theta, theta], duration=0.1, force=5)

        # move to the pre-grasp position
        is_in_contact = self.bhand.move(action['pos'], action['quat'], duration=0.5, stop_at_contact=True)
        
        # check if during reaching the pre-grasp position, the hand collides with some objects
        if not is_in_contact:
            
            # compute the distances of each object from other objects
            obs = self.get_observation()
            distances = p_utils.get_distances_from_target(obs)

            # push the hand forward
            rot = action['quat'].rotation_matrix()
            grasp_pos = action['pos'] + rot[0:3, 2] * action['push_distance']

            # if grasping only (without power push), comment out
            self.bhand.move(grasp_pos, action['quat'], duration=2)

            # compute the distances of each object from other objects after pushing
            next_obs = self.get_observation()
            next_distances = p_utils.get_distances_from_target(next_obs)

            # compute the difference between the distances (singulation distance, see paper)
            diffs = {}
            for obj_id in distances:
                if obj_id in next_distances and obj_id in distances:
                    diffs[obj_id] = next_distances[obj_id] - distances[obj_id]

            
            # close the fingers
            self.bhand.close()
        else:
            grasp_pos = action['pos']

        # move up when the object is picked
        final_pos = grasp_pos.copy()
        final_pos[2] += 0.4
        _ = self.bhand.move(final_pos, action['quat'], duration=0.1)

        # check grasp stability
        _ = self.bhand.move(final_pos, action['quat'], duration=0.05)

        stable_grasp, num_contacts = self.bhand.is_grasp_stable()

        # check the validity of the grasp
        grasp_label = stable_grasp

        # Filter stable grasps. Keep the ones that created space around the grasped object.
        # If there is an object above the table (grasped and remained in the hand) and the push-grasping
        # increased the distance of the grasped objects from others, then count it as a successful

        self.singulation_condition = False  # I do not want the singulation condition to hold
        if self.singulation_condition and stable_grasp:
            for obj in self.objects:
                pos, _ = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
                if pos[2] > 0.25 and diffs[obj.body_id] > 0.015:
                    grasp_label = True
                    break
                else:
                    grasp_label = False

        # move home
        _ = self.bhand.move(self.bhand.home_position, action['quat'], duration=0.1)

        # open fingers to drop grasped object
        self.bhand.open()

        self.remove_flat_objs()

        obs = self.get_observation()
        obs['traj_data'] = trajectories

        return obs, {'collision': is_in_contact,
                                'stable': grasp_label,
                                'num_contacts': num_contacts}

    def add_single_object(self, obj_path, pos, quat, size):
        """
        Adds an object to the scene
        """

        base_pos, base_orient = self.workspace2world(pos, quat)
        body_id = p_utils.load_obj(obj_path, scaling=1.0, position=base_pos, orientation=base_orient.as_vector("xyzw"))

        return env_components.Objects(name=obj_path.split('/')[-1].split('.')[0],
                                pos=base_pos, quat=base_orient, size=size, body_id=body_id)
    
    def add_objects(self):
        
        def get_pxl_distance(meters):
            return meters/self.pxl_size
        
        def get_xyz(pxl, obj_size):
            x = -(self.pxl_size * pxl[0] - self.bounds[0, 1])
            y = self.pxl_size * pxl[1] - self.bounds[1, 1]
            z = obj_size[2]
            return np.array([x, y, z])
        
        nr_objs = self.rng.randint(low=self.nr_objects[0], high=self.nr_objects[1])

        # nr_objs = 6

        obj_paths = self.rng.choice(self.obj_files, nr_objs)

        self.scene_nr_objs= nr_objs

        for i in range(len(obj_paths)):
            obj = env_components.Objects()
            base_pos, base_orient = self.workspace2world(np.array([1.0, 1.0, 0.0]), Quaternion())
            body_id = p_utils.load_obj(obj_path=obj_paths[i], scaling=1.0, position=base_pos, orientation=base_orient.as_vector("xyzw"))

            obj.body_id = body_id
            size = (np.array(p.getAABB(body_id)[1]) - np.array(p.getAABB(body_id)[0])) / 2.0
            max_size = np.sqrt(size[0] **2 + size[1] **2)
            erode_size = int(np.round(get_pxl_distance(meters=max_size)))

            obs = self.get_observation()
            state = general_utils.get_fused_heightmap(obs, cameras.RealSense.CONFIG, self.bounds, self.pxl_size)

            free = np.zeros(state.shape, dtype=np.uint8)
            free[state == 0] = 1
            free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
            free[0:10] = 0
            free[90:100] = 0
            free[:, 0:10] = 0
            free[:, 90:100] = 0
            free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))

            if np.sum(free) == 0:
                return
            
            pixx = general_utils.sample_distribution(np.float32(free), self.rng)
            pix = np.array([pixx[1], pixx[0]])

            if i == 0:
                pix = np.array([50, 50])

            pos = get_xyz(pix, size)
            theta = self.rng.rand() * 2 * np.pi
            quat = Quaternion().from_rotation_matrix(rot_z(theta))

            p.removeBody(body_id)
            self.objects.append(self.add_single_object(obj_paths[i], pos, quat, size))

    def seed(self, seed):
        self.session_seed = seed        
        self.rng.seed(seed)
        
    def workspace2world(self, pos=None, quat=None, inv=False):
        """
        Transforms a pose in workspace coordinates to world coordinates

        Parameters
        ----------
        pos: list
            The position in workspace coordinates

        quat: Quaternion
            The quaternion in workspace coordinates

        Returns
        -------

        list: position in worldcreate_scene coordinates
        Quaternion: quaternion in world coordinates
        """

        world_pos, world_quat = None, None
        tran = Affine3.from_vec_quat(self.workspace_pos, Quaternion()).matrix()

        if inv:
            tran = Affine3.from_matrix(np.linalg.inv(tran)).matrix()

        if pos is not None:
            world_pos = np.matmul(tran, np.append(pos, 1))[:3]
        
        if quat is not None:
            world_rot = np.matmul(tran[0:3, 0:3], quat.rotation_matrix())
            world_quat = Quaternion.from_rotation_matrix(world_rot)

        return world_pos, world_quat
    
    def remove_flat_objs(self):
        non_flats = []

        for obj in self.objects:
            obj_pos, obj_quat = p.getBasePositionAndOrientation(obj.body_id)

            rot_mat = Quaternion(x=obj_quat[0], y=obj_quat[1], z=obj_quat[2], w=obj_quat[3]).rotation_matrix()
            angle_z = np.arccos(np.dot(np.array([0, 0, 1]), rot_mat[0:3, 2]))

            # check if obj is flat and skip it.
            if obj_pos[2] < 0 or np.abs(angle_z) > 0.3:
                p.removeBody(obj.body_id)
                continue

            non_flats.append(obj)

        self.objects = non_flats

    def centralize_objs(self, force_magnitude=10, duration=2000):
        """
        Move objects towards the center of the workspace by applying constant force
        """

        count = 0
        while count < duration:
            for obj in self.objects:
                pos, quat = p.getBasePositionAndOrientation(obj.body_id)

                # ignore objects that have fallen off the table
                if pos[2] < 0:
                    continue 

                error = self.workspace2world(np.array([0.0, 0.0, 0.0]))[0] - pos
                error[2] = 0.0
                force_direction = error/np.linalg.norm(error)
                p.applyExternalForce(obj.body_id, -1, force_magnitude * force_direction,
                                     np.array([pos[0], pos[1], 0.0]), p.WORLD_FRAME)
                
            p.stepSimulation()
            count += 1

        for obj in self.objects:
            pos, quat = p.getBasePositionAndOrientation(obj.body_id)
            obj.pos, obj.quat = self.workspace2world(pos=np.array(pos),
                                                     quat=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                                                     inv=True)
        
    def is_moving(self):
        '''
        Checks if the objects are still moving
        '''
        for obj in self.objects:
            pos, quaternion = p.getBasePositionAndOrientation(bodyUniqueId=obj.body_id)
            if pos[2] < 0:
                continue
          
            vel, rot_vel = p.getBaseVelocity(bodyUniqueId=obj.body_id)
            norm1 = np.linalg.norm(vel)
            norm2 = np.linalg.norm(rot_vel)
            if norm1 > 0.001 or norm2 > 0.1:
                return False
          
        return True
