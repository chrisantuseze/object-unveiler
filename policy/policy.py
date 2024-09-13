import os
import pickle
from policy.models_attn2 import Regressor, ResFCN
# from policy.models_multi_task import Regressor, ResFCN
# from policy.models_obstacle import Regressor, ResFCN
# from policy.models_obstacle_attn import Regressor, ResFCN
# from policy.models_obstacle_heuristics import Regressor, ResFCN
# from policy.models_obstacle_vit import Regressor, ResFCN
# from policy.models_target import Regressor, ResFCN
from mask_rg.object_segmenter import ObjectSegmenter
import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform

from einops import rearrange

import pybullet as p
from trainer.memory import ReplayBuffer

import utils.general_utils as general_utils
import utils.orientation as ori
from utils.constants import *
import env.cameras as cameras
import utils.logger as logging
import policy.grasping as grasping
import policy.grasping2 as grasping2

from act.constants import SIM_TASK_CONFIGS

class Policy:
    def __init__(self, args, params) -> None:
        self.args = args
        self.params = params
        self.rng = np.random.RandomState()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.rotations = params['agent']['fcn']['rotations']
        self.aperture_limits = params['agent']['regressor']['aperture_limits']
        self.pxl_size = params['env']['pixel_size']
        self.bounds = np.array(params['env']['workspace']['bounds'])

        self.crop_size = 32
        self.push_distance = 0.12 #0.15 # distance of the floating hand from the object to be grasped
        self.z = 0.08 # distance of the floating hand from the table (vertical distance)

        self.fcn = ResFCN(args).to(self.device)
        self.fcn_optimizer = optim.Adam(self.fcn.parameters(), lr=params['agent']['fcn']['learning_rate'])
        self.fcn_criterion = nn.BCELoss(reduction='None')

        self.segmenter = ObjectSegmenter()

        self.reg = Regressor().to(self.device)
        self.reg_optimizer = optim.Adam(self.reg.parameters(), lr=params['agent']['regressor']['learning_rate'])
        self.reg_criterion = nn.L1Loss()

        self.policy, self.stats = self.make_act_policy()
        # self.policy, self.stats = None, None

        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

        demo_save_dir = 'save/ppg-dataset'
        self.replay_buffer = ReplayBuffer(demo_save_dir)

    def seed(self, seed):
        seed = 4018109721
        self.rng.seed(seed)

    def make_act_policy(self):
        from act.policy import ACTPolicy

        lr_backbone = 1e-5
        backbone = 'resnet18'
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        lr = 1e-5
        chunk_size = self.args.chunk_size
        kl_weight = 10
        hidden_dim = 512
        dim_feedforward = 3200
        task_config = SIM_TASK_CONFIGS['sim_object_unveiler']

        ckpt_dir = "act/ckpt"
        ckpt_name = f'policy_epoch_650_seed_0.ckpt'
        # ckpt_name = f'policy_best.ckpt'
        state_dim = 1

        self.camera_names = task_config['camera_names']

        policy_config = {
            'lr': lr,
            'num_queries': chunk_size, #@Chris: ensure the chunk size is 3 and not 100
            'kl_weight': kl_weight,
            'hidden_dim': hidden_dim,
            'dim_feedforward': dim_feedforward,
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'nheads': nheads,
            'camera_names': self.camera_names,
        }

        config = {
            'num_epochs': 2000,
            'ckpt_dir': ckpt_dir,
            'episode_len': 50,
            'state_dim': state_dim,
            'lr': lr,
            'policy_class': "ACT",
            'onscreen_render': True,
            'policy_config': policy_config,
            'task_name': 'sim_transfer_cube_scripted',
            'seed': 0,
            'temporal_agg': True,
            'camera_names': self.camera_names,
            'real_robot': False

            # for unveiler
            ,'split_ratio': 0.8
        }

        
        policy = ACTPolicy(policy_config)

        # load policy and stats
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        loading_status = policy.load_state_dict(torch.load(ckpt_path))

        policy.to(self.device)
        policy.eval()
        print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        return policy, stats

    def is_state_init_valid(self, obs):
        """
        Checks if the state/scene initialization is a valid
        """
        flat_objs = 0
        for obj in obs['full_state']:
            obj_pos, obj_quat = p.getBasePositionAndOrientation(obj.body_id)
            rot_mat = ori.Quaternion(x=obj_quat[0], y=obj_quat[1],
                                     z=obj_quat[2], w=obj_quat[3]).rotation_matrix()
            
            angle_z = np.arccos(np.dot(np.array([0, 0, 1]), rot_mat[0:3, 2]))

            if np.abs(angle_z) > 0.1:
                flat_objs += 1

        return (flat_objs != len(obs['full_state']))

    def state_representation(self, obs):
        state = general_utils.get_fused_heightmap(obs, cameras.RealSense.CONFIG, self.bounds, self.pxl_size)
        return state
    
    def get_state_representation(self, obs):
        state = general_utils.get_fused_heightmap(obs, cameras.RealSense.CONFIG, self.bounds, self.pxl_size)
        color_heightmap, depth_heightmap = general_utils.get_heightmap_(obs, cameras.RealSense.CONFIG, self.bounds, self.pxl_size)

        # print(color_heightmap.shape, depth_heightmap.shape)
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(state)
        # ax[1].imshow(color_heightmap)
        # ax[2].imshow(depth_heightmap)
        # plt.show()

        return state, depth_heightmap
    
    def random_sample(self, state):
        action = np.zeros((4,))
        action[0] = self.rng.randint(0, state.shape[0])
        action[1] = self.rng.randint(0, state.shape[1])
        action[2] = self.rng.randint(0, 16) * 2 * np.pi / self.rotations
        action[3] = self.rng.uniform(self.aperture_limits[0], self.aperture_limits[1])
        return action

    def guided_exploration(self, state, sample_limits=[0.1, 0.15]):        
        obj_ids = np.argwhere(state > self.z)

        # Sample initial position.
        valid_pxl_map = np.zeros(state.shape)
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                dists = np.linalg.norm(np.array([y, x]) - obj_ids, axis=1)
                if sample_limits[0] / self.pxl_size < np.min(dists) < sample_limits[1] / self.pxl_size:
                    valid_pxl_map[y, x] = 255

        valid_pxls = np.argwhere(valid_pxl_map == 1)
        valid_ids = np.arange(0, valid_pxls.shape[0])

        if len(valid_ids) < 1:
            return np.zeros((4,))

        objects_mask = np.zeros(state.shape)
        objects_mask[state > self.z] = 255

        _, thresh = cv2.threshold(objects_mask.astype(np.uint8), 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        count = 0

        while True:
            pxl = valid_pxls[self.rng.choice(valid_ids, 1)[0]]
            p1 = np.array([pxl[1], pxl[0]])

            # Sample pushing direction. Push directions point always towards the objects.
            # Keep only contour points that are around the sample pixel position.
            pushing_area = self.push_distance / self.pxl_size
            points = []
            for cnt in contours:
                for pnt in cnt:
                    if (p1[0] - pushing_area < pnt[0, 0] < p1[0] + pushing_area) and \
                       (p1[1] - pushing_area < pnt[0, 1] < p1[1] + pushing_area):
                        points.append(pnt[0])
            if len(points) > 0:
                break

            count += 1

            if count > 20:
                return np.zeros((4,))

        ids = np.arange(len(points))
        random_id = self.rng.choice(ids, 1)[0]
        p2 = points[random_id]
        push_dir = p2 - p1
        theta = -np.arctan2(push_dir[1], push_dir[0])
        step_angle = 2 * np.pi / self.rotations
        discrete_theta = round(theta / step_angle) * step_angle

        # Sample aperture uniformly
        aperture = self.rng.uniform(self.aperture_limits[0], self.aperture_limits[1])

        action = np.zeros((4,))
        action[0] = p1[0]
        action[1] = p1[1]
        action[2] = discrete_theta
        action[3] = aperture

        return action

    def guided_exploration_old(self, state, target_mask, sample_limits=[0.1, 0.15]):
        resized_target = general_utils.resize_mask(transform, target_mask)

        full_crop = general_utils.extract_target_crop(resized_target, state)
        if np.all(full_crop == 0):
            state = resized_target
        else:
            state = full_crop

        obj_ids = np.argwhere(state > self.z)

        # sample initial position
        valid_pxl_map = np.zeros(state.shape)
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                dists = np.linalg.norm(np.array([y, x]) - obj_ids, axis=1) # gets the distances of the pixels (objs) to the vertical pos of the hand
                if len(dists) < 1:
                    continue

                if sample_limits[0]/self.pxl_size < np.min(dists) < sample_limits[1]/self.pxl_size: # pixel/obj with shortest distance gets picked
                    valid_pxl_map[y, x] = 255

        valid_pxl_map = np.array(valid_pxl_map, dtype=np.uint8)
        valid_pxls = np.argwhere(valid_pxl_map == 255) # gets indices of the pixels with values equal to 255
        valid_ids = np.arange(0, valid_pxls.shape[0]) # creates an array containing values from 0 - valid_pxls.shape[0]

        objects_mask = np.zeros(state.shape)
        objects_mask[state > self.z] = 255
        _, thresh = cv2.threshold(objects_mask.astype(np.uint8), 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        while True:
            pxl = valid_pxls[self.rng.choice(valid_ids, 1)[0]]
            p1 = np.array([pxl[1], pxl[0]]) # p1 is the initial distance of the palm from the object

            # Sample pushing direction. Push directions point always towards the objects.
            # Keep only contour points that are around the sample pixel position.
            pushing_area = self.push_distance / self.pxl_size
            points = []
            for cnt in contours:
                for pnt in cnt:
                    if (p1[0] - pushing_area < pnt[0, 0] < p1[0] + pushing_area) and \
                       (p1[1] - pushing_area < pnt[0, 1] < p1[1] + pushing_area):
                        points.append(pnt[0])

            if len(points) > 0:
                break

        ids = np.arange(len(points))
        random_id = self.rng.choice(ids, 1)[0] # creates an array by randomizing items in the ids array and picking one item from it.
        p2 = points[random_id]
        push_dir = p2 - p1
        theta = -np.arctan2(push_dir[1], push_dir[0])
        step_angle = 2 * np.pi / self.rotations
        discrete_theta = round(theta / step_angle) * step_angle

        # sample aperture uniformly
        aperture = self.rng.uniform(self.aperture_limits[0], self.aperture_limits[1])

        action = np.zeros((4,))
        action[0] = p1[0] * 1.05
        action[1] = p1[1] * 1.05
        action[2] = discrete_theta
        action[3] = aperture

        # logging.info("action:", action)

        return action
    
    def generate_trajectory(self, state, target_mask, sample_limits=[0.1, 0.15], num_steps=400):
        resized_target = general_utils.resize_mask(transform, target_mask)
        full_crop = general_utils.extract_target_crop(resized_target, state)
        if np.all(full_crop == 0):
            state = resized_target
        else:
            state = full_crop

        obj_ids = np.argwhere(state > self.z)
        
        # Generate initial position and target position
        initial_position, pushing_direction = self._sample_initial_position(state, obj_ids, sample_limits)
        target_position = initial_position + pushing_direction
        
        # Generate trajectory
        trajectory = self._interpolate_trajectory(initial_position, target_position, num_steps)
        
        # Generate actions for each step in the trajectory
        actions = []
        for position in trajectory:
            action = self._generate_action(position, pushing_direction)
            actions.append(action)
        
        return actions

    def _sample_initial_position(self, state, obj_ids, sample_limits):
        valid_pxl_map = np.zeros(state.shape)
        for x in range(state.shape[0]):
            for y in range(state.shape[1]):
                dists = np.linalg.norm(np.array([y, x]) - obj_ids, axis=1)
                if len(dists) < 1:
                    continue
                if sample_limits[0]/self.pxl_size < np.min(dists) < sample_limits[1]/self.pxl_size:
                    valid_pxl_map[y, x] = 255
        
        valid_pxls = np.argwhere(valid_pxl_map == 255)
        valid_ids = np.arange(0, valid_pxls.shape[0])
        
        objects_mask = np.zeros(state.shape)
        objects_mask[state > self.z] = 255
        _, thresh = cv2.threshold(objects_mask.astype(np.uint8), 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        while True:
            pxl = valid_pxls[self.rng.choice(valid_ids, 1)[0]]
            initial_position = np.array([pxl[1], pxl[0]])
            
            pushing_area = self.push_distance / self.pxl_size
            points = []
            for cnt in contours:
                for pnt in cnt:
                    if (initial_position[0] - pushing_area < pnt[0, 0] < initial_position[0] + pushing_area) and \
                       (initial_position[1] - pushing_area < pnt[0, 1] < initial_position[1] + pushing_area):
                        points.append(pnt[0])
            if len(points) > 0:
                break
        
        random_point = points[self.rng.choice(len(points), 1)[0]]
        pushing_direction = random_point - initial_position
        
        return initial_position, pushing_direction

    def _interpolate_trajectory(self, initial_position, target_position, num_steps):
        return [initial_position + i/num_steps * (target_position - initial_position) for i in range(num_steps + 1)]

    def _generate_action(self, position, pushing_direction):
        theta = -np.arctan2(pushing_direction[1], pushing_direction[0])
        step_angle = 2 * np.pi / self.rotations
        discrete_theta = round(theta / step_angle) * step_angle
        aperture = self.rng.uniform(self.aperture_limits[0], self.aperture_limits[1])
        
        action = np.zeros((4,))
        action[0] = position[0] * 1.05
        action[1] = position[1] * 1.05
        action[2] = discrete_theta
        action[3] = aperture
        
        return action
    
    def exploit(self, state, target_mask):

        # find optimal position and orientation
        heightmap, self.padding_width = general_utils.preprocess_heightmap(state)
        target = general_utils.preprocess_target(target_mask)

        target = torch.FloatTensor(target).unsqueeze(0).to(self.device)
        x = torch.FloatTensor(heightmap).unsqueeze(0).to(self.device)

        out_prob = self.fcn(x, target, is_volatile=True)
        out_prob = general_utils.postprocess_multi(out_prob, self.padding_width)

        best_actions = []
        actions = []
        for i in range(out_prob.shape[0]):
            prob = out_prob[i]
            best_action = np.unravel_index(np.argmax(prob), prob.shape)
            best_actions.append(best_action)


            p1 = np.array([best_actions[i][3], best_actions[i][2]])
            theta = best_actions[i][0] * 2 * np.pi/self.rotations

            # find optimal aperture
            aperture_img = general_utils.preprocess_aperture_image(state, p1, theta, self.crop_size)
            x = torch.FloatTensor(aperture_img).unsqueeze(0).to(self.device)
            aperture = self.reg(x).detach().cpu().numpy()[0, 0]
        
            # undo normalization
            aperture = general_utils.min_max_scale(aperture, range=[0, 1], 
                                        target_range=[self.aperture_limits[0], 
                                                        self.aperture_limits[1]])

            action = np.zeros((4,))
            action[0] = p1[0]
            action[1] = p1[1]
            action[2] = theta
            action[3] = aperture

            print("action:", action)

            actions.append(action)

        print("\n")

        return actions
    
    def get_inputs(self, state, color_image, target_mask):
        processed_masks, pred_mask, raw_masks, bbox = self.segmenter.from_maskrcnn(color_image, bbox=True)
        # print("processed_masks.shape", processed_masks[0].shape)
        # print("pred_mask.shape", pred_mask.shape)

        processed_pred_mask = general_utils.preprocess_target(pred_mask)
        processed_pred_mask = torch.FloatTensor(processed_pred_mask).unsqueeze(0).to(self.device)
        # print("processed_pred_mask.shape", processed_pred_mask.shape)

        processed_target = general_utils.preprocess_target(target_mask)#, state)
        processed_target = torch.FloatTensor(processed_target).unsqueeze(0).to(self.device)
        # print("processed_target.shape", processed_target.shape)

        processed_obj_masks = []
        raw_obj_masks = []
        bboxes = []
        masks = []
        for id, mask in enumerate(processed_masks):
            processed_mask = general_utils.resize_mask(transform, mask)
            raw_obj_masks.append(processed_mask)
            masks.append(processed_mask)

            processed_mask = general_utils.preprocess_target(mask)#, state)
            processed_mask = torch.FloatTensor(processed_mask).to(self.device)
            processed_obj_masks.append(processed_mask)

            bboxes.append(general_utils.resize_bbox(bbox[id]))

        processed_obj_masks = torch.stack(processed_obj_masks).to(self.device)
        raw_obj_masks = torch.FloatTensor(np.array(raw_obj_masks)).to(self.device)

        target_id = grasping.get_target_id(general_utils.resize_mask(transform, target_mask), masks)
        objects_to_remove = grasping2.find_obstacles_to_remove(target_id, masks)
        objects_to_remove = torch.FloatTensor(objects_to_remove).to(self.device)
        print(objects_to_remove)

        bboxes = torch.FloatTensor(bboxes).to(self.device)
        if processed_obj_masks.shape[0] < self.args.num_patches:
            processed_obj_masks = processed_obj_masks.unsqueeze(0)
            padding_needed = max(0, self.args.num_patches - processed_obj_masks.size(1))
            
            processed_obj_masks = torch.nn.functional.pad(processed_obj_masks, (0,0, 0,0, 0,0, 0,padding_needed, 0,0), mode='constant', value=0)
            
            raw_obj_masks = raw_obj_masks.unsqueeze(0)
            raw_obj_masks = torch.nn.functional.pad(raw_obj_masks, (0,0, 0,0, 0,padding_needed, 0,0), mode='constant', value=0)

            objects_to_remove = objects_to_remove.unsqueeze(0)
            objects_to_remove = torch.nn.functional.pad(objects_to_remove, (0,padding_needed, 0,0), mode='constant')

            bboxes = bboxes.unsqueeze(0)
            bboxes = torch.nn.functional.pad(bboxes, (0,0, 0,padding_needed), mode='constant')
        else:
            processed_obj_masks = processed_obj_masks[:self.args.num_patches]
            processed_obj_masks = processed_obj_masks.unsqueeze(0)

            raw_obj_masks = raw_obj_masks[:self.args.num_patches]
            raw_obj_masks = raw_obj_masks.unsqueeze(0)

            objects_to_remove = objects_to_remove[:self.args.num_patches]
            objects_to_remove = objects_to_remove.unsqueeze(0)

            bboxes = bboxes[:self.args.num_patches]
            bboxes = bboxes.unsqueeze(0)

        # _, top_indices = torch.topk(objects_to_remove, k=self.args.sequence_length + 1, dim=1)
        # objects_to_remove = np.argmax(objects_to_remove)

        objects_to_remove_id = 0
        print("ground truth:", objects_to_remove[0])

        raw_pred_mask = torch.FloatTensor(pred_mask).unsqueeze(0).to(self.device)
        raw_target_mask = torch.FloatTensor(target_mask).unsqueeze(0).to(self.device)

        gt_object = processed_obj_masks[0, objects_to_remove_id].unsqueeze(0)

        return processed_pred_mask, processed_target, processed_obj_masks,\
              raw_pred_mask, raw_target_mask, raw_obj_masks, objects_to_remove, gt_object, bboxes
    
    def get_act_image(self, color_images, heightmap, target_mask, masks):
        # curr_images = []
        # for cam_name in self.camera_names:
        #     curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        #     curr_images.append(curr_image)
        # curr_image = np.stack(curr_images, axis=0)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # return curr_image

        # masks = np.array(masks)
        # N, H, W = masks.shape
        # if N < self.args.num_patches:
        #     object_masks = np.zeros((self.args.num_patches, H, W), dtype=masks.dtype)
        #     object_masks[:masks.shape[0], :, :] = masks
        # else:
        #     object_masks = masks[:self.args.num_patches]

        # print("color_images[0].shape", color_images[0].shape, "heightmap.shape", heightmap.shape, "object_masks.shape", object_masks.shape)

        color_images = []
        for i in range(2):
            color_images.append(np.random.random(size=(480, 640, 3)))

        heightmap = np.random.random(size=(480, 640, 3))
        target_mask = np.random.random(size=(480, 640, 3))

        image_dict = dict()
        for cam_name in self.camera_names:
            if cam_name == 'front':
                image_dict[cam_name] = color_images[0].astype(np.float32)
            elif cam_name == 'top':
                image_dict[cam_name] = color_images[1].astype(np.float32)
            elif cam_name == 'heightmap':
                image_dict[cam_name] = np.array(heightmap).astype(np.float32)
            else:
                # idx = int(cam_name)
                # image_dict[cam_name] = np.array(object_masks[idx]).astype(np.float32)
                image_dict[cam_name] = target_mask.astype(np.float32)

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            image = general_utils.resize_image(image_dict[cam_name])
            all_cam_images.append(image)
        all_cam_images = np.stack(all_cam_images, axis=0)

        image_data = torch.from_numpy(all_cam_images / 255.0).float().unsqueeze(0).to(self.device)
        image_data = torch.einsum('b k h w c -> b k c h w', image_data)

        return image_data

    def exploit_act(self, state, target_mask, obs):
        _, self.padding_width = general_utils.preprocess_heightmap(state) # only did this to get the padding_width

        if len(obs['traj_data']) == 0: #TODO Fix this
            print("No traj data. Getting random actions...")
            return torch.rand(1, self.args.chunk_size, 4).to(self.device)
        
        # print("Getting ACT actions...")

        # heightmap = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        trajectory_data = obs['traj_data'][0]
        qpos, qvel, img = trajectory_data

        color_images = img['color']
        # processed_masks, pred_mask, raw_masks = self.segmenter.from_maskrcnn(color_images[1])
        # masks = []
        # for id, mask in enumerate(processed_masks):
        #     mask = general_utils.resize_mask(transform, mask)
        #     masks.append(general_utils.extract_target_crop(mask, state))

        image_data = self.get_act_image(color_images, state, target_mask, masks=[])

        qpos = torch.from_numpy(np.array(qpos, dtype=np.float32)).unsqueeze(0).to(self.device)

        # print("image_data.shape", image_data.shape)
        
        actions = self.policy(image_data, qpos).detach()
        # print("actions.shape", actions.shape)

        return actions
    
    def exploit_act2(self, state, target_mask, images, qpos):
        _, self.padding_width = general_utils.preprocess_heightmap(state) # only did this to get the padding_width

        if len(qpos) == 0:
            print("No traj data. Getting random actions...")
            return torch.rand(1, self.args.chunk_size, 4).to(self.device)
    
        image_data = self.get_act_image(images, state, target_mask, masks=[])
        # print("image_data.shape", image_data.shape)

        qpos = torch.from_numpy(np.array(qpos, dtype=np.float32)).unsqueeze(0).to(self.device)
        # print("qpos.shape", qpos.shape)

        actions = self.policy(image_data, qpos).detach()
        return actions

    def post_process_action(self, state, action):
        pred_action = action.squeeze(0).cpu().numpy()
        # print("action.shape", action.shape)

        post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
        pred_action = post_process(pred_action)

        p1 = np.array([pred_action[3], pred_action[2]])
        theta = pred_action[0] * 2 * np.pi/self.rotations

        # ################################################################
        # out_prob = general_utils.postprocess_single(out_prob, self.padding_width)
        # best_action = np.unravel_index(np.argmax(out_prob), out_prob.shape)
        # p1 = np.array([best_action[3], best_action[2]])
        # theta = best_action[0] * 2 * np.pi/self.rotations
        # ################################################################

        # find optimal aperture
        aperture_img = general_utils.preprocess_aperture_image(state, p1, theta, self.padding_width)
        x = torch.FloatTensor(aperture_img).unsqueeze(0).to(self.device)
        aperture = self.reg(x).detach().cpu().numpy()[0, 0]
       
        # undo normalization
        aperture = general_utils.min_max_scale(aperture, range=[0, 1], 
                                       target_range=[self.aperture_limits[0], 
                                                     self.aperture_limits[1]])

        action = np.zeros((4,))
        action[0] = pred_action[0]
        action[1] = pred_action[1]
        action[2] = pred_action[2]
        action[3] = aperture

        return action
    
    def exploit_attn(self, state, color_image, target_mask):
        # find optimal position and orientation
        heightmap, self.padding_width = general_utils.preprocess_heightmap(state)
        x = torch.FloatTensor(heightmap).unsqueeze(0).to(self.device)

        processed_pred_mask, processed_target, processed_obj_masks,\
        raw_pred_mask, raw_target_mask, raw_processed_mask,\
              objects_to_remove, gt_object, bboxes = self.get_inputs(state, color_image, target_mask)

        object_logits, out_prob = self.fcn(x,
            # processed_target, processed_obj_masks, objects_to_remove,
            processed_target, processed_obj_masks, processed_pred_mask, 
            raw_pred_mask, raw_target_mask, raw_processed_mask, bboxes, 
            is_volatile=True
        )
        out_prob = general_utils.postprocess_single(out_prob, self.padding_width)

        best_action = np.unravel_index(np.argmax(out_prob), out_prob.shape)
        p1 = np.array([best_action[3], best_action[2]])
        theta = best_action[0] * 2 * np.pi/self.rotations

        # find optimal aperture
        aperture_img = general_utils.preprocess_aperture_image(state, p1, theta, self.padding_width)
        x = torch.FloatTensor(aperture_img).unsqueeze(0).to(self.device)
        aperture = self.reg(x).detach().cpu().numpy()[0, 0]
       
        # undo normalization
        aperture = general_utils.min_max_scale(aperture, range=[0, 1], 
                                       target_range=[self.aperture_limits[0], 
                                                     self.aperture_limits[1]])

        action = np.zeros((4,))
        action[0] = p1[0]
        action[1] = p1[1]
        action[2] = theta
        action[3] = aperture

        return action
    
    def exploit_old(self, state, target_mask):
        # find optimal position and orientation
        heightmap, self.padding_width = general_utils.preprocess_heightmap(state)
        x = torch.FloatTensor(heightmap).unsqueeze(0).to(self.device)

        target = general_utils.preprocess_target(target_mask, state)
        target = torch.FloatTensor(target).unsqueeze(0).to(self.device)

        out_prob = self.fcn(x, target, is_volatile=True)
        out_prob = general_utils.postprocess_single(out_prob, self.padding_width)

        best_action = np.unravel_index(np.argmax(out_prob), out_prob.shape)
        p1 = np.array([best_action[3], best_action[2]])
        theta = best_action[0] * 2 * np.pi/self.rotations

        # find optimal aperture
        aperture_img = general_utils.preprocess_aperture_image(state, p1, theta, self.padding_width)
        x = torch.FloatTensor(aperture_img).unsqueeze(0).to(self.device)
        aperture = self.reg(x).detach().cpu().numpy()[0, 0]
       
        # undo normalization
        aperture = general_utils.min_max_scale(aperture, range=[0, 1], 
                                       target_range=[self.aperture_limits[0], 
                                                     self.aperture_limits[1]])

        action = np.zeros((4,))
        action[0] = p1[0]
        action[1] = p1[1]
        action[2] = theta
        action[3] = aperture

        return action
    
    def explore(self, state, target_mask):
        explore_prob = max(0.8 * np.power(0.9998, self.learn_step_counter), 0.1)

        if self.rng.rand() < explore_prob:
            action = self.guided_exploration(state, target_mask)
        else:
            action = self.exploit(state, target_mask)
        logging.info('explore action:', action)
        return action
    
    def get_fcn_labels(self, state, action):
        heightmap = self.preprocess(state)
        angle = (action[2] + (2 * np.pi)) % (2 * np.pi)
        rot_id = round(angle/(2 * np.pi/16))

        action_area = np.zeros((state.shape[0], state.shape[1]))
        action_area[int(action[1]), int(action[0])] = 1.0
        label = np.zeros((1, heightmap.shape[1], heightmap.shape[2]))
        label[0, self.padding_width:heightmap.shape[1] - self.padding_width,
              self.padding_width:heightmap.shape[2] - self.padding_width] = action_area

        return heightmap, rot_id, label
    
    # This is redundant
    def learn(self, transition): 
        """
        This is not useful since we use the heightmap_dataset and aperture_dataset to present our collected demonstrations in a form that can be
        fed into a pytorch dataloader which is used in training the modules.
        """
        
        # check if grasp is stable and successful
        if not transition['label']: 
            return
        
        self.replay_buffer.store(transition) # I am not sure this is important

        # sample from replay buffer
        state, action = self.replay_buffer.sample()

        # update resFCN
        heightmap, rot_id, label = self.get_fcn_labels(state, action)
       
        x = torch.FloatTensor(heightmap).unsqueeze(0).to(self.device)
        label = torch.FloatTensor(label).unsqueeze(0).to(self.device)
        rotations = np.array([rot_id])
        q_maps = self.fcn(x, specific_rotation=rotations)

        # compute loss in the whole scene
        loss = self.fcn_criterion(q_maps, label)
        loss = torch.sum(loss)
        logging.info('fcn_loss:', loss.detach().cpu().numpy())
        self.info['fcn_loss'].append(loss.detach().cpu().numpy())

        self.fcn_optimizer.zero_grad()
        loss.backward()
        self.fcn_optimizer.step()

        # update regression network
        aperture_img = general_utils.preprocess_aperture_image(state, p1=np.array([action[0], action[1]]), crop_size=self.crop_size)
        x = torch.FloatTensor(aperture_img, theta=action[2]).unsqueeze(0).to(self.device)
        
        # normalize aperture to range 0-1
        normalized_aperture = general_utils.min_max_scale(action[3],
                                                  range=[self.aperture_limits[0], self.aperture_limits[1]],
                                                  target_range=[0, 1])
        gt_aperture = torch.FloatTensor(np.array([normalized_aperture])).unsqueeze(0).to(self.device)
        pred_aperture = self.reg(x)

        logging.info('APERTURES')
        logging.info(gt_aperture, pred_aperture)

        # compute loss
        loss = self.reg_criterion(pred_aperture, gt_aperture)
        logging.info('reg_loss:', loss.detach().cpu().numpy())
        self.info['reg_loss'].append(loss.detach().cpu().numpy())

        self.reg_optimizer.zero_grad()
        loss.backward()
        self.reg_optimizer.step()

        self.learn_step_counter += 1

    def action3d(self, action):
        """
        convert from pixels to 3d coordinates
        """

        x = -(self.pxl_size * action[0] - self.bounds[0][1])
        y = self.pxl_size * action[1] - self.bounds[1][1]
        quat = ori.Quaternion.from_rotation_matrix(np.matmul(ori.rot_y(-np.pi/2),
                                                             ori.rot_x(action[2])))
        return {'pos': np.array([x, y, self.z]),
                'quat': quat,
                'aperture': action[3],
                'push_distance': self.push_distance}
    
    def save(self, epoch):
        folder_name = os.path.join(self.params['log_dir'], 'model_' + str(epoch))
        os.mkdir(folder_name)
        torch.save(self.fcn.state_dict(), os.path.join(folder_name, 'fcn.pt'))
        torch.save(self.reg.state_dict(), os.path.join(folder_name, 'reg.pt'))

        log_data = {'params': self.params.copy()}
        pickle.dump(log_data, open(os.path.join(folder_name, 'log_data'), 'wb'))

        self.info['learn_step_counter'] = self.learn_step_counter
        pickle.dump(self.info, open(os.path.join(folder_name, 'info'), 'wb'))

    def load(self, fcn_model, reg_model):
        self.fcn.load_state_dict(torch.load(fcn_model, map_location=self.device))
        self.fcn.eval()

        self.reg.load_state_dict(torch.load(reg_model, map_location=self.device))
        self.reg.eval()

    def is_terminal(self, next_obs: ori.Quaternion):
        # check if there is only one object left in the scene TODO This won't be used for mine

        objects_above = 0
        for obj in next_obs['full_state']:
            if obj.pos[2] > 0:

                # check if there is at least one object in the scene with the axis parallel to world z
                rot_mat = obj.quat.rotation_matrix()
                angle_z = np.arccos(np.dot(np.array([0, 0, 1]), rot_mat[0:3, 2]))

                if np.abs(angle_z) < 0.5:
                    objects_above += 1
                else:
                    return True
                
        return objects_above <= 1


def plot_maps(state, out_prob):

    glob_max_prob = np.max(out_prob)
    fig, ax = plt.subplots(4, 4)
    for i in range(16):
        x = int(i / 4)
        y = i % 4

        min_prob = np.min(out_prob[i][0])
        max_prob = np.max(out_prob[i][0])

        prediction_vis = general_utils.min_max_scale(out_prob[i][0],
                                             range=(min_prob, max_prob),
                                             target_range=(0, 1))
        best_pt = np.unravel_index(prediction_vis.argmax(), prediction_vis.shape)
        maximum_prob = np.max(out_prob[i][0])
        #
        ax[x, y].imshow(state, cmap='gray')
        ax[x, y].imshow(prediction_vis, alpha=0.5)
        ax[x, y].set_title(str(i) + ', ' + str(format(maximum_prob, ".3f")))

        if glob_max_prob == max_prob:
            ax[x, y].plot(best_pt[1], best_pt[0], 'rx')
        else:
            ax[x, y].plot(best_pt[1], best_pt[0], 'ro')
        dx = 20 * np.cos((i / 16) * 2 * np.pi)
        dy = -20 * np.sin((i / 16) * 2 * np.pi)
        ax[x, y].arrow(best_pt[1], best_pt[0], dx, dy, width=2, color='g')
        plt.imshow(state, cmap='gray')
        plt.imshow(prediction_vis, alpha=0.5)
        # plt.savefig(os.path.join(self.params['log_dir'], 'map_' + str(i) + '.png'), dpi=720)

    plt.show()
    # plt.savefig(os.path.join(self.params['log_dir'], 'maps', 'map_' + str(self.learn_step_counter) + '.png'),
    #             dpi=720)
    # plt.close()