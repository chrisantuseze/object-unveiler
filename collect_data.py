import yaml
from env.environment import Environment
import numpy as np
import copy
import argparse
import os
import cv2
import torch
from skimage import transform

from mask_rg.object_segmenter import ObjectSegmenter
from policy.policy import Policy

from trainer.memory import ReplayBuffer
import utils.general_utils as general_utils
import policy.grasping as grasping
import policy.grasping2 as grasping2
import policy.grasping3 as grasping3
from utils.constants import *

def collect_episodic_dataset(args, params):
    save_dir = 'save/ppg-dataset'

    # create buffer to store the data
    memory = ReplayBuffer(save_dir)

    # create the environment for the agent
    env = Environment(params)
    env.singulation_condition = args.singulation_condition

    policy = Policy(args, params)
    policy.seed(args.seed)

    rng = np.random.RandomState()
    rng.seed(args.seed)

    segmenter = ObjectSegmenter()

    for i in range(args.n_samples):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        env.seed(episode_seed)
        obs = env.reset()
        print('Episode: {}, seed: {}'.format(i, episode_seed))

        while not policy.is_state_init_valid(obs):
            obs = env.reset()

        id = 1
        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], dir=TRAIN_EPISODES_DIR)
        cv2.imwrite(os.path.join(TRAIN_DIR, "initial_scene.png"), pred_mask)

        # get a randomly picked target mask from the segmented image
        target_mask, target_id = general_utils.get_target_mask(processed_masks, obs, rng)
        print("target_id", target_id)

        cv2.imwrite(os.path.join(TRAIN_DIR, "initial_target_mask.png"), target_mask)

        node_id = -1
        grasp_status = []
        is_target_grasped = False
        episode_data_list = []

        steps = 0

        # NOTE: During the next iteration you need to search through the masks and identify the target, 
        # then use its id. Don't maintain the old target id because the scene has been resegmented
        while node_id != target_id:
            objects_to_remove = grasping2.find_obstacles_to_remove(target_id, processed_masks)
            print("\nobjects_to_remove:", objects_to_remove)

            # node_id, prev_node_id = grasping.get_obstacle_id(raw_masks, target_id, prev_node_id)

            node_id = objects_to_remove[0]
            cv2.imwrite(os.path.join(TRAIN_DIR, "target_obstacle.png"), processed_masks[node_id])
            cv2.imwrite(os.path.join(TRAIN_DIR, "target_mask.png"), target_mask)
            cv2.imwrite(os.path.join(TRAIN_DIR, "scene.png"), pred_mask)
            
            print("target id:", target_id)

            state, depth_heightmap = policy.get_state_representation(obs)
            # action = policy.guided_exploration_old(depth_heightmap, processed_masks[node_id])
            try:
                # Select action
                action = policy.guided_exploration_old(depth_heightmap, processed_masks[node_id])
            except Exception as e:
                obs = env.reset()
                print("Resetting environment:", e)
                continue

            print(action)

            env_action3d = policy.action3d(action)
            next_obs, grasp_info = env.step(env_action3d)

            # if this is the first loop, set the initial reset obs with the joint and images from the next_obs
            if steps == 0:
                obs['traj_data'] = next_obs['traj_data']

            grasp_status.append(grasp_info['stable'])

            # if not grasp_info['stable']:
            #     print("A failure has been recorded. Episode cancelled.")
            #     break

            print(grasp_info)
            print('---------')

            general_utils.delete_episodes_misc(TRAIN_EPISODES_DIR)

            if grasp_info['stable']:
                new_id, mask = grasping.get_grasped_object(processed_masks, action)
                
                new_masks = []
                for mask in processed_masks:
                    new_masks.append(general_utils.resize_mask(transform, mask))
                    
                transition = {
                    'color_obs': obs['color'][1], 
                    'depth_obs': obs['depth'][1], 
                    'state': state, 
                    'depth_heightmap': depth_heightmap,
                    'target_mask': general_utils.resize_mask(transform, processed_masks[target_id]), 
                    'obstacle_mask': general_utils.resize_mask(transform, mask),
                    'scene_mask': general_utils.resize_mask(transform, pred_mask),
                    'object_masks': new_masks,
                    'action': action, 
                    'label': grasp_info['stable'],
                    'traj_data': obs['traj_data'],
                }
                episode_data_list.append(transition)

                if grasping.is_target(processed_masks[target_id], mask):
                    is_target_grasped = True
                    print(">>>>>>>>>>> Target retrieved! >>>>>>>>>>>>>")
                    print('------------------------------------------')
                    break

            obs = copy.deepcopy(next_obs)

            processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], dir=TRAIN_EPISODES_DIR)
            target_id, target_mask = grasping.find_target(processed_masks, target_mask)
            if target_id == -1:
                print("Target is no longer available in the scene.")
                break

            steps += 1

        if grasping.episode_status(grasp_status, is_target_grasped):
            memory.store_episode(episode_data_list)
            print("Episode was successful. So data saved to memory!")

        # We do not need to waste the successful grasp
        elif len(episode_data_list) == 1:
            transition = episode_data_list[0]
            transition['target_mask'] = transition['obstacle_mask']

            memory.store_episode([transition])
            print("Saved the only successful grasp")
        else:
            print("Episode was not successful.")

def collect_random_target_dataset(args, params):
    save_dir = 'save/ppg-dataset'

    # create buffer to store the data
    memory = ReplayBuffer(save_dir)

    # create the environment for the agent
    env = Environment(params)
    env.singulation_condition = args.singulation_condition

    policy = Policy(args, params)
    policy.seed(args.seed)

    rng = np.random.RandomState()
    rng.seed(args.seed)

    segmenter = ObjectSegmenter()

    for i in range(args.n_samples):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        env.seed(episode_seed)
        obs = env.reset()
        print('Episode: {}, seed: {}'.format(i, episode_seed))

        while not policy.is_state_init_valid(obs):
            obs = env.reset()

        for i in range(15):
            # get a randomly picked target mask from the segmented image
            target_mask = obs['color'][0] #utils.get_target_mask(segmenter, obs, rng)

            cv2.imwrite(os.path.join("save/misc", "target_mask.png"), target_mask)

            state = policy.state_representation(obs)
            action = policy.guided_exploration(state, target_mask)
            env_action3d = policy.action3d(action)
            print("env_action:", env_action3d)

            next_obs, grasp_info = env.step(env_action3d)

            if grasp_info['stable']:
                transition = {'obs': obs, 'state': state, 'target_mask': general_utils.resize_mask(transform, target_mask), 'action': action, 'label': grasp_info['stable']}
                memory.store(transition)

            print(action)
            print(grasp_info)
            print('---------')

            if policy.is_terminal(next_obs):
                break

            obs = copy.deepcopy(next_obs)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_samples', default=10000, type=int, help='')
    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--singulation_condition', action='store_true', default=False, help='')

    # not needed for this operation, but if its not here, it causes problem in policy.py
    parser.add_argument('--patch_size', default=64, type=int, help='')
    parser.add_argument('--num_patches', default=10, type=int, help='')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    general_utils.create_dirs()

    with open('yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    collect_episodic_dataset(args, params)
