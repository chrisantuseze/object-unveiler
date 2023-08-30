import yaml
from env.environment import Environment
import numpy as np
import copy
import argparse
import os
import cv2
import torch

from policy.object_segmenter import ObjectSegmenter
from policy.policy import Policy

from trainer.memory import ReplayBuffer
import utils.utils as utils
import policy.grasping as grasping
from utils.constants import *

def collect_demos(args):
    save_dir = 'save/ppg-dataset'

    # create buffer to store the data
    memory = ReplayBuffer(save_dir)

    # create the environment for the agent
    env = Environment()
    env.singulation_condition = args.singulation_condition

    with open('yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    policy = Policy(params)
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
        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], dir=TRAIN_EPISODES_DIR, plot=True)
        cv2.imwrite(os.path.join(TRAIN_DIR, "initial_scene.png"), pred_mask)

        # get a randomly picked target mask from the segmented image
        target_mask, target_id = utils.get_target_mask(processed_masks, obs, rng)
        print("target_id", target_id)

        cv2.imwrite(os.path.join(TRAIN_DIR, "initial_target_mask.png"), target_mask)

        node_id = -1
        prev_node = 0
        grasp_status = []
        is_target_grasped = False
        episode_data_list = []

        # NOTE: During the next iteration you need to search through the masks and identify the target, 
        # then use its id. Don't maintain the old target id because the scene has been resegmented
        while node_id != target_id:
            utils.save_image(color_img=obs['color'][1], name="color" + str(i), dir=TRAIN_EPISODES_DIR)

            nodes, edges = grasping.build_graph(raw_masks)
            # print("nodes:", nodes)
            # print("edges:", edges)

            if len(edges) > 0:
                optimal_nodes = grasping.get_optimal_target_path(edges, target_id)
                print("optimal_nodes:", optimal_nodes)

                if len(optimal_nodes) > 0:
                    node_id = optimal_nodes[0]

                    if prev_node == node_id and len(optimal_nodes) > 1:
                        node_id = optimal_nodes[1]
                        
                else: # if target is not occluded
                    node_id = target_id

            else: # if target is not occluded
                print("Object is not occluded")
                node_id = target_id
            print("node_id:", node_id)

            prev_node = node_id

            action = grasping.compute_grasping_point_for_object1(processed_masks, node_id, policy.aperture_limits, policy.rotations, rng)
            env_action3d = policy.action3d(action)
            # print("env_action__:", env_action3d)

            state = policy.state_representation(obs)
            next_obs, grasp_info = env.step(env_action3d)

            grasp_status.append(grasp_info['stable'])

            if grasp_info['stable']:
                transition = {
                    'obs': obs, 
                    'state': state, 
                    'target_mask': processed_masks[node_id], 
                    'masks': processed_masks, 
                    'action': action, 
                    'label': grasp_info['stable']
                }
                episode_data_list.append(transition)

            print(action)
            print(grasp_info)
            print('---------')

            utils.delete_episodes_misc(TRAIN_EPISODES_DIR)

            if node_id == target_id and grasp_info['stable']:
                is_target_grasped = True
                print(">>>>>>>>>>> Target retrieved! >>>>>>>>>>>>>")
                print('------------------------------------------')
                break

            elif policy.is_terminal(next_obs):
                break

            obs = copy.deepcopy(next_obs)

            processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], dir=TRAIN_EPISODES_DIR, plot=True)
            target_id, target_mask = grasping.get_new_target(processed_masks, target_mask)
            if target_id == -1:
                print("Target is no longer available in the scene.")
                break

            cv2.imwrite(os.path.join(TRAIN_EPISODES_DIR, "target_mask.png"), target_mask)

        if grasping.episode_status(grasp_status, is_target_grasped):
            for episode_data in episode_data_list:
                memory.store(episode_data)
            print("Episode was successful. So data saved to memory!")
        else:
            print("Episode was not successful.")


def collect_demonstrations(args):
    save_dir = 'save/ppg-dataset'

    # create buffer to store the data
    memory = ReplayBuffer(save_dir)

    # create the environment for the agent
    env = Environment()
    env.singulation_condition = args.singulation_condition

    with open('yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    policy = Policy(params)
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

        for i in range(15): # we need to have a planner at this point. We stop only when the target has been grasped.

            utils.save_image(color_img=obs['color'][1], name="color" + str(i), dir=TRAIN_EPISODES_DIR)

            # get a randomly picked target mask from the segmented image
            target_mask = obs['color'][0] #utils.get_target_mask(segmenter, obs, rng)

            cv2.imwrite(os.path.join("save/misc", "target_mask.png"), target_mask)

            state = policy.state_representation(obs)
            action = policy.guided_exploration(state)
            env_action3d = policy.action3d(action)
            print("env_action:", env_action3d)

            next_obs, grasp_info = env.step(env_action3d)

            if grasp_info['stable']:
                transition = {'obs': obs, 'state': state, 'target_mask': target_mask, 'action': action, 'label': grasp_info['stable']}
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    utils.create_dirs()

    # collect_demonstrations(args)
    collect_demos(args)
