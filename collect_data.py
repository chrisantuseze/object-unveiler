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
        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], plot=True)

        # get a randomly picked target mask from the segmented image
        target_mask, target_id = utils.get_target_mask(segmenter, obs, rng)

        node = -1
        while node != target_id:
            utils.save_image(color_img=obs['color'][1], name="color" + str(i))
            cv2.imwrite(os.path.join("save/misc", "target_mask.png"), target_mask)

            nodes, edges = grasping.build_graph(raw_masks)
            optimal_nodes = grasping.get_optimal_target_path(edges)
            print("optimal_nodes:", optimal_nodes)

            if len(optimal_nodes) <= 0:
                break
            node = optimal_nodes[0]

            action = grasping.compute_grasping_point_for_object(processed_masks[node], policy.aperture_limits, policy.rotations, rng)
            env_action3d = policy.action3d(action)
            print("env_action__:", env_action3d)

            state = policy.state_representation(obs)
            next_obs, grasp_info = env.step(env_action3d)

            if grasp_info['stable']:
                transition = {
                    'obs': obs, 
                    'state': state, 
                    'target_mask': target_mask, 
                    'masks': processed_masks, 
                    'action': action, 
                    'label': grasp_info['stable']
                }
                memory.store(transition)

            print(action)
            print(grasp_info)
            print('---------')

            if policy.is_terminal(next_obs):
                break

            obs = copy.deepcopy(next_obs)

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

            utils.save_image(color_img=obs['color'][1], name="color" + str(i))

            # get a randomly picked target mask from the segmented image
            target_mask = utils.get_target_mask(segmenter, obs, rng)

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

    collect_demonstrations(args)
