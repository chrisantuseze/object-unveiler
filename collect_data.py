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

        id = 1
        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], plot=True)

        objs = [4, 5]
        for obj in objs:
            utils.save_image(color_img=obs['color'][1], name="color" + str(i))
            
            # get a randomly picked target mask from the segmented image
            # target_mask = utils.get_target_mask(segmenter, obs, rng)

            #0, 3 - >3 is the target
            target_mask = processed_masks[obj]

            cv2.imwrite(os.path.join("save/misc", "target_mask.png"), target_mask)

            ######################
            action = policy.compute_grasping_point_for_object(target_mask)
            env_action3d = policy.action3d(action)
            print("env_action__:", env_action3d)
            ###########################

            state = policy.state_representation(obs)
            # action = policy.guided_exploration(state)
            # env_action3d = policy.action3d(action)
            # print("env_action:", env_action3d)

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

        for i in range(15): # we need to have a planner at this point. We stop only when the target has been grasped.

            # utils.save_image(color_img=obs['color'][1], name="color" + str(i))
            
            # # get a randomly picked target mask from the segmented image
            # # target_mask = utils.get_target_mask(segmenter, obs, rng)

            # id = 1
            # mask_info = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], plot=True)
            # #0, 3 - >3 is the target
            # target_mask = mask_info[4]

            # cv2.imwrite(os.path.join("save/misc", "target_mask.png"), target_mask)

            # ######################
            # action = policy.compute_grasping_point_for_object(target_mask)
            # env_action3d = policy.action3d(action)
            # print("env_action__:", env_action3d)
            # ###########################

            # state = policy.state_representation(obs)
            # # action = policy.guided_exploration(state)
            # # env_action3d = policy.action3d(action)
            # # print("env_action:", env_action3d)

            # next_obs, grasp_info = env.step(env_action3d)

            # if grasp_info['stable']:
            #     transition = {'obs': obs, 'state': state, 'action': action, 'label': grasp_info['stable']}
            #     memory.store(transition)

            # print(action)
            # print(grasp_info)
            # print('---------')

            # if policy.is_terminal(next_obs):
            #     break

            # obs = copy.deepcopy(next_obs)
            pass


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
