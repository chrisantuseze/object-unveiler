import numpy as np
import yaml
import argparse
import copy
import os
import sys
import cv2
from scipy import ndimage

from env.environment import Environment
from policy.object_segmenter import ObjectSegmenter
from policy.policy import Policy
import utils.general_utils as general_utils
from utils.constants import *
import policy.grasping as grasping

import utils.logger as logging
from skimage import transform

# multi output using attn
def run_episode_multi(policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, success_count, max_steps=15, train=True, grp_count=0):
    env.seed(episode_seed)
    obs = env.reset()

    while not policy.is_state_init_valid(obs):
        obs = env.reset()

    episode_data = {'sr-1': 0,
                    'sr-n': 0,
                    'fails': 0,
                    'attempts': 0,
                    'collisions': 0,
                    'objects_removed': 0,
                    'objects_in_scene': len(obs['full_state'])}
    
    processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][1], dir=TEST_EPISODES_DIR, plot=True)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_scene.png"), pred_mask)

    # get a randomly picked target mask from the segmented image
    target_mask, target_id = general_utils.get_target_mask(processed_masks, obs, rng)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_target_mask.png"), target_mask)
    
    i = 0
    n_prev_masks = 0
    count = 0

    max_steps = 3
    while episode_data['attempts'] < max_steps:
        grp_count += 1
        logging.info("Grasping count -", grp_count)

        general_utils.save_image(color_img=obs['color'][1], name="color" + str(i), dir=TEST_EPISODES_DIR)

        cv2.imwrite(os.path.join(TEST_DIR, "target_mask.png"), target_mask)
        cv2.imwrite(os.path.join(TEST_DIR, "scene.png"), pred_mask)

        state = policy.state_representation(obs)
        action = policy.exploit_attn(state, obs['color'][1], target_mask)

        env_action3d = policy.action3d(action)
        next_obs, grasp_info = env.step(env_action3d)

        episode_data['attempts'] += 1
        if grasp_info['collision']:
            episode_data['collisions'] += 1

        if grasp_info['stable'] and i ==0:
            episode_data['sr-1'] += 1

        if grasp_info['stable']:
            episode_data['sr-n'] += 1
            episode_data['objects_removed'] += 1

        else:
            episode_data['fails'] += 1

        print(action)
        print(grasp_info)
        print('---------')

        general_utils.delete_episodes_misc(TEST_EPISODES_DIR)

        if policy.is_terminal(next_obs):
            break

        obs = copy.deepcopy(next_obs)

        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][1], dir=TEST_EPISODES_DIR, plot=True)
        if len(processed_masks) == n_prev_masks:
            count += 1

        if count >= 2:
            logging.info("Robot is in an infinite loop")
            break

        target_id, target_mask = grasping.find_target(processed_masks, target_mask)
        if target_id == -1:
            if grasp_info['stable']:
                logging.info("Target has been grasped!")
                success_count += 1
            else:
                logging.info("Target could not be grasped. And it is no longer available in the scene.")

            print('------------------------------------------')
            break

        n_prev_masks = len(processed_masks)

    logging.info('--------')
    return episode_data, success_count, grp_count

# original
def run_episode_old2(policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, success_count=0, max_steps=15, train=True):
    env.seed(episode_seed)
    obs = env.reset()

    while not policy.is_state_init_valid(obs):
        obs = env.reset()

    episode_data = {'sr-1': 0,
                    'sr-n': 0,
                    'fails': 0,
                    'attempts': 0,
                    'collisions': 0,
                    'objects_removed': 0,
                    'objects_in_scene': len(obs['full_state'])}
    
    
    i = 0
    while episode_data['attempts'] < max_steps:
        general_utils.save_image(color_img=obs['color'][1], name="color" + str(i), dir=TEST_EPISODES_DIR)

        state = policy.state_representation(obs)
        action = policy.exploit_old(state)
        env_action3d = policy.action3d(action)

        next_obs, grasp_info = env.step(env_action3d)
        episode_data['attempts'] += 1

        if grasp_info['collision']:
            episode_data['collisions'] += 1

        if grasp_info['stable'] and i ==0:
            episode_data['sr-1'] += 1

        if grasp_info['stable']:
            episode_data['sr-n'] += 1
            episode_data['objects_removed'] += 1

        else:
            episode_data['fails'] += 1

        if policy.is_terminal(next_obs):
            break

        obs = copy.deepcopy(next_obs)

        i += 1

    logging.info('--------')
    return episode_data


def eval_agent(args):
    with open('yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    env = Environment(params)

    policy = Policy(args, params)
    policy.load(fcn_model=args.fcn_model, reg_model=args.reg_model)

    segmenter = ObjectSegmenter()

    rng = np.random.RandomState()
    rng.seed(args.seed)

    eval_data = []
    sr_n, sr_1, attempts, objects_removed = 0, 0, 0, 0

    success_count = 0
    grasping_action_count = 0

    for i in range(args.n_scenes):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        logging.info('Episode: {}, seed: {}'.format(i, episode_seed))

        episode_data, success_count, grasping_action_count = run_episode_multi(policy, env, segmenter, rng, episode_seed, success_count=success_count, train=False, grp_count=grasping_action_count)
        eval_data.append(episode_data)

        sr_1 += episode_data['sr-1']
        sr_n += episode_data['sr-n']
        attempts += episode_data['attempts']
        objects_removed += (episode_data['objects_removed'] + 1)/float(episode_data['objects_in_scene'])

        logging.info(f">>>>>>>>> {success_count}/{i+1} >>>>>>>>>>>>>")

        if i % 5 == 0:
            logging.info('Episode: {}, SR-1:{}, SR-N: {}, Scene Clearance: {}'.format(i, sr_1 / (i+1),
                                                                               sr_n / attempts,
                                                                               objects_removed / len(eval_data)))

    logging.info('SR-1:{}, SR-N: {}, Scene Clearance: {}'.format(sr_1 / args.n_scenes,
                                                          sr_n / attempts,
                                                          objects_removed / len(eval_data)))
    logging.info(f"Success rate was -> {success_count}/{args.n_scenes} = {success_count/args.n_scenes}")