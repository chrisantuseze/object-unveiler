import numpy as np
import yaml
import argparse
import copy
import os
import sys
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

from env.environment import Environment
from mask_rg.object_segmenter import ObjectSegmenter
from policy.policy import Policy
import utils.general_utils as general_utils
from utils.constants import *
import policy.grasping as grasping

import utils.logger as logging
from skimage import transform

def run_episode_obstacle(policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, success_count, max_steps=15, train=True):
    """
    Run a single episode of obstacle and target grasping with heuristics.
    This function evaluates the performance of a policy in an environment where the goal is to grasp a target object
    while potentially removing obstacles in the way. The episode continues until the target is grasped or a terminal
    state is reached.
    Parameters:
    policy (Policy): The policy to be evaluated.
    env (Environment): The environment in which the policy operates.
    segmenter (ObjectSegmenter): The object segmenter used to process observations.
    rng: Random number generator for reproducibility.
    episode_seed: Seed for the episode to ensure reproducibility.
    success_count (int): Counter for the number of successful grasps.
    max_steps (int, optional): Maximum number of steps in the episode. Default is 15.
    train (bool, optional): Flag indicating whether the episode is for training or evaluation. Default is True.
    Returns:
    tuple: A tuple containing:
        - episode_data (dict): A dictionary with episode statistics including:
            - 'sr-1': Success rate for the first attempt.
            - 'sr-n': Success rate for multiple attempts.
            - 'fails': Number of failed grasps.
            - 'attempts': Number of grasp attempts.
            - 'collisions': Number of collisions.
            - 'objects_removed': Number of objects removed.
            - 'objects_in_scene': Number of objects in the initial scene.
        - success_count (int): Updated counter for the number of successful grasps.
    """
    
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
    
    id = 1
    processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], dir=TEST_EPISODES_DIR)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_scene.png"), pred_mask)

    # get a randomly picked target mask from the segmented image
    target_mask, target_id = general_utils.get_target_mask(processed_masks, obs['color'][1], rng)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_target_mask.png"), target_mask)
    
    i = 0
    node_id = -1
    n_prev_masks = 0
    count = 0
    prev_node = -1

    # NOTE: During the next iteration you need to search through the masks and identify the target, 
    # then use its id. Don't maintain the old target id because the scene has been resegmented
    while node_id != target_id:
        objects_to_remove = grasping.find_obstacles_to_remove(target_id, processed_masks)
        print("\nobjects_to_remove:", objects_to_remove)

        node_id = objects_to_remove[0]
        obstacle_mask = processed_masks[node_id]
        cv2.imwrite(os.path.join(TEST_DIR, "scene.png"), pred_mask)
        cv2.imwrite(os.path.join(TEST_DIR, "target_mask.png"), target_mask)
        cv2.imwrite(os.path.join(TEST_DIR, "obstacle_mask.png"), obstacle_mask)

        state = policy.state_representation(obs)
        action = policy.exploit(state, obstacle_mask)

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

        if node_id == target_id and grasp_info['stable']:
            print(">>>>>>>>>>> Target retrieved! >>>>>>>>>>>>>")
            print('------------------------------------------')
            break

        elif policy.is_terminal(next_obs):
            break

        obs = copy.deepcopy(next_obs)

        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], dir=TEST_EPISODES_DIR)
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

            break

        n_prev_masks = len(processed_masks)

    logging.info('--------')
    return episode_data, success_count

def run_episode_target(policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, success_count, max_steps=15, train=True):
    """
    Runs a single episode for evaluating direct target grasping with heuristics.
    Parameters:
    policy (Policy): The policy to be used for decision making.
    env (Environment): The environment in which the agent operates.
    segmenter (ObjectSegmenter): The object segmenter used for processing observations.
    rng: Random number generator for reproducibility.
    episode_seed: Seed for the episode to ensure reproducibility.
    success_count (int): Counter for successful grasps.
    max_steps (int, optional): Maximum number of steps in the episode. Default is 15.
    train (bool, optional): Flag indicating whether the agent is in training mode. Default is True.
    Returns:
    tuple: A tuple containing episode data and updated success count.
    Episode Data Dictionary:
    - 'sr-1': Success rate for the first attempt.
    - 'sr-n': Success rate for multiple attempts.
    - 'fails': Number of failed grasp attempts.
    - 'attempts': Total number of grasp attempts.
    - 'collisions': Number of collisions encountered.
    - 'objects_removed': Number of objects successfully removed.
    - 'objects_in_scene': Number of objects present in the initial scene.
    """

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
    
    id = 1
    processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], dir=TEST_EPISODES_DIR)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_scene.png"), pred_mask)

    # get a randomly picked target mask from the segmented image
    target_mask, target_id = general_utils.get_target_mask(processed_masks, obs['color'][1], rng)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_target_mask.png"), target_mask)
    
    i = 0
    n_prev_masks = 0
    count = 0
    max_steps = 5
    while episode_data['attempts'] < max_steps:
        cv2.imwrite(os.path.join(TEST_DIR, "scene.png"), pred_mask)
        cv2.imwrite(os.path.join(TEST_DIR, "target_mask.png"), target_mask)

        state = policy.state_representation(obs)
        action = policy.exploit(state, target_mask)

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

        # if policy.is_terminal(next_obs):  # checks if only one object is left in the scene and terminates the episode
        #     break

        if grasp_info['stable']:
            pass

        obs = copy.deepcopy(next_obs)

        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], dir=TEST_EPISODES_DIR)
        if len(processed_masks) == n_prev_masks:
            count += 1

        if count >= 1:
            logging.info("Robot is in an infinite loop")
            break

        target_id, target_mask = grasping.find_target(processed_masks, target_mask)
        if target_id == -1:
            if grasp_info['stable']:
                logging.info("Target has been grasped!")
                success_count += 1
            else:
                logging.info("Target could not be grasped. And it is no longer available in the scene.")

            break

        n_prev_masks = len(processed_masks)

        i += 1

    logging.info('--------')
    return episode_data, success_count

# original
def run_episode(policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, success_count=0, max_steps=15, train=True):
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
        state = policy.state_representation(obs)
        action = policy.exploit(state)
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
    policy.load(ae_model=args.ae_model, reg_model=args.reg_model, sre_model=args.sre_model)

    segmenter = ObjectSegmenter()

    rng = np.random.RandomState()
    rng.seed(args.seed)

    eval_data = []
    sr_n, sr_1, attempts, objects_removed = 0, 0, 0, 0

    success_count = 0

    for i in range(args.n_scenes):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        logging.info('Episode: {}, seed: {}'.format(i, episode_seed))

        episode_data, success_count = run_episode_obstacle(policy, env, segmenter, rng, episode_seed, success_count=success_count, train=False)
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