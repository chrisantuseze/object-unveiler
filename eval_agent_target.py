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

def run_episode_obstacle(policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, success_count, max_steps=15):
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
    max_steps (int, optional): Maximum number of steps in the episode. Default is 15.
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
    
    initial_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][1], dir=TEST_EPISODES_DIR)
    processed_masks = copy.deepcopy(initial_masks)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_scene.png"), pred_mask)
    cv2.imwrite(os.path.join(TEST_DIR, "color0.png"), obs['color'][0])
    cv2.imwrite(os.path.join(TEST_DIR, "color1.png"), obs['color'][1])

    # get a randomly picked target mask from the segmented image
    target_mask, target_id = general_utils.get_target_mask(processed_masks, obs['color'][1], rng)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_target_mask.png"), target_mask)
    
    i = 0
    node_id = -1
    n_prev_masks, count = 0, 0
    total_clutter_score = 0.0

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
        action = policy.exploit_target_ppg(state, obs['color'][1], obstacle_mask)

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

        obs = copy.deepcopy(next_obs)

        new_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][1], dir=TEST_EPISODES_DIR)
        if len(new_masks) == n_prev_masks:
            count += 1

        if count > 2:
            logging.info("Robot is in an infinite loop")
            
            res = input("\nDo you still want to continue? (y/n) ")
            if res.lower() == "n":
                res = input("\nDo you think the grasp was successful? (y/n) ")
                if grasp_info['stable'] or res.lower() == "y":
                    logging.info("Target has been grasped!")

                    final_clutter_score = grasping.compute_singulation(initial_masks, new_masks)
                    episode_data['final_clutter_score'] = final_clutter_score
                    episode_data['total_clutter_score'] = total_clutter_score if total_clutter_score > 0 else final_clutter_score
                    episode_data['successful'] = True
                else:
                    logging.info("Target could not be grasped. And it is no longer available in the scene.")

                break

        target_id, target_mask = grasping.find_target(new_masks, target_mask)
        if target_id == -1:
            res = input("\nDo you think the target is available? (y/n) ")
            if res.lower() == "y":
                target_id = int(input("\nWhat is the index? "))
                target_mask = new_masks[target_id]
                continue

            res = input("\nDo you think the grasp was successful? (y/n) ")
            if grasp_info['stable'] or res.lower() == "y":
                logging.info("Target has been grasped!")

                final_clutter_score = grasping.compute_singulation(initial_masks, new_masks)
                episode_data['final_clutter_score'] = final_clutter_score
                episode_data['total_clutter_score'] = total_clutter_score if total_clutter_score > 0 else final_clutter_score
                episode_data['successful'] = True
            else:
                logging.info("Target could not be grasped. And it is no longer available in the scene.")

            print('------------------------------------------')
            break

        if policy.is_terminal(next_obs):
            break

        ############# Calculating scores ##########
        total_clutter_score += grasping.compute_singulation(processed_masks, new_masks)

        processed_masks = copy.deepcopy(new_masks)
        n_prev_masks = len(processed_masks)

    logging.info('--------')
    return episode_data

def run_episode_target(policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, max_steps=15):
    """
    Runs a single episode for evaluating direct target grasping with heuristics.
    Parameters:
    policy (Policy): The policy to be used for decision making.
    env (Environment): The environment in which the agent operates.
    segmenter (ObjectSegmenter): The object segmenter used for processing observations.
    rng: Random number generator for reproducibility.
    episode_seed: Seed for the episode to ensure reproducibility.
    max_steps (int, optional): Maximum number of steps in the episode. Default is 15.
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
    
    initial_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][1], dir=TEST_EPISODES_DIR)
    processed_masks = copy.deepcopy(initial_masks)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_scene.png"), pred_mask)
    cv2.imwrite(os.path.join(TEST_DIR, "color0.png"), obs['color'][0])
    cv2.imwrite(os.path.join(TEST_DIR, "color1.png"), obs['color'][1])

    # get a randomly picked target mask from the segmented image
    target_mask, target_id = general_utils.get_target_mask(processed_masks, obs['color'][1], rng)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_target_mask.png"), target_mask)
    
    i = 0
    n_prev_masks, count = 0, 0
    total_clutter_score = 0.0
    while episode_data['attempts'] < max_steps:
        cv2.imwrite(os.path.join(TEST_DIR, "scene.png"), pred_mask)
        cv2.imwrite(os.path.join(TEST_DIR, "target_mask.png"), target_mask)

        state = policy.state_representation(obs)
        action = policy.exploit_target_ppg(state, obs['color'][1], target_mask)

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

        obs = copy.deepcopy(next_obs)

        new_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][1], dir=TEST_EPISODES_DIR)
        if len(new_masks) == n_prev_masks:
            count += 1

        if count > 2:
            logging.info("Robot is in an infinite loop")
            
            res = input("\nDo you still want to continue? (y/n) ")
            if res.lower() == "n":
                res = input("\nDo you think the grasp was successful? (y/n) ")
                if grasp_info['stable'] or res.lower() == "y":
                    logging.info("Target has been grasped!")

                    final_clutter_score = grasping.compute_singulation(initial_masks, new_masks)
                    episode_data['final_clutter_score'] = final_clutter_score
                    episode_data['total_clutter_score'] = total_clutter_score if total_clutter_score > 0 else final_clutter_score
                    episode_data['successful'] = True
                else:
                    logging.info("Target could not be grasped. And it is no longer available in the scene.")

                break

        target_id, target_mask = grasping.find_target(new_masks, target_mask)
        if target_id == -1:
            res = input("\nDo you think the target is available? (y/n) ")
            if res.lower() == "y":
                target_id = int(input("\nWhat is the index? "))
                target_mask = new_masks[target_id]
                continue

            res = input("\nDo you think the grasp was successful? (y/n) ")
            if grasp_info['stable'] or res.lower() == "y":
                logging.info("Target has been grasped!")

                final_clutter_score = grasping.compute_singulation(initial_masks, new_masks)
                episode_data['final_clutter_score'] = final_clutter_score
                episode_data['total_clutter_score'] = total_clutter_score if total_clutter_score > 0 else final_clutter_score
                episode_data['successful'] = True
            else:
                logging.info("Target could not be grasped. And it is no longer available in the scene.")

            print('------------------------------------------')
            break

        if policy.is_terminal(next_obs):
            break

        ############# Calculating scores ##########
        total_clutter_score += grasping.compute_singulation(processed_masks, new_masks)

        processed_masks = copy.deepcopy(new_masks)
        n_prev_masks = len(processed_masks)

    logging.info('--------')
    return episode_data

# original
def run_episode(policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, max_steps=15):
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

    segmenter = ObjectSegmenter(args)

    rng = np.random.RandomState()
    rng.seed(args.seed)

    eval_data = []
    sr_n, sr_1, attempts, objects_removed = 0, 0, 0, 0
    avg_clutter_score, final_clutter_score = 0.0, 0.0
    planning_steps = 0

    success_count = 0

    for i in range(args.n_scenes):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        logging.info('Episode: {}, seed: {}'.format(i, episode_seed))

        episode_data = run_episode_target(policy, env, segmenter, rng, episode_seed)
        eval_data.append(episode_data)

        sr_1 += episode_data['sr-1']
        sr_n += episode_data['sr-n']
        attempts += episode_data['attempts']

        if episode_data['successful']:
            success_count += 1
            with open('target_ppg_results.txt', 'a') as file:
                    file.write(f"Success rate (success/total): {success_count}/{i+1}, final_clutter_score: {episode_data['final_clutter_score']}, total_clutter_score: {episode_data['total_clutter_score']}, planning steps: {episode_data['attempts']}\n")

            final_clutter_score += episode_data['final_clutter_score']
            avg_clutter_score += (episode_data['total_clutter_score']/episode_data['attempts'])
            planning_steps += episode_data['attempts']


        objects_removed += (episode_data['objects_removed'] + 1)/float(episode_data['objects_in_scene'])

        logging.info(f">>>>>>>>> {success_count}/{i+1} >>>>>>>>>>>>>")

        if i % 5 == 0:
            logging.info('Episode: {}, Avg. Clutter Score:{}, Final Clutter Score: {}, Planning Steps: {}'.format(i, avg_clutter_score, final_clutter_score, planning_steps))

    with open('target_ppg_results.txt', 'a') as file:
                    file.write(f"\nAvg Total Clutter Score: {avg_clutter_score/success_count}, Avg Final Clutter Score: {final_clutter_score/success_count}, Avg Planning Steps: {planning_steps/success_count}\n")
    
    logging.info(f"Success rate was -> {success_count}/{args.n_scenes} = {success_count/args.n_scenes}")