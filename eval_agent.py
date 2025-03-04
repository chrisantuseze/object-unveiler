import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml
import argparse
import copy
import os
import sys
import cv2
from scipy import ndimage

from env.environment import Environment
from mask_rg.object_segmenter import ObjectSegmenter
from policy.policy import Policy
import utils.general_utils as general_utils
from utils.constants import *
import policy.grasping as grasping

import utils.logger as logging
from skimage import transform
from env.env_components import ActionState

def run_episode_multi(args, policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, success_count, max_steps=15, grp_count=0):
    """
    Runs a single episode of the multi-step grasping task using the object-unveiler policy in a cluttered environment.
    Args:
        args: Additional arguments for the episode.
        policy (Policy): The policy object that defines the agent's behavior.
        env (Environment): The environment in which the agent operates.
        segmenter (ObjectSegmenter): The object segmenter used to process observations.
        rng: Random number generator for reproducibility.
        episode_seed: Seed for the episode to ensure reproducibility.
        success_count (int): Counter for successful grasps.
        max_steps (int, optional): Maximum number of steps in the episode. Defaults to 15.
        grp_count (int, optional): Counter for the number of grasp attempts. Defaults to 0.
    Returns:
        tuple: A tuple containing:
            - episode_data (dict): Data collected during the episode, including success rates, failures, attempts, collisions, objects removed, clutter score, and singulation score.
            - success_count (int): Updated counter for successful grasps.
            - grp_count (int): Updated counter for the number of grasp attempts.
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
                    'objects_in_scene': len(obs['full_state']),
                    'avg_clutter_score': 0.0,
                    'final_clutter_score': 0.0,
                }
    
    initial_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][1], dir=TEST_DIR)
    processed_masks = copy.deepcopy(initial_masks)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_scene.png"), pred_mask)
    cv2.imwrite(os.path.join(TEST_DIR, "color0.png"), obs['color'][0])
    cv2.imwrite(os.path.join(TEST_DIR, "color1.png"), obs['color'][1])

    # get a randomly picked target mask from the segmented image
    target_mask, target_id = general_utils.get_target_mask(processed_masks, obs, rng)
    print("Target ID:", target_id)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_target_mask.png"), target_mask)
    
    i = 0
    n_prev_masks = count = 0
    avg_clutter_score = 0.0

    max_steps = 5
    while episode_data['attempts'] < max_steps:
        grp_count += 1
        logging.info("Grasping count -", grp_count)

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

        # general_utils.delete_episodes_misc(TEST_DIR)

        if policy.is_terminal(next_obs):
            break

        obs = copy.deepcopy(next_obs)

        new_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][1], dir=TEST_DIR)
        if len(new_masks) == n_prev_masks:
            count += 1

        if count > 1:
            logging.info("Robot is in an infinite loop")
            break

        target_id, target_mask = grasping.find_target(new_masks, target_mask)
        if target_id == -1:
            if grasp_info['stable']:
                logging.info("Target has been grasped!")
                success_count += 1

                episode_data['final_clutter_score'] = grasping.compute_singulation(initial_masks, new_masks)
                episode_data['avg_clutter_score'] = avg_clutter_score
            else:
                logging.info("Target could not be grasped. And it is no longer available in the scene.")

            print('------------------------------------------')
            break

        ############# Calculating scores ##########
        avg_clutter_score += grasping.compute_singulation(processed_masks, new_masks)

        processed_masks = copy.deepcopy(new_masks)
        n_prev_masks = len(processed_masks)

    logging.info('--------')
    return episode_data, success_count, grp_count

def run_episode_act(args, policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, success_count, max_steps=15, grp_count=0):
    """
    Runs a single episode of obstacle and target grasping using ACT with heuristics.
    Args:
        args: Additional arguments for the episode.
        policy (Policy): The policy object that defines the agent's behavior.
        env (Environment): The environment in which the agent operates.
        segmenter (ObjectSegmenter): The object segmenter used to process observations.
        rng: Random number generator for reproducibility.
        episode_seed: Seed for the episode to ensure reproducibility.
        success_count (int): Counter for successful grasps.
        max_steps (int, optional): Maximum number of steps in the episode. Defaults to 15.
        grp_count (int, optional): Counter for the number of grasp attempts. Defaults to 0.
    Returns:
        tuple: A tuple containing:
            - episode_data (dict): Data collected during the episode, including success rates, failures, attempts, collisions, objects removed, clutter score, and singulation score.
            - success_count (int): Updated counter for successful grasps.
            - grp_count (int): Updated counter for the number of grasp attempts.
    """
    query_frequency = args.chunk_size
    temporal_agg = args.temporal_agg
    # state_dim = 8
    state_dim = 4
    if temporal_agg:
        query_frequency = 1
        num_queries = args.chunk_size

    max_timesteps = ActionState.NUM_STEPS + 1

    if temporal_agg:
        all_time_actions = torch.zeros([max_timesteps + 5, max_timesteps+num_queries, state_dim]).to(args.device)
        print("All time actions shape -", all_time_actions.shape)

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
                    'objects_in_scene': len(obs['full_state']),
                    'avg_clutter_score': 0.0,
                    'final_clutter_score': 0.0,
                }
    
    initial_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][1], dir=TEST_EPISODES_DIR)
    processed_masks = copy.deepcopy(initial_masks)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_scene.png"), pred_mask)

    # get a randomly picked target mask from the segmented image
    target_mask, target_id = general_utils.get_target_mask(processed_masks, obs, rng)

    cv2.imwrite(os.path.join(TEST_DIR, "initial_target_mask.png"), target_mask)
    
    i = 0
    n_prev_masks = count = 0
    avg_clutter_score = 0.0

    max_steps = 5
    while episode_data['attempts'] < max_steps:
        grp_count += 1
        logging.info("Grasping count -", grp_count)

        objects_to_remove = grasping.find_obstacles_to_remove(target_id, processed_masks)
        print("\nobjects_to_remove:", objects_to_remove)

        obstacle_id = objects_to_remove[0]
        object_mask = processed_masks[obstacle_id]
        cv2.imwrite(os.path.join(TEST_DIR, "obstacle_mask.png"), object_mask)
        cv2.imwrite(os.path.join(TEST_DIR, "target_mask.png"), target_mask)
        cv2.imwrite(os.path.join(TEST_DIR, "scene.png"), pred_mask)

        # c_object_mask = general_utils.extract_target_crop2(object_mask, obs['color'][1])
        c_target_mask = general_utils.extract_target_crop2(target_mask, obs['color'][1])

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(obs['color'][1])
        # ax[1].imshow(c_object_mask)
        ax[1].imshow(c_target_mask)
        plt.show()

        end_of_episode = False
        t = 0
        preds = gt = []
        state = policy.state_representation(obs)
        while not end_of_episode:
            if t % query_frequency == 0:
                actions = policy.exploit_act(state, c_target_mask, obs)

            if temporal_agg:
                all_time_actions[[t], t:t+num_queries] = actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).to(args.device).unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = actions[:, t % query_frequency]

            action = policy.post_process_action(raw_action)
            preds.append(action)

            if t % 10 == 0:
                print(t, ",", env.current_state)

            env_action3d = policy.action3d(action)
            obs, grasp_info = env.step_act(env_action3d, eval=True)

            t += 1
            end_of_episode = grasp_info['eoe']

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

        print(grasp_info)
        print('---------') # 16 dp

        general_utils.delete_episodes_misc(TEST_EPISODES_DIR)

        if policy.is_terminal(obs):
            break

        new_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][1], dir=TEST_EPISODES_DIR)
        if len(new_masks) == n_prev_masks:
            count += 1

        if count > 1:
            logging.info("Robot is in an infinite loop")
            break

        target_id, target_mask = grasping.find_target(new_masks, target_mask)
        if target_id == -1:
            if grasp_info['stable']:
                logging.info("Target has been grasped!")
                success_count += 1

                episode_data['final_clutter_score'] = grasping.compute_singulation(initial_masks, new_masks)
                episode_data['avg_clutter_score'] = avg_clutter_score

            else:
                logging.info("Target could not be grasped. And it is no longer available in the scene.")

            print('------------------------------------------')
            break

        ############# Calculating scores ##########
        avg_clutter_score += grasping.compute_singulation(processed_masks, new_masks)

        processed_masks = copy.deepcopy(new_masks)
        n_prev_masks = len(processed_masks)

    logging.info('--------')
    return episode_data, success_count, grp_count

def plot_joint_positions_over_time(ground_truth, predicted, filename='joint_positions_plot.png'):
    """
    Plot ground truth and predicted joint positions over time and save to a file.
    
    :param ground_truth: Numpy array of shape (time_steps, 8) for ground truth joint positions
    :param predicted: Numpy array of shape (time_steps, 8) for predicted joint positions
    :param filename: String, the filename to save the plot (default: 'joint_positions_plot.png')
    """
    time_steps = ground_truth.shape[0]
    joint_count = ground_truth.shape[1]
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Comparison of Ground Truth and Predicted Joint Positions Over Time', fontsize=16)
    
    for joint in range(joint_count):
        ax = axes[joint // 2, joint % 2]
        
        ax.plot(range(time_steps), ground_truth[:, joint], label='Ground Truth', color='blue')
        ax.plot(range(time_steps), predicted[:, joint], label='Predicted', color='red', linestyle='--')
        
        ax.set_title(f'Joint {joint + 1}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Position')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
    
    print(f"Plot saved to {filename}")

def eval_agent(args):
    print("Running eval...")
    with open('yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    env = Environment(params)

    policy = Policy(args, params)
    # policy.load(ae_model=args.ae_model, reg_model=args.reg_model, sre_model="save/unveiler-xformer-encoder/unveiler_model_61.pt")

    segmenter = ObjectSegmenter()

    rng = np.random.RandomState()
    rng.seed(args.seed)

    eval_data = []
    sr_n, sr_1, attempts, objects_removed = 0, 0, 0, 0
    avg_clutter_score, final_clutter_score = 0.0, 0.0

    success_count = 0
    grasping_action_count = 0

    for i in range(args.n_scenes):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        logging.info('Episode: {}, seed: {}'.format(i, episode_seed))

        episode_data, success_count, grasping_action_count = run_episode_act(
            args, policy, env, segmenter, rng, episode_seed, 
            success_count=success_count, grp_count=grasping_action_count
        )
        eval_data.append(episode_data)

        sr_1 += episode_data['sr-1']
        sr_n += episode_data['sr-n']
        attempts += episode_data['attempts']
        avg_clutter_score += (episode_data['avg_clutter_score']/attempts)
        final_clutter_score += episode_data['final_clutter_score']

        objects_removed += (episode_data['objects_removed'] + 1)/float(episode_data['objects_in_scene'])

        logging.info(f">>>>>>>>> {success_count}/{i+1} >>>>>>>>>>>>>")

        if i % 5 == 0:
            # logging.info('Episode: {}, SR-1:{}, SR-N: {}, Scene Clearance: {}'.format(i, sr_1 / (i+1),
            #                                                                    sr_n / attempts,
            #                                                                    objects_removed / len(eval_data)))
            logging.info('Episode: {}, Avg. Clutter Score:{}, Final Clutter Score: {}'.format(i, avg_clutter_score, final_clutter_score))

    # logging.info('SR-1:{}, SR-N: {}, Scene Clearance: {}'.format(sr_1 / args.n_scenes,
    #                                                       sr_n / attempts,
    #                                                       objects_removed / len(eval_data)))
    logging.info('Avg Clutter Score:{}, Avg Singulation Score: {}'.format(avg_clutter_score / attempts, final_clutter_score / attempts))
    logging.info(f"Success rate was -> {success_count}/{args.n_scenes} = {success_count/args.n_scenes}")