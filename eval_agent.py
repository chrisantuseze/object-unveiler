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
import utilities.general_utils as general_utils
from utils.constants import *
import policy.grasping as grasping

import utilities.logger as logging

# multi output
def run_episode(policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, success_count, max_steps=15, train=True):
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
    processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], dir=TEST_EPISODES_DIR, plot=True)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_scene.png"), pred_mask)

    # get a randomly picked target mask from the segmented image
    target_mask, target_id = general_utils.get_target_mask(processed_masks, obs, rng)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_target_mask.png"), target_mask)
    
    i = 0
    n_prev_masks = 0
    count = 0
    end_scene = False

    max_steps = 3
    while episode_data['attempts'] < max_steps:
        general_utils.save_image(color_img=obs['color'][1], name="color" + str(i), dir=TEST_EPISODES_DIR)

        cv2.imwrite(os.path.join(TEST_DIR, "target_mask.png"), target_mask)

        state = policy.state_representation(obs)
        actions = policy.exploit(state, target_mask)

        # evaluates thee actions to check if the actions are for the neighbors or the target
        # attempts = 0
        # while attempts < 2:
        #     actions = policy.exploit(state, target_mask)
        #     actions = utils.evaluate_actions(actions, target_mask)
        #     if len(actions) > 0:
        #         break
        #     state, target_mask = ndimage.rotate(state, 45, reshape=False), ndimage.rotate(target_mask, 45, reshape=False)
        #     attempts += 1

        # # switching to heuristics
        # print("Generating grasp action using heuristics...")
        # if len(actions) == 0:
        #     obstacle_id, _ = utils.get_obstacle_id(raw_masks, target_id, prev_node_id=-1)
        #     # action = grasping.compute_grasping_point_for_object1(processed_masks, obstacle_id, policy.aperture_limits, policy.rotations, rng)
        #     action = policy.guided_exploration(state, processed_masks[obstacle_id])
        #     actions = [action]

        # action = actions[0]

        action = general_utils.get_closest_neighbor(actions, target_mask)

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
            # end_scene = True
            break

        obs = copy.deepcopy(next_obs)

        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], dir=TEST_EPISODES_DIR, plot=True)
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

            # end_scene = True
            break

        cv2.imwrite(os.path.join(TEST_DIR, "target_mask.png"), target_mask)
        n_prev_masks = len(processed_masks)

        # episode_data['attempts'] += 1
        # if end_scene:
        #     break

    logging.info('--------')
    return episode_data, success_count

# single output with heuristics for evaluating direct target grasping 
def run_episode_old0(policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, success_count, max_steps=15, train=True):
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
    processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], dir=TEST_EPISODES_DIR, plot=True)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_scene.png"), pred_mask)

    # get a randomly picked target mask from the segmented image
    target_mask, target_id = general_utils.get_target_mask(processed_masks, obs, rng)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_target_mask.png"), target_mask)
    
    i = 0
    node_id = -1
    n_prev_masks = 0
    count = 0
    prev_node = -1
    grasp_status = []

    # NOTE: During the next iteration you need to search through the masks and identify the target, 
    # then use its id. Don't maintain the old target id because the scene has been resegmented
    while node_id != target_id:
        general_utils.save_image(color_img=obs['color'][1], name="color" + str(i), dir=TEST_EPISODES_DIR)

        node_id, prev_node = general_utils.get_obstacle_id(raw_masks, target_id, prev_node_id=prev_node)

        obstacle_mask = processed_masks[node_id]
        cv2.imwrite(os.path.join(TEST_DIR, "obstacle_mask.png"), obstacle_mask)

        state = policy.state_representation(obs)
        actions = policy.exploit(state, obstacle_mask)

        # for action in actions:
        env_action3d = policy.action3d(actions[0])
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

        grasp_status.append(grasp_info['stable'])

        print(actions)
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

        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], dir=TEST_EPISODES_DIR, plot=True)
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

        cv2.imwrite(os.path.join(TEST_DIR, "target_mask.png"), target_mask)
        n_prev_masks = len(processed_masks)

    logging.info('--------')
    return episode_data, success_count

# single output with heuristics
def run_episode_old1(policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, success_count, max_steps=15, train=True):
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
    processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], dir=TEST_EPISODES_DIR, plot=True)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_scene.png"), pred_mask)

    # get a randomly picked target mask from the segmented image
    target_mask, target_id = general_utils.get_target_mask(processed_masks, obs, rng)
    cv2.imwrite(os.path.join(TEST_DIR, "initial_target_mask.png"), target_mask)
    
    i = 0
    n_prev_masks = 0
    count = 0
    max_steps = 5
    while episode_data['attempts'] < max_steps:
        general_utils.save_image(color_img=obs['color'][1], name="color" + str(i), dir=TEST_EPISODES_DIR)

        # nodes, edges = grasping.build_graph(raw_masks)
        # if len(edges) > 0:
        #     optimal_nodes = grasping.get_optimal_target_path(edges, target_id)
        #     if len(optimal_nodes) > 0:
        #         target_mask = processed_masks[optimal_nodes[0]]            

        cv2.imwrite(os.path.join(TEST_DIR, "target_mask.png"), target_mask)

        state = policy.state_representation(obs)
        actions = policy.exploit_old(state, target_mask)

        # for action in actions:
        env_action3d = policy.action3d(actions)
        logging.info("env_action3d:", env_action3d)

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

        general_utils.delete_episodes_misc(TEST_EPISODES_DIR)

        # if policy.is_terminal(next_obs):  # checks if only one object is left in the scene and terminates the episode
        #     break

        if grasp_info['stable']:
            pass

        obs = copy.deepcopy(next_obs)

        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], dir=TEST_EPISODES_DIR, plot=True)
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

        # break

    logging.info('--------')
    return episode_data, success_count

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

    env = Environment()

    policy = Policy(args, params)
    policy.load(fcn_model=args.fcn_model, reg_model=args.reg_model)

    segmenter = ObjectSegmenter()

    rng = np.random.RandomState()
    rng.seed(args.seed)

    eval_data = []
    sr_n, sr_1, attempts, objects_removed = 0, 0, 0, 0

    success_count = 0

    for i in range(args.n_scenes):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        logging.info('Episode: {}, seed: {}'.format(i, episode_seed))

        episode_data, success_count = run_episode(policy, env, segmenter, rng, episode_seed, success_count=success_count, train=False)
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

# def parse_args():
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--fcn_model', default='', type=str, help='')
#     parser.add_argument('--reg_model', default='', type=str, help='')
#     parser.add_argument('--seed', default=6, type=int, help='')
#     parser.add_argument('--n_scenes', default=100, type=int, help='')
#     parser.add_argument('--object_set', default='seen', type=str, help='')
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     eval_agent(args)