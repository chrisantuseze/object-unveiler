import numpy as np
import yaml
import argparse
import copy
import os
import sys
import cv2

from env.environment import Environment
from policy.object_segmenter import ObjectSegmenter
from policy.policy import Policy
import utils.utils as utils

def run_episode(policy: Policy, env: Environment, segmenter: ObjectSegmenter, rng, episode_seed, max_steps=15, train=True):
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
        utils.save_image(color_img=obs['color'][1], name="color" + str(i))

        # get a randomly picked target mask from the segmented image
        target_mask = utils.get_target_mask(segmenter, obs, rng)
        cv2.imwrite(os.path.join("save/misc", "target_mask.png"), target_mask)

        # add the target seegmentation mask to the observation dictionary
        obs = utils.add_to_obs(obs, target_mask)

        state = policy.state_representation(obs)

        if train:
            action = policy.explore(state)
        else:
            action = policy.predict(state)

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

    print('--------')
    return episode_data

def eval_agent(args):
    with open('yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    env = Environment()

    policy = Policy(params)
    policy.load(fcn_model=args.fcn_model, reg_model=args.reg_model)

    segmenter = ObjectSegmenter()

    rng = np.random.RandomState()
    rng.seed(args.seed)

    eval_data = []
    sr_n, sr_1, attempts, objects_removed = 0, 0, 0, 0

    for i in range(args.n_scenes):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        print('Episode: {}, seed: {}'.format(i, episode_seed))

        episode_data = run_episode(policy, env, segmenter, rng, episode_seed, train=False)
        eval_data.append(episode_data)

        sr_1 += episode_data['sr-1']
        sr_n += episode_data['sr-n']
        attempts += episode_data['attempts']
        objects_removed += (episode_data['objects_removed'] + 1)/float(episode_data['objects_in_scene'])

        if i % 5 == 0:
            print('Episode: {}, SR-1:{}, SR-N: {}, Scene Clearance: {}'.format(i, sr_1 / (i+1),
                                                                               sr_n / attempts,
                                                                               objects_removed / len(eval_data)))

    print('SR-1:{}, SR-N: {}, Scene Clearance: {}'.format(sr_1 / args.n_scenes,
                                                          sr_n / attempts,
                                                          objects_removed / len(eval_data)))

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