from matplotlib import pyplot as plt
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
from utils.constants import *
from env.env_components import ActionState, AdaptiveActionState

def collect_episodic_dataset(args, params):
    save_dir = "save/pc-ou-dataset"
    # save_dir = 'save/act-dataset'

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
        try:
            run_episode(i, policy, segmenter, env, memory, rng)
        except Exception as e:
            print(e)

def run_episode(i, policy: Policy, segmenter: ObjectSegmenter, env: Environment, memory: ReplayBuffer, rng):
    episode_seed = rng.randint(0, pow(2, 32) - 1)
    env.seed(episode_seed)
    obs = env.reset()
    print('Episode: {}, seed: {}'.format(i, episode_seed))

    while not policy.is_state_init_valid(obs):
        obs = env.reset()

    processed_masks, pred_mask, raw_masks, bboxes = segmenter.from_maskrcnn(obs['color'][1], dir=TRAIN_EPISODES_DIR, bbox=True)
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
        objects_to_remove = grasping.find_obstacles_to_remove(target_id, processed_masks)

        if len(objects_to_remove) < 4 and target_id in objects_to_remove:
            objects_to_remove.remove(target_id)
            objects_to_remove = [target_id] + objects_to_remove
        
        print("\nobjects_to_remove:", objects_to_remove)

        node_id = objects_to_remove[0]
        # cv2.imwrite(os.path.join(TRAIN_DIR, "target_obstacle.png"), processed_masks[node_id])
        # cv2.imwrite(os.path.join(TRAIN_DIR, "target_mask.png"), target_mask)
        # cv2.imwrite(os.path.join(TRAIN_DIR, "scene.png"), pred_mask)
        print("target id:", target_id)

        state, depth_heightmap = policy.get_state_representation(obs)
        try:
            # Select action
            action = policy.guided_exploration_old(depth_heightmap, processed_masks[node_id])
        except Exception as e:
            obs = env.reset()
            print("Resetting environment:", e)
            break

        print(action)

        env_action3d = policy.action3d(action)
        next_obs, grasp_info = env.step(env_action3d)

        grasp_status.append(grasp_info['stable'])

        # if not grasp_info['stable']:
        #     print("A failure has been recorded. Episode cancelled.")
        #     break

        print(grasp_info)
        print('---------')

        if grasp_info['stable']:
            if len(processed_masks) == 0 or target_id == -1:
                print(">>>>>>>>>>> No objects masks or target id is negative >>>>>>>>>>>>>")
                print('------------------------------------------')
                break

            resized_new_masks, extracted_object_masks, resized_bboxes = [], [], []
            for idx, mask in enumerate(processed_masks):
                resized_new_masks.append(general_utils.resize_mask(mask))
                extracted_object_masks.append(general_utils.extract_target_crop(mask, state))
                resized_bboxes.append(general_utils.resize_bbox(bboxes[idx]))
                
            resized_target_mask = general_utils.resize_mask(target_mask)
            extracted_target = general_utils.extract_target_crop(resized_target_mask, state)

            # grasped_object_id = grasping.get_grasped_object(processed_masks, action) # we want to find the exact object that was grasped - Not working

            resized_obstacle_mask = general_utils.resize_mask(processed_masks[node_id])
            extracted_obstacle = general_utils.extract_target_crop(resized_obstacle_mask, state)
            transition = {
                'color_obs': obs['color'][1],
                'next_color_obs': next_obs['color'][1],
                'depth_obs': obs['depth'][1], 
                'state': state, 
                'depth_heightmap': depth_heightmap,
                'target_mask': resized_target_mask, 
                'c_target_mask': extracted_target, 
                'obstacle_mask': resized_obstacle_mask,
                'c_obstacle_mask': extracted_obstacle, 
                'scene_mask': general_utils.resize_mask(pred_mask),
                'object_masks': resized_new_masks,
                'c_object_masks': extracted_object_masks,
                'action': action, 
                'label': grasp_info['stable'],
                'bboxes': resized_bboxes,
                'target_id': target_id,
                'objects_to_remove': objects_to_remove
            }

            episode_data_list.append(transition)

            if grasping.is_target(processed_masks[target_id], processed_masks[node_id]):
                is_target_grasped = True
                print(">>>>>>>>>>> Target retrieved! >>>>>>>>>>>>>")
                print('------------------------------------------')
                break

        obs = copy.deepcopy(next_obs) # this needs to come after the grasping check. what we save is the obs before the grasping

        # general_utils.delete_episodes_misc(TRAIN_EPISODES_DIR)

        processed_masks, pred_mask, raw_masks, bboxes = segmenter.from_maskrcnn(obs['color'][1], dir=TRAIN_EPISODES_DIR, bbox=True)
        target_id, target_mask = grasping.find_target(processed_masks, target_mask)

        if target_id == -1:
            print("Target is no longer available in the scene.")
            break

        steps += 1

    # Note: Even though we are collecting the data for the entire episodes (2 or 3 grasps), we only use the first grasp for training

    if grasping.episode_status(grasp_status, is_target_grasped):
        memory.store_episode(episode_data_list)
        print("Episode was successful. So data saved to memory!")
        print("The scene_nr_objs:", env.scene_nr_objs, ", Session seed:", env.session_seed)

        with open('episode_info.txt', 'a') as file:
            file.write(f"The scene_nr_objs: {env.scene_nr_objs}, Session seed: {env.session_seed}, Target id: {target_id}\n")

    # We do not need to waste the successful grasp
    elif len(episode_data_list) > 0:
        memory.store_episode(episode_data_list)
        print("Saved the only successful grasp")
        print("The scene_nr_objs:", env.scene_nr_objs, ", Session seed:", env.session_seed, ", Target id:", target_id)

        with open('episode_info.txt', 'a') as file:
            file.write(f"The scene_nr_objs: {env.scene_nr_objs}, Session seed: {env.session_seed}, Target id: {target_id}\n")
    else:
        print("Episode was not successful.")

def run_episode_act(i, policy: Policy, segmenter: ObjectSegmenter, env: Environment, memory: ReplayBuffer, rng):
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
    max_steps = 4

    # NOTE: During the next iteration you need to search through the masks and identify the target, 
    # then use its id. Don't maintain the old target id because the scene has been resegmented
    while steps < max_steps:
        objects_to_remove = grasping.find_obstacles_to_remove(target_id, processed_masks)
        print("\nobjects_to_remove:", objects_to_remove)

        obstacle_id = objects_to_remove[0]
        object_mask = processed_masks[obstacle_id] if len(processed_masks) > obstacle_id else target_mask
        cv2.imwrite(os.path.join(TRAIN_DIR, "obstacle_mask.png"), object_mask)
        cv2.imwrite(os.path.join(TRAIN_DIR, "target_mask.png"), target_mask)
        cv2.imwrite(os.path.join(TRAIN_DIR, "scene.png"), pred_mask)
        print("target id:", target_id)

        state, depth_heightmap = policy.get_state_representation(obs)
        try:
            actions = policy.generate_trajectory(state, object_mask, num_steps=AdaptiveActionState.EXPECTED_STEPS + 1)
        except Exception as e:
            print("Error occurred - ", e)
            break

        print("AdaptiveActionState.EXPECTED_STEPS", AdaptiveActionState.EXPECTED_STEPS, len(actions))

        end_of_episode = False
        t = 0
        traj_data = []
        while not end_of_episode:
            try:
                action = actions[t]
            except Exception as e:
                t = 0
                print("Resetting environment:", e)
                obs = env.reset()
                break

            if t % 1 == 0:
                print(action, t, env.current_state)

            env_action3d = policy.action3d(action)
            obs, grasp_info = env.step_act(env_action3d, eval=False)

            traj_data.extend(obs['traj_data'])

            t += 1
            end_of_episode = grasp_info['eoe']
        
        print("len(traj_data)", len(traj_data))

        # if len(traj_data) < AdaptiveActionState.EXPECTED_STEPS:
        #     steps += 1
        #     continue

        print(grasp_info)
        print('---------')

        # general_utils.delete_episodes_misc(TRAIN_EPISODES_DIR)

        # old_objects_count = len(processed_masks)
        # processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], dir=TRAIN_EPISODES_DIR)
        # obstacle_id, object_mask = grasping.find_target(processed_masks, object_mask)
        # new_objects_count = len(processed_masks)

        # save = int(input("Do you want to save this episode? (0/1): "))
        # save = 0

        # if len(traj_data) >= AdaptiveActionState.EXPECTED_STEPS and (obstacle_id != -1 or old_objects_count == new_objects_count):
        #     print("Scene rearranged. Episode cancelled.")
        #     break
        
        # if grasp_info['stable'] or save == 1:

        # if old_objects_count != new_objects_count and obstacle_id == -1:

        if grasp_info['stable']:
            if len(processed_masks) == 0 or target_id == -1:
                print(">>>>>>>>>>> No objects masks or target id is negative >>>>>>>>>>>>>")
                print('------------------------------------------')
                break
        
            transition = {
                'color_obs': obs['color'][1], 
                'depth_obs': obs['depth'][1], 
                'state': state, 
                'depth_heightmap': depth_heightmap,
                'target_mask': target_mask, 
                'c_target_mask': general_utils.extract_target_crop2(target_mask, obs['color'][1]), 
                'obstacle_mask': object_mask,
                'c_obstacle_mask': general_utils.extract_target_crop2(object_mask, obs['color'][1]),
                'scene_mask': pred_mask,
                'object_masks': processed_masks,
                'action': action, 
                'label': grasp_info['stable'],
                'traj_data': traj_data,
                'actions': actions, 
            }
            episode_data_list.append(transition)

            print(">>>>>>>>>>> Object grasped! >>>>>>>>>>>>>")
            print('------------------------------------------')
            break

        # if obstacle_id == -1:
        #     print("Object is no longer available in the scene.")
        #     break

        processed_masks, pred_mask, raw_masks, bboxes = segmenter.from_maskrcnn(obs['color'][1], dir=TRAIN_EPISODES_DIR, bbox=True)
        target_id, target_mask = grasping.find_target(processed_masks, target_mask)

        if target_id == -1:
            print("Target is no longer available in the scene.")
            break

        steps += 1

    if len(episode_data_list) == 1:
        memory.store_episode(episode_data_list)
        print("Saved the only successful grasp")
        
        with open('act_episode_info.txt', 'a') as file:
            file.write(f"The scene_nr_objs: {env.scene_nr_objs}, Session seed: {env.session_seed}, Target id: {target_id}, Obstacle id: {obstacle_id}\n")
    else:
        print("Episode was not successful.")

def collect_random_target_dataset(args, params):
    save_dir = 'save/ppg-ou-dataset'

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
            id = 1
            processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], dir=TRAIN_EPISODES_DIR)

            # get a randomly picked target mask from the segmented image
            target_mask, target_id = general_utils.get_target_mask(processed_masks, obs, rng)

            cv2.imwrite(os.path.join("save/misc", "target_mask.png"), target_mask)

            state = policy.state_representation(obs)
            action = policy.guided_exploration_old(state, target_mask)
            env_action3d = policy.action3d(action)
            print("env_action:", env_action3d)

            next_obs, grasp_info = env.step(env_action3d)

            if grasp_info['stable']:
                resized_target = general_utils.resize_mask(target_mask)
                extracted_target = general_utils.extract_target_crop(resized_target, state)

                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(state)
                ax[1].imshow(resized_target)
                ax[2].imshow(extracted_target)
                plt.show()

                transition = {'obs': obs, 'state': state, 'target_mask': resized_target, 'extracted_target': extracted_target, 'action': action, 'label': grasp_info['stable']}
                memory.store(transition)
                break

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

    parser.add_argument('--chunk_size', default="3", action='store', type=int, help='chunk_size', required=False)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    general_utils.create_dirs()

    with open('yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"You are using {args.device}")

    # Writing to a file
    with open('episode_info.txt', 'w') as file:
        file.write("\n")

    collect_episodic_dataset(args, params)
    # collect_random_target_dataset(args, params)
