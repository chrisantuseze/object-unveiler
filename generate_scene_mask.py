from random import Random
import shutil
from typing import Any
import numpy as np
import pickle
import os
import cv2
import torch
from skimage import transform

from policy.object_segmenter import ObjectSegmenter
from trainer.memory import ReplayBuffer
import utils.logger as logging
import utils.general_utils as general_utils
import env.cameras as cameras
import policy.grasping as grasping
import policy.grasping2 as grasping2

dataset_dir = 'save/pc-ou-dataset'

def modify_episode(memory: ReplayBuffer, episode_dir, index):
    try:
        episode_data = pickle.load(open(os.path.join(dataset_dir, episode_dir), 'rb'))
    except Exception as e:
        logging.info(e, "- Failed episode:", episode_dir)

    episode_data_list = []
    for data in episode_data:
        obs = data['obs']
        segmenter = ObjectSegmenter()
        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][1], plot=True)

        new_masks = []
        for mask in processed_masks:
            new_masks.append(general_utils.resize_mask(transform, mask))

        _, edges = grasping.build_graph(raw_masks)
        optimal_nodes = []
        if len(edges) > 0:
            target_id = grasping.get_target_id(data['target_mask'], processed_masks)
            optimal_nodes = grasping.get_optimal_target_path(edges, target_id)
    
        transition = {
            'color_obs': obs['color'][1],
            'depth_obs': obs['depth'][1],
            'state': data['state'], 
            'target_mask': general_utils.resize_mask(transform, data['target_mask']), 
            'obstacle_mask': general_utils.resize_mask(transform, data['obstacle_mask']),
            'scene_mask': general_utils.resize_mask(transform, pred_mask),
            'object_masks': new_masks,
            'action': data['action'], 
            'label': data['label'],
            'optimal_nodes': optimal_nodes
        }
        episode_data_list.append(transition)

    memory.store_episode(episode_data_list)
    logging.info(f"{index} - Episode with dir {episode_dir} updated...")

def modify_episode2(episode_dir, index):
    try:
        episode_data = pickle.load(open(os.path.join(dataset_dir, episode_dir), 'rb'))
    except Exception as e:
        logging.info(e, "- Failed episode:", episode_dir)

    episode_data_list = []
    for data in episode_data:
        heightmap = data['state']
        object_masks = data['object_masks']

        new_masks = []
        for mask in object_masks:
            new_masks.append(general_utils.extract_target_crop(mask, heightmap))

        # get optimal nodes
        objects_to_remove = grasping2.get_target_objects_distance(data['target_mask'], data['object_masks'])
        # print("\nobjects_to_remove:", objects_to_remove)

        transition = {
            # 'color_obs': data['color_obs'],
            # 'depth_obs': data['depth_obs'],
            'state': data['state'], 
            # 'depth_heightmap': data['depth_heightmap'],
            'target_mask': data['target_mask'], 
            'c_target_mask': general_utils.extract_target_crop(data['target_mask'], heightmap), 
            'c_obstacle_mask': general_utils.extract_target_crop(data['obstacle_mask'], heightmap),
            'scene_mask': data['scene_mask'],
            'c_object_masks': new_masks,
            'object_masks': object_masks,
            'action': data['action'],
            'optimal_nodes': objects_to_remove,
            'label': data['label'],
        }
        episode_data_list.append(transition)

    memory.store_episode(episode_data_list)
    logging.info(f"{index} - Episode with dir {episode_dir} updated...")


def modify_transitions(memory: ReplayBuffer, transition_dir, idx):
    heightmap = cv2.imread(os.path.join(dataset_dir, transition_dir, 'heightmap.exr'), -1)
    target_mask = cv2.imread(os.path.join(dataset_dir, transition_dir, 'target_mask.png'), -1)
    action = pickle.load(open(os.path.join(dataset_dir, transition_dir, 'action'), 'rb'))
    full_state = action #pickle.load(open(os.path.join(dataset_dir, transition_dir, 'full_state'), 'rb'))

    color = cv2.imread(os.path.join(dataset_dir, transition_dir, 'color_0.png'), -1)
    depth = cv2.imread(os.path.join(dataset_dir, transition_dir, 'depth_0.exr'), -1)
    seg = cv2.imread(os.path.join(dataset_dir, transition_dir, 'seg_0.png'), -1)

    obs = {
        'color': [color],
        'depth': [depth],
        'seg': [seg],
        'full_state': full_state,
    }
    bounds = [[-0.25, 0.25], [-0.25, 0.25], [0.01, 0.3]]
    pxl_size = 0.005
    color_heightmap, depth_heightmap = general_utils.get_heightmap_(obs, cameras.RealSense.CONFIG, bounds, pxl_size)

    transition = {
        'state': heightmap,
        'target_mask': target_mask,
        'action': action,
        'obs': obs,
        'depth_heightmap': depth_heightmap,
        'color_heightmap': color_heightmap,
    }
    memory.store(transition)
    logging.info(f"{idx} - Episode with dir {transition_dir} updated...")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    new_dir = dataset_dir + '2'
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    memory = ReplayBuffer(new_dir)


    episode_dirs = os.listdir(dataset_dir)
    
    for file_ in episode_dirs:
        if not file_.startswith("episode"):
            print(file_)
            episode_dirs.remove(file_)

        # if not file_.startswith("transition"):
        #     episode_dirs.remove(file_)

    for i, episode_dir in enumerate(episode_dirs):
        #  modify_episode(memory, episode_dir, i)
        # modify_transitions(memory, episode_dir, i)

        modify_episode2(episode_dir, i)

    logging.info(f"Dataset modified and saved in {new_dir}")
    