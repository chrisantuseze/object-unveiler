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
import policy.grasping as grasping

# dataset_dir = 'save/ou-dataset-consolidated'
dataset_dir = 'save/ppg-dataset-'

def modify_episode(memory: ReplayBuffer, episode_dir, index):
    try:
        episode_data = pickle.load(open(os.path.join(dataset_dir, episode_dir), 'rb'))
    except Exception as e:
        logging.info(e, "- Failed episode:", episode_dir)

    episode_data_list = []
    for data in episode_data:
        obs = data['obs']
        segmenter = ObjectSegmenter()
        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][1], obs['depth'][1], dir=None, plot=False)

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

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    new_dir = dataset_dir + "2"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    memory = ReplayBuffer(new_dir)


    episode_dirs = os.listdir(dataset_dir)
    
    for file_ in episode_dirs:
        if not file_.startswith("episode"):
            episode_dirs.remove(file_)

    for i, episode_dir in enumerate(episode_dirs):
         modify_episode(memory, episode_dir, i)

    logging.info(f"Dataset modified and saved in {new_dir}")
    