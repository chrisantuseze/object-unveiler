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

dataset_dir = 'save/ou-dataset-consolidated'
# dataset_dir = 'save/ppg-dataset-'

def modify_episode(memory: ReplayBuffer, episode_dir):
    try:
        episode_data = pickle.load(open(os.path.join(dataset_dir, episode_dir), 'rb'))
    except Exception as e:
        logging.info(e, "- Failed episode:", episode_dir)

    episode_data_list = []
    for data in episode_data:
        obs = data['obs']
        segmenter = ObjectSegmenter()
        _, pred_mask, _ = segmenter.from_maskrcnn(obs['color'][1], obs['depth'][1], dir=None, plot=False)
    
        transition = {
            'state': data['state'], 
            'target_mask': general_utils.resize_mask(transform, data['target_mask']), 
            'obstacle_mask': general_utils.resize_mask(transform, data['obstacle_mask']),
            'scene_mask': general_utils.resize_mask(transform, pred_mask),
            'action': data['action'], 
            'label': data['label']
        }
        episode_data_list.append(transition)

    memory.store_episode(episode_data_list)
    logging.info(f"Episode with dir {episode_dir} updated...")

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

    for episode_dir in episode_dirs:
         modify_episode(memory, episode_dir)

    logging.info(f"Dataset modified and saved in {new_dir}")
    