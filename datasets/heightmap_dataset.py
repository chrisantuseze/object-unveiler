import torch
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import os
from skimage import transform
import matplotlib.pyplot as plt

from trainer.memory import ReplayBuffer
import utils.general_utils as general_utils
from utils.constants import *


class HeightMapDataset(data.Dataset):
    def __init__(self, args, dir_ids):
        super(HeightMapDataset, self).__init__()
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dir_ids = dir_ids
        self.memory = ReplayBuffer(self.dataset_dir)

    # single - input, single - output for ppg-ou-dataset
    def __getitem__1(self, id):
        heightmap, target_mask, action = self.memory.load(self.dir_ids, id)

        padded_heightmap, padding_width_depth = general_utils.preprocess_image(heightmap)
        padded_target_mask = general_utils.preprocess_target(target_mask)

        # convert theta to range 0-360 and then compute the rot_id
        angle = (action[2] + (2 * np.pi)) % (2 * np.pi)
        rot_id = round(angle / (2 * np.pi / 16))

        action_area = np.zeros((heightmap.shape[0], heightmap.shape[1]))
        # action_area[int(action[1]), int(action[0])] = 1.0

        if int(action[1]) > 99 or int(action[0]):
            i = min(int(action[1]) * 0.95, 99)
            j = min(int(action[0]) * 0.95, 99)
        else:
            i = action[1]
            j = action[0]

        action_area[int(i), int(j)] = 1.0
        
        label = np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2]))
        label[0, padding_width_depth:padded_heightmap.shape[1] - padding_width_depth,
                 padding_width_depth:padded_heightmap.shape[2] - padding_width_depth] = action_area
        
        label = np.array(label)
        return padded_heightmap, padded_target_mask, rot_id, label

    # single - input, single - output for pc-ou-dataset with target action
    def __getitem__(self, id):
        episode_data = self.memory.load_episode_target_ppg(self.dir_ids[id])
        heightmap, c_obstacle_mask, action = episode_data[0]

        padded_heightmap, padding_width_depth = general_utils.preprocess_image(heightmap)
        padded_target_mask = general_utils.preprocess_target(c_obstacle_mask)

        # convert theta to range 0-360 and then compute the rot_id
        angle = (action[2] + (2 * np.pi)) % (2 * np.pi)
        rot_id = round(angle / (2 * np.pi / 16))

        action_area = np.zeros((heightmap.shape[0], heightmap.shape[1]))
        # action_area[int(action[1]), int(action[0])] = 1.0

        if int(action[1]) > 99 or int(action[0]):
            i = min(int(action[1]) * 0.95, 99)
            j = min(int(action[0]) * 0.95, 99)
        else:
            i = action[1]
            j = action[0]

        action_area[int(i), int(j)] = 1.0
        
        label = np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2])) # this was np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2])) before
        label[0, padding_width_depth:padded_heightmap.shape[1] - padding_width_depth,
                 padding_width_depth:padded_heightmap.shape[2] - padding_width_depth] = action_area
        
        return padded_heightmap, padded_target_mask, rot_id, label

    def __len__(self):
        return len(self.dir_ids)