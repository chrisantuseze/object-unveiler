import torch
from torch.utils import data
import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import utils.utils as utils


class HeightMapDataset(data.Dataset):
    def __init__(self, dataset_dir, dir_ids):
        super(HeightMapDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.dir_ids = dir_ids

    def __getitem__(self, id):
        heightmap = cv2.imread(os.path.join(self.dataset_dir, self.dir_ids[id], 'heightmap.exr'), -1)
        target_mask = cv2.imread(os.path.join(self.dataset_dir, self.dir_ids[id], 'target_mask.png'), -1)
        action = pickle.load(open(os.path.join(self.dataset_dir, self.dir_ids[id], 'action'), 'rb'))

        # add extra padding (to handle rotations inside the network)
        diagonal_length = float(heightmap.shape[0]) * np.sqrt(2)
        diagonal_length = np.ceil(diagonal_length / 16) * 16
        padding_width = int((diagonal_length - heightmap.shape[0]) / 2)
        padded_heightmap = np.pad(heightmap, padding_width, 'constant', constant_values=-0.01)

        # normalize heightmap
        image_mean = 0.01
        image_std = 0.03
        padded_heightmap = (padded_heightmap - image_mean)/image_std

        # add extra channel
        padded_heightmap = np.expand_dims(padded_heightmap, axis=0)

        # convert theta to range 0-360 and then compute the rot_id
        angle = (action[2] + (2 * np.pi)) % (2 * np.pi)
        rot_id = round(angle / (2 * np.pi / 16))

        action_area = np.zeros((heightmap.shape[0], heightmap.shape[1]))
        action_area[int(action[1]), int(action[0])] = 1.0
        label = np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2]))
        label[0, padding_width:padded_heightmap.shape[1] - padding_width,
                 padding_width:padded_heightmap.shape[2] - padding_width] = action_area

        return padded_heightmap, target_mask, rot_id, label
    
    def __len__(self):
        return len(self.dir_ids)