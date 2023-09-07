import torch
from torch.utils import data
import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from skimage import transform, io

from trainer.memory import ReplayBuffer
import utils.utils as utils


class HeightMapDataset(data.Dataset):
    def __init__(self, dataset_dir, dir_ids, sequence_length, data_transform=None):
        super(HeightMapDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.dir_ids = dir_ids
        self.sequence_length = sequence_length
        self.data_transform = data_transform

        self.memory = ReplayBuffer(self.dataset_dir)

    def _preprocess_data(self, data):
         # add extra padding (to handle rotations inside the network)
        diagonal_length_data = float(data.shape[0]) * np.sqrt(2)
        diagonal_length_data = np.ceil(diagonal_length_data / 16) * 16
        padding_width_data = int((diagonal_length_data - data.shape[0]) / 2)
        padded_data = np.pad(data, padding_width_data, 'constant', constant_values=-0.01)

        # normalize heightmap
        image_mean = 0.01
        image_std = 0.03
        padded_data = (padded_data - image_mean)/image_std

        # add extra channel
        padded_data = np.expand_dims(padded_data, axis=0)

        return padded_data, padding_width_data

    def __getitem__(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])

        sequence = []
        labels, rot_ids = [], []
        for data in episode_data:
            heightmap, target_mask, obstacle_mask, action = data

            padded_heightmap, padded_heightmap_width_depth = None, None
            if self.data_transform:
                padded_heightmap, padded_heightmap_width_depth = self._preprocess_data(heightmap)
                heightmap = self.data_transform(padded_heightmap)

                target_mask = utils.resize_mask(transform, target_mask)
                target_mask = self._preprocess_data(target_mask)
                target_mask = self.data_transform(target_mask)

                obstacle_mask = utils.resize_mask(transform, obstacle_mask)
                obstacle_mask = self._preprocess_data(obstacle_mask)
                obstacle_mask = self.data_transform(obstacle_mask)

            # Combine heightmap and mask into a single tensor and append to the sequence
            # input_data = torch.cat((heightmap, target_mask, obstacle_mask), dim=0)
            # sequence.append(input_data)
            sequence.append((heightmap, target_mask, obstacle_mask))

            # convert theta to range 0-360 and then compute the rot_id
            angle = (action[2] + (2 * np.pi)) % (2 * np.pi)
            rot_id = round(angle / (2 * np.pi / 16))
            rot_ids.append(rot_id)

            action_area = np.zeros((heightmap.shape[0], heightmap.shape[1]))
            action_area[int(action[1]), int(action[0])] = 1.0
            label = np.zeros((2, padded_heightmap.shape[1], padded_heightmap.shape[2])) # this was np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2])) before
            label[0, padded_heightmap_width_depth:padded_heightmap.shape[1] - padded_heightmap_width_depth,
                    padded_heightmap_width_depth:padded_heightmap.shape[2] - padded_heightmap_width_depth] = action_area
            
            labels.append(label)

        return sequence, rot_ids, labels
        
        # Stack the sequence along a new dimension to create the input tensor
        # input_data_stack = torch.stack(sequence, dim=0)
        # rot_id_stack = torch.stack(rot_ids, dim=0)
        # label_stack = torch.stack(labels, dim=0)
        
        # return input_data_stack, rot_id_stack, label_stack

    def __getitem__old(self, id):
        heightmap = cv2.imread(os.path.join(self.dataset_dir, self.dir_ids[id], 'heightmap.exr'), -1)
        target_mask = cv2.imread(os.path.join(self.dataset_dir, self.dir_ids[id], 'target_mask.png'), -1)
        action = pickle.load(open(os.path.join(self.dataset_dir, self.dir_ids[id], 'action'), 'rb'))

        # add extra padding (to handle rotations inside the network)
        diagonal_length_depth = float(heightmap.shape[0]) * np.sqrt(2)
        diagonal_length_depth = np.ceil(diagonal_length_depth / 16) * 16
        padding_width_depth = int((diagonal_length_depth - heightmap.shape[0]) / 2)
        padded_heightmap = np.pad(heightmap, padding_width_depth, 'constant', constant_values=-0.01)

        diagonal_length_target = float(target_mask.shape[0]) * np.sqrt(2)
        diagonal_length_target = np.ceil(diagonal_length_target / 16) * 16
        padding_width_target = int((diagonal_length_target - target_mask.shape[0]) / 2)
        padded_target_mask = np.pad(target_mask, padding_width_target, 'constant', constant_values=-0.01)

        # normalize heightmap
        image_mean = 0.01
        image_std = 0.03
        padded_heightmap = (padded_heightmap - image_mean)/image_std
        padded_target_mask = (padded_target_mask - image_mean)/image_std

        # add extra channel
        padded_heightmap = np.expand_dims(padded_heightmap, axis=0)
        padded_target_mask = np.expand_dims(padded_target_mask, axis=0)

        # convert theta to range 0-360 and then compute the rot_id
        angle = (action[2] + (2 * np.pi)) % (2 * np.pi)
        rot_id = round(angle / (2 * np.pi / 16))

        action_area = np.zeros((heightmap.shape[0], heightmap.shape[1]))
        action_area[int(action[1]), int(action[0])] = 1.0
        label = np.zeros((2, padded_heightmap.shape[1], padded_heightmap.shape[2])) # this was np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2])) before
        label[0, padding_width_depth:padded_heightmap.shape[1] - padding_width_depth,
                 padding_width_depth:padded_heightmap.shape[2] - padding_width_depth] = action_area

        return padded_heightmap, padded_target_mask, rot_id, label
    
    def __len__(self):
        return len(self.dir_ids)