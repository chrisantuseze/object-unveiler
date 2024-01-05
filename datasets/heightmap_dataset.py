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
        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize to the input size expected by ResNet (can be adjusted)
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=(0.449), std=(0.226))
        ])

        self.memory = ReplayBuffer(self.dataset_dir)

    # single - input, single - output for ppg-ou-dataset
    def __getitem__(self, id):
        heightmap, depth_heightmap, target_mask, action = self.memory.load(self.dir_ids, id)

        resized_target = general_utils.resize_mask(transform, target_mask)
        full_crop = general_utils.extract_target_crop(resized_target, heightmap)

        padded_heightmap, padding_width_depth = general_utils.preprocess_image(heightmap, skip_transform=True)
        padded_target_mask, padding_width_target = general_utils.preprocess_image(full_crop, skip_transform=True)

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

    # single - input, single - output for ou-dataset with target action
    def __getitem__old2(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])
        heightmap, _, target_mask, _, action = episode_data[-1]

        padded_heightmap, padding_width_depth = general_utils.preprocess_image(heightmap, skip_transform=True)
        padded_target_mask, padding_width_target = general_utils.preprocess_image(target_mask)

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