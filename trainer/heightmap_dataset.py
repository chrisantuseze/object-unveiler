import torch
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from skimage import transform, io

from trainer.memory import ReplayBuffer
import utils.utils as utils
import utils.logger as logging


class HeightMapDataset(data.Dataset):
    def __init__(self, args, dir_ids):
        super(HeightMapDataset, self).__init__()
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dir_ids = dir_ids
        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Resize to the input size expected by ResNet (can be adjusted)
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=(0.449), std=(0.226))
        ])

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
        # padded_data = np.expand_dims(padded_data, axis=0)

        # ensure data is float32 to prevent 'TypeError: Input type float64 is not supported' error
        padded_data = padded_data.astype(np.float32)

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
                transformed_heightmap = self.data_transform(padded_heightmap)

                object_mask = utils.resize_mask(transform, target_mask)
                padded_object_mask, _ = self._preprocess_data(object_mask)
                transformed_object_mask = self.data_transform(padded_object_mask)

            sequence.append((transformed_heightmap, transformed_object_mask))

            # convert theta to range 0-360 and then compute the rot_id
            angle = (action[2] + (2 * np.pi)) % (2 * np.pi)
            rot_id = round(angle / (2 * np.pi / 16))
            rot_ids.append(rot_id)
            
            action_area = np.zeros((heightmap.shape[0], heightmap.shape[1]))

            if int(action[1]) > 99 or int(action[0]):
                i = min(int(action[1]) * 0.95, 99)
                j = min(int(action[0]) * 0.95, 99)
            else:
                i = action[1]
                j = action[0]

            action_area[int(i), int(j)] = 1.0
            label = np.zeros((1, transformed_heightmap.shape[1], transformed_heightmap.shape[2])) # this was np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2])) before
            label[0, padded_heightmap_width_depth:padded_heightmap.shape[0] - padded_heightmap_width_depth,
                    padded_heightmap_width_depth:padded_heightmap.shape[1] - padded_heightmap_width_depth] = action_area
            
            labels.append(label)

        # pad dataset
        seq_len = self.args.sequence_length
        if len(episode_data) < seq_len:
            required_len = seq_len - len(sequence)
            c, h, w = sequence[0][0].shape

            empty_array = np.zeros((c, w, h))
            labels = labels + [empty_array] * required_len

            empty_tuple = (torch.zeros((c, w, h)), torch.zeros((c, w, h)), torch.zeros((c, w, h)))
            sequence = sequence + [empty_tuple] * required_len

            rot_ids = rot_ids + [0] * required_len


        # print(len(sequence), len(rot_ids), len(labels))
            
        return sequence, rot_ids, labels
    
    def __getitem__old1(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])

        sequence = []
        labels, rot_ids = [], []
        for data in episode_data:
            heightmap, target_mask, obstacle_mask, action = data

            padded_heightmap, padded_heightmap_width_depth = None, None
            if self.data_transform:
                padded_heightmap, padded_heightmap_width_depth = self._preprocess_data(heightmap)
                transformed_heightmap = self.data_transform(padded_heightmap)

                target_mask = utils.resize_mask(transform, target_mask)
                padded_target_mask, _ = self._preprocess_data(target_mask)
                transformed_target_mask = self.data_transform(padded_target_mask)

                obstacle_mask = utils.resize_mask(transform, obstacle_mask)
                padded_obstacle_mask, _ = self._preprocess_data(obstacle_mask)
                transformed_obstacle_mask = self.data_transform(padded_obstacle_mask)

            # Combine heightmap and mask into a single tensor and append to the sequence
            # input_data = torch.cat((heightmap, target_mask, obstacle_mask), dim=0)
            # sequence.append(input_data)
            sequence.append((transformed_heightmap, transformed_target_mask, transformed_obstacle_mask))

            # convert theta to range 0-360 and then compute the rot_id
            angle = (action[2] + (2 * np.pi)) % (2 * np.pi)
            rot_id = round(angle / (2 * np.pi / 16))
            rot_ids.append(rot_id)
            
            # logging.info("transformed_heightmap.shape:", transformed_heightmap.shape)
            # logging.info("padded_heightmap:", padded_heightmap.shape)


            action_area = np.zeros((heightmap.shape[0], heightmap.shape[1]))

            if int(action[1]) > 99 or int(action[0]):
                i = min(int(action[1]) * 0.95, 99)
                j = min(int(action[0]) * 0.95, 99)
            else:
                i = action[1]
                j = action[0]

            action_area[int(i), int(j)] = 1.0
            label = np.zeros((1, transformed_heightmap.shape[1], transformed_heightmap.shape[2])) # this was np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2])) before
            label[0, padded_heightmap_width_depth:padded_heightmap.shape[0] - padded_heightmap_width_depth,
                    padded_heightmap_width_depth:padded_heightmap.shape[1] - padded_heightmap_width_depth] = action_area
            
            labels.append(label)

        # pad dataset
        seq_len = self.args.sequence_length
        if len(episode_data) < seq_len:
            required_len = seq_len - len(sequence)
            c, h, w = sequence[0][0].shape

            empty_array = np.zeros((c, w, h))
            labels = labels + [empty_array] * required_len
            
            # empty_tuple = (torch.zeros((c, w, h)), torch.zeros((c, w, h)), torch.zeros((c, w, h)))
            # sequence = sequence + [empty_tuple] * required_len

            # rot_ids = rot_ids + [0] * required_len

        return sequence, rot_ids, labels

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
        label = np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2])) # this was np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2])) before
        label[0, padding_width_depth:padded_heightmap.shape[1] - padding_width_depth,
                 padding_width_depth:padded_heightmap.shape[2] - padding_width_depth] = action_area

        # return padded_heightmap, padded_target_mask, rot_id, label
        return padded_heightmap, rot_id, label
    
    def __len__(self):
        return len(self.dir_ids)