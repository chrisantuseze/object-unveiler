import torch
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import os
from skimage import transform

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

    def __getitem__old1(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])

        sequence = []
        labels, rot_ids = [], []
        for data in episode_data:
            heightmap, _, target_mask, obstacle_mask, action = data

            padded_heightmap, padded_heightmap_width_depth = None, None
            if self.data_transform:
                padded_heightmap, padded_heightmap_width_depth = general_utils.preprocess_data(heightmap, root=2)
                transformed_heightmap = self.data_transform(padded_heightmap)

                target_mask = general_utils.resize_mask(transform, target_mask)
                padded_target_mask, _ = general_utils.preprocess_data(target_mask, root=2)
                transformed_target_mask = self.data_transform(padded_target_mask)

                obstacle_mask = general_utils.resize_mask(transform, obstacle_mask)
                padded_obstacle_mask, _ = general_utils.preprocess_data(obstacle_mask, root=2)
                transformed_obstacle_mask = self.data_transform(padded_obstacle_mask)

            sequence.append((transformed_heightmap, transformed_target_mask, transformed_obstacle_mask))

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

        return sequence, rot_ids, labels
    
    # mult - input, multi - output
    def __getitem__old2(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])

        sequence = []
        labels, rot_ids = [], []
        for data in episode_data:
            heightmap, _, target_mask, obstacle_mask, action = data

            padded_heightmap, padded_heightmap_width_depth = None, None
            if self.data_transform:
                padded_heightmap, padded_heightmap_width_depth = general_utils.preprocess_data(heightmap)
                transformed_heightmap = self.data_transform(padded_heightmap)

                target_mask = general_utils.resize_mask(transform, target_mask)
                padded_target_mask, _ = general_utils.preprocess_data(target_mask)
                transformed_target_mask = self.data_transform(padded_target_mask)

                obstacle_mask = general_utils.resize_mask(transform, obstacle_mask)
                padded_obstacle_mask, _ = general_utils.preprocess_data(obstacle_mask)
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

    # single - input, multi - output
    def __getitem__old3(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])
        heightmap, _, target_mask, _, _ = episode_data[0]

        # add extra padding (to handle rotations inside the network)
        diagonal_length_depth = float(heightmap.shape[0]) * np.sqrt(2)
        diagonal_length_depth = np.ceil(diagonal_length_depth / 16) * 16
        padding_width_depth = int((diagonal_length_depth - heightmap.shape[0]) / 2)
        padded_heightmap = np.pad(heightmap, padding_width_depth, 'constant', constant_values=-0.01)
        padded_heightmap = padded_heightmap.astype(np.float32)

        # try:
        #     target_mask = general_utils.resize_mask(transform, target_mask)
        # except:
        #     print(os.path.join(self.dataset_dir, self.dir_ids[id]))
            
        diagonal_length_target = float(target_mask.shape[0]) * np.sqrt(2)
        diagonal_length_target = np.ceil(diagonal_length_target / 16) * 16
        padding_width_target = int((diagonal_length_target - target_mask.shape[0]) / 2)
        padded_target_mask = np.pad(target_mask, padding_width_target, 'constant', constant_values=-0.01)
        padded_target_mask = padded_target_mask.astype(np.float32)

        # normalize heightmap
        image_mean = 0.01
        image_std = 0.03
        padded_heightmap = (padded_heightmap - image_mean)/image_std
        padded_target_mask = (padded_target_mask - image_mean)/image_std

        # add extra channel
        padded_heightmap = np.expand_dims(padded_heightmap, axis=0)
        padded_target_mask = np.expand_dims(padded_target_mask, axis=0)

        labels, rot_ids = [], []
        for data in episode_data:
            _, _, _, action = data

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
            label = np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2])) # this was np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2])) before
            label[0, padding_width_depth:padded_heightmap.shape[1] - padding_width_depth,
                    padding_width_depth:padded_heightmap.shape[2] - padding_width_depth] = action_area
            
            labels.append(label)

        # pad dataset
        seq_len = self.args.sequence_length
        if len(episode_data) < seq_len:
            required_len = seq_len - len(labels)
            c, h, w = padded_heightmap.shape

            empty_array = np.zeros((c, w, h))
            labels = labels + [empty_array] * required_len

            rot_ids = rot_ids + [0] * required_len

        return padded_heightmap, padded_target_mask, rot_ids, labels

    # single - input, single - output for ou-dataset with obstacle action
    def __getitem__old4(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])
        heightmap, _, target_mask, _, action = episode_data[0]

        # add extra padding (to handle rotations inside the network)
        diagonal_length_depth = float(heightmap.shape[0]) * np.sqrt(2)
        diagonal_length_depth = np.ceil(diagonal_length_depth / 16) * 16
        padding_width_depth = int((diagonal_length_depth - heightmap.shape[0]) / 2)
        padded_heightmap = np.pad(heightmap, padding_width_depth, 'constant', constant_values=-0.01)
        padded_heightmap = padded_heightmap.astype(np.float32)

        # try:
        #     target_mask = general_utils.resize_mask(transform, target_mask)
        # except:
        #     print(os.path.join(self.dataset_dir, self.dir_ids[id]))
            
        diagonal_length_target = float(target_mask.shape[0]) * np.sqrt(2)
        diagonal_length_target = np.ceil(diagonal_length_target / 16) * 16
        padding_width_target = int((diagonal_length_target - target_mask.shape[0]) / 2)
        padded_target_mask = np.pad(target_mask, padding_width_target, 'constant', constant_values=-0.01)
        padded_target_mask = padded_target_mask.astype(np.float32)

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
    
     # single - input, single - output for ou-dataset with target action
    
    # single - input, single - output for ou-dataset with target action and scene mask
    def __getitem__(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])
        heightmap, scene_mask, target_mask, _, action = episode_data[-1]

        # add extra padding (to handle rotations inside the network)
        diagonal_length_depth = float(heightmap.shape[0]) * np.sqrt(2)
        diagonal_length_depth = np.ceil(diagonal_length_depth / 16) * 16
        padding_width_depth = int((diagonal_length_depth - heightmap.shape[0]) / 2)
        padded_heightmap = np.pad(heightmap, padding_width_depth, 'constant', constant_values=-0.01)
        padded_heightmap = padded_heightmap.astype(np.float32)

        # # once new dataset with the logic to handle this at the collection stage iis collected, this becomes redundant
        # try:
        #     target_mask = general_utils.resize_mask(transform, target_mask)
        #     scene_mask = general_utils.resize_mask(transform, scene_mask)
        # except:
        #     print(os.path.join(self.dataset_dir, self.dir_ids[id]))
            
        diagonal_length_target = float(target_mask.shape[0]) * np.sqrt(2)
        diagonal_length_target = np.ceil(diagonal_length_target / 16) * 16
        padding_width_target = int((diagonal_length_target - target_mask.shape[0]) / 2)
        padded_target_mask = np.pad(target_mask, padding_width_target, 'constant', constant_values=-0.01)
        padded_target_mask = padded_target_mask.astype(np.float32)

        diagonal_length_scene = float(target_mask.shape[0]) * np.sqrt(2)
        diagonal_length_scene = np.ceil(diagonal_length_scene / 16) * 16
        padding_width_scene = int((diagonal_length_scene - target_mask.shape[0]) / 2)
        padded_scene_mask = np.pad(scene_mask, padding_width_scene, 'constant', constant_values=-0.01)
        padded_scene_mask = padded_scene_mask.astype(np.float32)

        # normalize heightmap
        image_mean = 0.01
        image_std = 0.03
        padded_heightmap = (padded_heightmap - image_mean)/image_std
        padded_target_mask = (padded_target_mask - image_mean)/image_std
        padded_scene_mask = (padded_scene_mask - image_mean)/image_std

        # add extra channel
        padded_heightmap = np.expand_dims(padded_heightmap, axis=0)
        padded_target_mask = np.expand_dims(padded_target_mask, axis=0)
        padded_scene_mask = np.expand_dims(padded_scene_mask, axis=0)

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
        
        return padded_heightmap, padded_target_mask, padded_scene_mask, rot_id, label
    

    # single - input, single - output for ppg-ou-dataset
    def __getitem__old5(self, id):
        heightmap, _, target_mask, action = self.memory.load(self.dir_ids, id)

        # add extra padding (to handle rotations inside the network)
        diagonal_length_depth = float(heightmap.shape[0]) * np.sqrt(2)
        diagonal_length_depth = np.ceil(diagonal_length_depth / 16) * 16
        padding_width_depth = int((diagonal_length_depth - heightmap.shape[0]) / 2)
        padded_heightmap = np.pad(heightmap, padding_width_depth, 'constant', constant_values=-0.01)
        padded_heightmap = padded_heightmap.astype(np.float32)

        # try:
        #     target_mask = general_utils.resize_mask(transform, target_mask)
        # except:
        #     print(os.path.join(self.dataset_dir, self.dir_ids[id]))
            
        diagonal_length_target = float(target_mask.shape[0]) * np.sqrt(2)
        diagonal_length_target = np.ceil(diagonal_length_target / 16) * 16
        padding_width_target = int((diagonal_length_target - target_mask.shape[0]) / 2)
        padded_target_mask = np.pad(target_mask, padding_width_target, 'constant', constant_values=-0.01)
        padded_target_mask = padded_target_mask.astype(np.float32)

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