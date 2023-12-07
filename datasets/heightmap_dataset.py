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

        labels = np.array(labels)
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

        labels = np.array(labels)
        return sequence, rot_ids, labels

    # single - input, multi - output
    def __getitem__old3(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])
        heightmap, _, target_mask, _, _ = episode_data[0]

        padded_heightmap, padding_width_depth = general_utils.preprocess_image(heightmap, skip_transform=True)
        padded_target_mask, padding_width_target = general_utils.preprocess_image(target_mask)

        labels, rot_ids = [], []
        for data in episode_data:
            _, _, _, _, action = data

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

        labels = np.array(labels)
        return padded_heightmap, padded_target_mask, rot_ids, labels

    # single - input, multi - output for models_attn
    def __getitem__old31(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])
        heightmap, scene_color, target_mask, _, _ = episode_data[0]

        padded_heightmap, padding_width_depth = general_utils.preprocess_image(heightmap, skip_transform=True)
        padded_target_mask, padding_width_target = general_utils.preprocess_image(target_mask)

        labels, rot_ids = [], []
        for data in episode_data:
            _, _, _, _, action = data

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

        labels = np.array(labels)
        return scene_color, target_mask, rot_ids, labels

    # single - input, multi - output for models_attn with processed inputs
    def __getitem__(self, id):
        episode_data = self.memory.load_episode_attn(self.dir_ids[id])
        heightmap, scene_mask, target_mask, object_masks, _ = episode_data[0]

        padded_heightmap, padding_width_depth = general_utils.preprocess_image(heightmap, skip_transform=True)
        padded_scene_mask, _ = general_utils.preprocess_image(scene_mask)
        padded_target_mask, _ = general_utils.preprocess_image(target_mask)

        padded_obj_masks = []
        for obj_mask in object_masks:
            padded_obj_mask, _ = general_utils.preprocess_image(obj_mask)
            padded_obj_masks.append(padded_obj_mask)

        padded_obj_masks = np.array(padded_obj_masks)

        labels, rot_ids = [], []
        for data in episode_data:
            _, _, _, _, action = data

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


        N, C, H, W = padded_obj_masks.shape
        object_masks = np.array(object_masks)
        if N < self.args.num_patches:
            new_padded_obj_masks = np.zeros((self.args.num_patches, C, H, W), dtype=padded_obj_masks.dtype)
            new_padded_obj_masks[:padded_obj_masks.shape[0], :, :, :] = padded_obj_masks

            N, H, W = object_masks.shape
            new_obj_masks = np.zeros((self.args.num_patches, H, W), dtype=object_masks.dtype)
            new_obj_masks[:object_masks.shape[0], :, :] = object_masks

        else:
            new_padded_obj_masks = padded_obj_masks[:self.args.num_patches]
            new_obj_masks = object_masks[:self.args.num_patches]

        labels = np.array(labels)
        return padded_scene_mask, padded_target_mask, new_padded_obj_masks, scene_mask, target_mask, new_obj_masks, rot_ids, labels

    # single - input, single - output for ou-dataset with obstacle action
    def __getitem__old4(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])
        heightmap, _, target_mask, _, action = episode_data[0]

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
        
        labels = np.array(labels)
        return padded_heightmap, padded_target_mask, rot_id, label
    
     # single - input, single - output for ou-dataset with target action
    
    # single - input, single - output for ou-dataset with target action
    def __getitem__old5(self, id):
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
        
        labels = np.array(labels)
        return padded_heightmap, padded_target_mask, rot_id, label
    

    # single - input, single - output for ppg-ou-dataset
    def __getitem__old6(self, id):
        heightmap, target_mask, action = self.memory.load(self.dir_ids, id)

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
        
        label = np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2]))
        label[0, padding_width_depth:padded_heightmap.shape[1] - padding_width_depth,
                 padding_width_depth:padded_heightmap.shape[2] - padding_width_depth] = action_area
        
        labels = np.array(labels)
        return padded_heightmap, padded_target_mask, rot_id, label
    
    def __len__(self):
        return len(self.dir_ids)