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


class UnveilerDataset(data.Dataset):
    def __init__(self, args, dir_ids):
        super(UnveilerDataset, self).__init__()
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dir_ids = dir_ids

        self.memory = ReplayBuffer(self.dataset_dir)

    # single - input, multi - output for models_attn
    def __getitem__old1(self, id):
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
        heightmap, scene_mask, target_mask, object_masks, optimal_nodes, _ = episode_data[0]

        processed_heightmap, padding_width_depth = general_utils.preprocess_heightmap(heightmap)
        processed_scene_mask = general_utils.preprocess_target(scene_mask)
        processed_target_mask = general_utils.preprocess_target(target_mask, heightmap)

        _processed_obj_masks = []
        for obj_mask in object_masks:
            processed_obj_mask = general_utils.preprocess_target(obj_mask, heightmap)
            _processed_obj_masks.append(processed_obj_mask)
        _processed_obj_masks = np.array(_processed_obj_masks)

        # get labels and rot_ids
        labels, rot_ids = [], []
        for data in episode_data:
            _, _, _, _, _, action = data

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
            label = np.zeros((1, processed_heightmap.shape[1], processed_heightmap.shape[2]))
            label[0, padding_width_depth:processed_heightmap.shape[1] - padding_width_depth,
                    padding_width_depth:processed_heightmap.shape[2] - padding_width_depth] = action_area
            
            labels.append(label)

        # pad labels and rot_ids
        labels, rot_ids = self.pad_labels_and_rot(episode_data, processed_heightmap, labels, rot_ids)

        # pad object masks
        processed_obj_masks, obj_masks, optimal_nodes = self.pad_object_masks(_processed_obj_masks, object_masks, optimal_nodes)

        return processed_heightmap, processed_scene_mask, processed_target_mask, processed_obj_masks, rot_ids, labels

        # return processed_heightmap, processed_scene_mask, processed_target_mask, processed_obj_masks, scene_mask, target_mask, obj_masks, rot_ids, labels
    
    def __len__(self):
        return len(self.dir_ids)
    
    def pad_labels_and_rot(self, episode_data, processed_heightmap, labels, rot_ids):
        seq_len = self.args.sequence_length
        if len(episode_data) < seq_len:
            required_len = seq_len - len(labels)
            c, h, w = processed_heightmap.shape

            empty_array = np.zeros((c, w, h))
            labels = labels + [empty_array] * required_len

            rot_ids = rot_ids + [0] * required_len

        labels = np.array(labels)
        return labels, rot_ids

    def pad_object_masks(self, _processed_obj_masks, object_masks, optimal_nodes):
        N, C, H, W = _processed_obj_masks.shape
        object_masks = np.array(object_masks)
        if N < self.args.num_patches:
            processed_obj_masks = np.zeros((self.args.num_patches, C, H, W), dtype=_processed_obj_masks.dtype)
            processed_obj_masks[:_processed_obj_masks.shape[0], :, :, :] = _processed_obj_masks

            N, H, W = object_masks.shape
            obj_masks = np.zeros((self.args.num_patches, H, W), dtype=object_masks.dtype)
            obj_masks[:object_masks.shape[0], :, :] = object_masks

        else:
            processed_obj_masks = _processed_obj_masks[:self.args.num_patches]
            obj_masks = object_masks[:self.args.num_patches]

        if len(optimal_nodes) < self.args.sequence_length:
            optimal_nodes = optimal_nodes + [0] * (self.args.sequence_length - len(optimal_nodes))
        else:
            optimal_nodes = optimal_nodes[:self.args.sequence_length]

        optimal_nodes = np.array(optimal_nodes)

        return processed_obj_masks, obj_masks, optimal_nodes