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
    def __getitem__1(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])
        heightmap, scene_color, target_mask, _, bboxes, _ = episode_data[0]

        padded_heightmap, padding_width_depth = general_utils.preprocess_image(heightmap, skip_transform=True)
        padded_target_mask, padding_width_target = general_utils.preprocess_image(target_mask)

        labels, rot_ids = [], []
        for data in episode_data:
            _, _, _, _, bboxes, action = data

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

    # single - input, multi - output and multi-obstacles for models_attn with processed inputs
    def __getitem__2(self, id):
        episode_data = self.memory.load_episode_attn(self.dir_ids[id])
        # based on my little observation, picking the first isn't horrible, but I feel the best should be the last since that has the most valid target.
        heightmap, scene_mask, target_mask, c_target_mask, object_masks, c_object_masks, objects_to_remove, bboxes, target_id, _ = episode_data[0]

        processed_heightmap, padding_width_depth = general_utils.preprocess_heightmap(heightmap)

        # commented out heightmap since we already extracted the crop in real-ou-dataset2
        processed_target_mask = general_utils.preprocess_target(c_target_mask)#, heightmap)
        processed_scene_mask = general_utils.preprocess_target(scene_mask)#, heightmap)

        _processed_obj_masks = []
        for obj_mask in c_object_masks:
            processed_obj_mask = general_utils.preprocess_target(obj_mask)#, heightmap)
            _processed_obj_masks.append(processed_obj_mask)
        _processed_obj_masks = np.array(_processed_obj_masks)

        # get labels and rot_ids
        labels, rot_ids, obstacle_ids = [], [], []
        for data in episode_data:
            heightmap, _, _, _, _, _, objects_to_remove, _, _, action = data # we use the different heightmap for the different steps here to ensure the labels reflect that.

            # we need one obstacle per episode
            obstacle_ids.append(objects_to_remove[0])

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

        # pad labels and rot_ids. The object_ids are not required anymore
        labels, rot_ids = self.pad_labels_and_rot(len(episode_data), processed_heightmap, labels, rot_ids)
        obstacle_ids = obstacle_ids[0] if obstacle_ids[0] < self.args.num_patches else self.args.num_patches-1 # Refer to notebook for why I did this.

        # pad object masks
        processed_obj_masks, obj_masks, bbox = self.pad_object_masks_and_nodes(_processed_obj_masks, c_object_masks, bboxes)

        return processed_heightmap, processed_target_mask, processed_obj_masks\
             , processed_scene_mask, rot_ids, labels, obstacle_ids, bbox

        # return processed_heightmap, processed_target_mask, processed_obj_masks\
        #         , processed_scene_mask, scene_mask, target_mask, obj_masks, rot_ids, labels, obstacle_ids, bbox

    # single - input, multi - output for models_attn with processed inputs
    def __getitem__(self, id):
        episode_data = self.memory.load_episode_attn(self.dir_ids[id])
        # based on my little observation, picking the first isn't horrible, but I feel the best should be the last since that has the most valid target.
        heightmap, c_target_mask, c_object_masks, objects_to_remove, bboxes, target_id, _ = episode_data[0]

        processed_heightmap, padding_width_depth = general_utils.preprocess_heightmap(heightmap)

        # commented out heightmap since we already extracted the crop in real-ou-dataset2
        processed_target_mask = general_utils.preprocess_target(c_target_mask)

        _processed_obj_masks = []
        for obj_mask in c_object_masks:
            processed_obj_mask = general_utils.preprocess_target(obj_mask)
            _processed_obj_masks.append(processed_obj_mask)
        _processed_obj_masks = np.array(_processed_obj_masks)

        # get labels and rot_ids
        labels, rot_ids, obstacle_ids = [], [], []
        for data in episode_data:
            _, _, _, _, _, _, objects_to_remove, _, _, action = data

            # we need one obstacle per episode
            obstacle_ids.append(objects_to_remove[0])

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

        # pad labels and rot_ids. The object_ids are not required anymore
        labels, rot_ids = self.pad_labels_and_rot(len(episode_data), processed_heightmap, labels, rot_ids)
        obstacle_ids = obstacle_ids[0] if obstacle_ids[0] < self.args.num_patches else self.args.num_patches-1 # Refer to notebook for why I did this.

        # pad object masks
        processed_obj_masks, obj_masks, bbox = self.pad_object_masks_and_nodes(_processed_obj_masks, c_object_masks, bboxes)

        return processed_heightmap, processed_target_mask, processed_obj_masks, rot_ids, labels, obstacle_ids, bbox

    def __len__(self):
        return len(self.dir_ids)
    
    def pad_labels_and_rot(self, episode_len, processed_heightmap, labels, rot_ids):
        seq_len = self.args.sequence_length
        if episode_len < seq_len:
            required_len = seq_len - len(labels)
            c, h, w = processed_heightmap.shape

            empty_array = np.zeros((c, w, h))
            labels = labels + [empty_array] * required_len

            rot_ids = rot_ids + [0] * required_len
        else:
            labels = labels[:seq_len]
            rot_ids = rot_ids[:seq_len]

        return np.array(labels), rot_ids
    
    def pad_heightmap_and_target(self, heightmaps, target_masks, targets_id):
        N, C, H, W = heightmaps.shape
        seq_len = self.args.sequence_length
        if N < seq_len:
            padded_heightmaps = np.zeros((seq_len, C, H, W), dtype=heightmaps.dtype)
            padded_heightmaps[:heightmaps.shape[0], :, :, :] = heightmaps

            padded_target_masks = np.zeros((seq_len, C, H, W), dtype=target_masks.dtype)
            padded_target_masks[:target_masks.shape[0], :, :, :] = target_masks

            padded_targets_id = np.zeros((seq_len, 1), dtype=targets_id.dtype)
            padded_targets_id[:len(targets_id)] = targets_id

        else:
            padded_heightmaps = heightmaps[:seq_len]
            padded_target_masks = target_masks[:seq_len]
            padded_targets_id = targets_id[:seq_len]

        return padded_heightmaps, padded_target_masks, padded_targets_id

    def pad_object_masks_and_nodes(self, _processed_obj_masks, object_masks, bboxes):
        N, C, H, W = _processed_obj_masks.shape
        object_masks = np.array(object_masks)
        bboxes = np.array(bboxes)
        if N < self.args.num_patches:
            processed_obj_masks = np.zeros((self.args.num_patches, C, H, W), dtype=_processed_obj_masks.dtype)
            processed_obj_masks[:_processed_obj_masks.shape[0], :, :, :] = _processed_obj_masks

            N, H, W = object_masks.shape
            obj_masks = np.zeros((self.args.num_patches, H, W), dtype=object_masks.dtype)
            obj_masks[:object_masks.shape[0], :, :] = object_masks

            num_zeros = self.args.num_patches - bboxes.shape[0]
            bbox = np.pad(bboxes, pad_width=((0, num_zeros), (0, 0)), mode='constant')
        else:
            processed_obj_masks = _processed_obj_masks[:self.args.num_patches]
            obj_masks = object_masks[:self.args.num_patches]
            bbox = bboxes[:self.args.num_patches]

        return processed_obj_masks, obj_masks, bbox