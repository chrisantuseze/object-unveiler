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


class SREDataset(data.Dataset):
    def __init__(self, args, dir_ids):
        super(SREDataset, self).__init__()
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dir_ids = dir_ids

        self.memory = ReplayBuffer(self.dataset_dir)

    # single - input, multi - output for models_attn with processed inputs
    def __getitem__old(self, id):
        target_mask, object_masks, objects_to_remove, bboxes = self.memory.load_episode_sre(self.dir_ids[id])

        # commented out heightmap since we already extracted the crop in real-ou-dataset2
        processed_target_mask = general_utils.preprocess_image(target_mask)[0]

        _processed_obj_masks = []
        for obj_mask in object_masks:
            processed_obj_mask = general_utils.preprocess_image(obj_mask)[0]
            _processed_obj_masks.append(processed_obj_mask)
        _processed_obj_masks = np.array(_processed_obj_masks)

        # pad object masks
        padded_processed_obj_masks, padded_bbox, is_valid = self.pad_old(_processed_obj_masks, bboxes)

        objects_to_remove = np.array(objects_to_remove[0] if objects_to_remove[0] < self.args.num_patches else 0)

        return processed_target_mask, padded_processed_obj_masks, padded_bbox, is_valid, objects_to_remove

    def pad_old(self, object_masks, bboxes):
        N, C, H, W = object_masks.shape
        object_masks = np.array(object_masks)
        bboxes = np.array(bboxes)

        is_valid = np.zeros(self.args.num_patches)
        is_valid[N:] = 0

        if N < self.args.num_patches:
            padded_obj_masks = np.zeros((self.args.num_patches, C, H, W), dtype=object_masks.dtype)
            padded_obj_masks[:object_masks.shape[0], :, :, :] = object_masks

            num_zeros = self.args.num_patches - bboxes.shape[0]
            bbox = np.pad(bboxes, pad_width=((0, num_zeros), (0, 0)), mode='constant')
        else:
            padded_obj_masks = object_masks[:self.args.num_patches]
            bbox = bboxes[:self.args.num_patches]

        is_valid = torch.from_numpy(is_valid).bool()
        return padded_obj_masks, bbox, is_valid

    # single - input, multi - output for models_attn with processed inputs
    def __getitem__(self, id):
        target_mask, object_masks, objects_to_remove, bboxes = self.memory.load_episode_sre(self.dir_ids[id])

        target_mask = target_mask.astype(np.float32)
        target_mask = np.expand_dims(target_mask, axis=0)

        object_masks = np.array(object_masks).astype(np.float32)
        _processed_obj_masks = np.expand_dims(object_masks, axis=1)

        # pad object masks
        padded_obj_masks, padded_bbox, is_valid = self.pad(_processed_obj_masks, bboxes)

        objects_to_remove = np.array(objects_to_remove[0] if objects_to_remove[0] < self.args.num_patches else 0)

        return target_mask, padded_obj_masks, padded_bbox, is_valid, objects_to_remove

    def __len__(self):
        return len(self.dir_ids)
    
    def pad(self, object_masks, bboxes):
        N, C, H, W = object_masks.shape
        object_masks = np.array(object_masks)
        bboxes = np.array(bboxes)

        is_valid = np.zeros(self.args.num_patches)
        is_valid[N:] = 0

        if N < self.args.num_patches:
            padded_obj_masks = np.zeros((self.args.num_patches, C, H, W), dtype=object_masks.dtype)
            padded_obj_masks[:object_masks.shape[0], :, :, :] = object_masks

            num_zeros = self.args.num_patches - bboxes.shape[0]
            bbox = np.pad(bboxes, pad_width=((0, num_zeros), (0, 0)), mode='constant')
        else:
            padded_obj_masks = object_masks[:self.args.num_patches]
            bbox = bboxes[:self.args.num_patches]

        is_valid = torch.from_numpy(is_valid).bool()
        return padded_obj_masks, bbox, is_valid