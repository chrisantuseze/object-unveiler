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
    def __getitem__(self, id):
        scene_mask, target_mask, object_masks, objects_to_remove, bboxes = self.memory.load_episode_sre(self.dir_ids[id])

        # commented out heightmap since we already extracted the crop in real-ou-dataset2
        # processed_target_mask = general_utils.preprocess_sre_mask(target_mask)
        processed_target_mask = general_utils.preprocess_image(target_mask)[0]

        _processed_obj_masks = []
        for obj_mask in object_masks:
            # processed_obj_mask = general_utils.preprocess_sre_mask(obj_mask)
            processed_obj_mask = general_utils.preprocess_image(obj_mask)[0]
            _processed_obj_masks.append(processed_obj_mask)
        _processed_obj_masks = np.array(_processed_obj_masks)

        # pad object masks
        padded_processed_obj_masks, padded_obj_masks, padded_bbox = self.pad(_processed_obj_masks, object_masks, bboxes)

        raw_scene_mask, raw_target_mask = np.array(scene_mask), np.array(target_mask)
        objects_to_remove = np.array(objects_to_remove[0] if objects_to_remove[0] < self.args.num_patches else 0)

        return processed_target_mask, padded_processed_obj_masks, padded_bbox, objects_to_remove, raw_scene_mask, raw_target_mask, padded_obj_masks

    def __len__(self):
        return len(self.dir_ids)
    
    def pad(self, _processed_obj_masks, object_masks, bboxes):
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