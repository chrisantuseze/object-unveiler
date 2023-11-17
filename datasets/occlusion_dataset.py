import torch
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import os
from skimage import transform

from trainer.memory import ReplayBuffer
import utils.general_utils as general_utils
from utils.constants import *


class OcclusionDataset(data.Dataset):
    def __init__(self, args, dir_ids):
        super(OcclusionDataset, self).__init__()
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dir_ids = dir_ids
        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize to the input size expected by ResNet (can be adjusted)
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.449), std=(0.226))
        ])

        self.memory = ReplayBuffer(self.dataset_dir)

    def __getitem__(self, id):
        scene_masks, target_mask, optimal_nodes = self.memory.load_occlusion_data(self.dir_ids[id])
        
        # Convert the list of numpy arrays to a list of PyTorch tensors
        P = self.args.patch_size
        scene_masks = [torch.tensor(general_utils.resize_mask(transform, mask, new_size=(P, P))) for mask in scene_masks]
        required_len = self.args.num_patches - len(scene_masks)
        if len(scene_masks) < self.args.num_patches:
            scene_masks = scene_masks + [torch.zeros_like(scene_masks[0]) for _ in range(required_len)]
        else:
            scene_masks = scene_masks[:self.args.num_patches]

        scene_masks = torch.stack(scene_masks)

        target_mask = general_utils.resize_mask(transform, target_mask, new_size=(P, P))

        # Pad the list to the desired size
        label = np.zeros(self.args.sequence_length)
        if len(optimal_nodes) <= self.args.sequence_length:
            label[:len(optimal_nodes)] = np.array(optimal_nodes, dtype=np.int32)
        else:
            label = np.array(optimal_nodes[:self.args.sequence_length] , dtype=np.int32)

        assert (
            len(label) == self.args.sequence_length
        ), "Length of label should be same as the sequence length"
        
        # Convert to one-hot encoded list
        # label = np.eye(self.args.num_patches)[(label-1).astype(int)]
        label = np.array(self.one_hot_encoding(label, self.args.num_patches))
        # print(label)
        return scene_masks, target_mask, label
    
    def one_hot_encoding(self, lst, max_value):
        # Initialize an empty list to store the one-hot encoded vectors
        one_hot_encoded_list = []

        # Iterate through each value in the input list
        for value in lst:
            # Create a new one-hot encoded vector
            one_hot_encoded = [0] * max_value
            one_hot_encoded[int(value)] = 1

            # Append the one-hot encoded vector to the result list
            one_hot_encoded_list.append(one_hot_encoded)

        return one_hot_encoded_list

    def __len__(self):
        return len(self.dir_ids)