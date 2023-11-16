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

        optimal_nodes.sort()

        return scene_masks, target_mask, optimal_nodes

    def __len__(self):
        return len(self.dir_ids)