import torch
from torch.utils import data
import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from trainer.memory import ReplayBuffer
import utils.general_utils as general_utils
import utils.logger as logging

class ApertureDataset(data.Dataset):
    def __init__(self, args, dir_ids):
        super(ApertureDataset, self).__init__()
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dir_ids = dir_ids
        self.widths = np.array([0.6, 0.8, 1.1])
        self.crop_size = 32
        self.plot = False

        self.memory = ReplayBuffer(self.dataset_dir)

    def _get_single_aperture(self, heightmap, action):
        # add extra padding (to handle rotations inside the network)
        diag_length = float(heightmap.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 16) * 16
        padding_width = int((diag_length - heightmap.shape[0]) / 2)
        depth_heightmap = np.pad(heightmap, padding_width, 'constant')
        padded_shape = depth_heightmap.shape

        # rotate image (push always on the right)
        p1 = np.array([action[0], action[1]]) + padding_width
        theta = -((action[2] + (2 * np.pi)) % (2 * np.pi))
        rot = cv2.getRotationMatrix2D((int(padded_shape[0] / 2), int(padded_shape[1] / 2)),
                                      theta * 180 / np.pi, 1.0)
        rotated_heightmap = cv2.warpAffine(depth_heightmap, rot, (padded_shape[0], padded_shape[1]),
                                           flags=cv2.INTER_NEAREST)
        
        # compute the position of p1 on rotated heightmap
        rotated_pt = np.dot(rot, (p1[0], p1[1], 1.0))
        rotated_pt = (int(rotated_pt[0]), int(rotated_pt[1]))

        # crop heightmap
        cropped_map = np.zeros((2 * self.crop_size, 2 * self.crop_size), dtype=np.float32)
        y_start = max(0, rotated_pt[1] - self.crop_size)
        y_end = min(padded_shape[0], rotated_pt[1] + self.crop_size)
        x_start = rotated_pt[0]
        x_end = min(padded_shape[0], rotated_pt[0] + 2 * self.crop_size)
        cropped_map[0:y_end - y_start, 0:x_end - x_start] = rotated_heightmap[y_start: y_end, x_start: x_end]

        # normalize maps ( ToDo: find mean and std) # Todo
        image_mean = 0.01
        image_std = 0.03
        cropped_map = (cropped_map - image_mean) / image_std

        if self.plot:
            logging.info(action[3])

            p2 = np.array([0, 0])
            p2[0] = p1[0] + 20 * np.cos(action[2])
            p2[1] = p1[1] - 20 * np.sin(action[2])

            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(depth_heightmap)
            ax[0].plot(p1[0], p1[1], 'o', 2)
            ax[0].plot(p2[0], p2[1], 'x', 2)
            ax[0].arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], width=1)

            rotated_p2 = np.array([0, 0])
            rotated_p2[0] = rotated_pt[0] + 20 * np.cos(0)
            rotated_p2[1] = rotated_pt[1] - 20 * np.sin(0)
            ax[1].imshow(rotated_heightmap)
            ax[1].plot(rotated_pt[0], rotated_pt[1], 'o', 2)
            ax[1].plot(rotated_p2[0], rotated_p2[1], 'x', 2)
            ax[1].arrow(rotated_pt[0], rotated_pt[1], rotated_p2[0] - rotated_pt[0], rotated_p2[1] - rotated_pt[1], width=1)

            ax[2].imshow(cropped_map)
            plt.show()

        aperture_img = np.zeros((3, 2 * self.crop_size, 2 * self.crop_size))
        aperture_img[0] = cropped_map
        aperture_img[1] = cropped_map
        aperture_img[2] = cropped_map

        normalized_aperture = general_utils.min_max_scale(action[3], range=[0.6, 1.1], target_range=[0, 1])

        return aperture_img, np.array([normalized_aperture])
    
    def __getitem__(self, id):
        episode_data = self.memory.load_episode(self.dir_ids[id])

        aperture_imgs, norm_apertures = [], []
        for data in episode_data:
            heightmap, scene_mask, target_mask, obstacle_mask, action = data

            aperture_img, norm_aperture = self._get_single_aperture(heightmap, action)
            aperture_imgs.append(aperture_img)
            norm_apertures.append(norm_aperture)

        aperture_img_stack = torch.stack(aperture_imgs, dim=0)
        norm_aperture_stack = torch.stack(norm_apertures, dim=0)

        return aperture_img_stack, norm_aperture_stack
    
        # Stack the sequence along a new dimension to create the input tensor
        # aperture_img_stack = torch.stack(aperture_imgs, dim=0)
        # norm_aperture_stack = torch.stack(norm_apertures, dim=0)

        # return aperture_img_stack, norm_aperture_stack


    def __getitem__old(self, id):
        heightmap = cv2.imread(os.path.join(self.dataset_dir, self.dir_ids[id], 'heightmap.exr'), -1)

        # the target_mask might not be useful in this module 
        target_mask = cv2.imread(os.path.join(self.dataset_dir, self.dir_ids[id], 'target_mask.png'), -1)

        action = pickle.load(open(os.path.join(self.dataset_dir, self.dir_ids[id], 'action'), 'rb'))

        # add extra padding (to handle rotations inside the network)
        diag_length = float(heightmap.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 16) * 16
        padding_width = int((diag_length - heightmap.shape[0]) / 2)
        depth_heightmap = np.pad(heightmap, padding_width, 'constant')
        padded_shape = depth_heightmap.shape

        # rotate image (push always on the right)
        p1 = np.array([action[0], action[1]]) + padding_width
        theta = -((action[2] + (2 * np.pi)) % (2 * np.pi))
        rot = cv2.getRotationMatrix2D((int(padded_shape[0] / 2), int(padded_shape[1] / 2)),
                                      theta * 180 / np.pi, 1.0)
        rotated_heightmap = cv2.warpAffine(depth_heightmap, rot, (padded_shape[0], padded_shape[1]),
                                           flags=cv2.INTER_NEAREST)
        
        # compute the position of p1 on rotated heightmap
        rotated_pt = np.dot(rot, (p1[0], p1[1], 1.0))
        rotated_pt = (int(rotated_pt[0]), int(rotated_pt[1]))

        # crop heightmap
        cropped_map = np.zeros((2 * self.crop_size, 2 * self.crop_size), dtype=np.float32)
        y_start = max(0, rotated_pt[1] - self.crop_size)
        y_end = min(padded_shape[0], rotated_pt[1] + self.crop_size)
        x_start = rotated_pt[0]
        x_end = min(padded_shape[0], rotated_pt[0] + 2 * self.crop_size)
        cropped_map[0:y_end - y_start, 0:x_end - x_start] = rotated_heightmap[y_start: y_end, x_start: x_end]

        # normalize maps ( ToDo: find mean and std) # Todo
        image_mean = 0.01
        image_std = 0.03
        cropped_map = (cropped_map - image_mean) / image_std

        if self.plot:
            logging.info(action[3])

            p2 = np.array([0, 0])
            p2[0] = p1[0] + 20 * np.cos(action[2])
            p2[1] = p1[1] - 20 * np.sin(action[2])

            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(depth_heightmap)
            ax[0].plot(p1[0], p1[1], 'o', 2)
            ax[0].plot(p2[0], p2[1], 'x', 2)
            ax[0].arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], width=1)

            rotated_p2 = np.array([0, 0])
            rotated_p2[0] = rotated_pt[0] + 20 * np.cos(0)
            rotated_p2[1] = rotated_pt[1] - 20 * np.sin(0)
            ax[1].imshow(rotated_heightmap)
            ax[1].plot(rotated_pt[0], rotated_pt[1], 'o', 2)
            ax[1].plot(rotated_p2[0], rotated_p2[1], 'x', 2)
            ax[1].arrow(rotated_pt[0], rotated_pt[1], rotated_p2[0] - rotated_pt[0], rotated_p2[1] - rotated_pt[1], width=1)

            ax[2].imshow(cropped_map)
            plt.show()

        aperture_img = np.zeros((3, 2 * self.crop_size, 2 * self.crop_size))
        aperture_img[0] = cropped_map
        aperture_img[1] = cropped_map
        aperture_img[2] = cropped_map

        normalized_aperture = general_utils.min_max_scale(action[3], range=[0.6, 1.1], target_range=[0, 1])

        return aperture_img, np.array([normalized_aperture])
    
    def __len__(self):
        return len(self.dir_ids)



