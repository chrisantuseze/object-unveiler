from random import Random
import random
import shutil
from typing import Any
import numpy as np
import pickle
import os
import cv2
import torch
from skimage import transform
import matplotlib.pyplot as plt

from mask_rg.object_segmenter import ObjectSegmenter
from trainer.memory import ReplayBuffer
import utils.logger as logging
import utils.general_utils as general_utils
import env.cameras as cameras
import policy.grasping as grasping
import policy.grasping2 as grasping2

# dataset_dir = 'save/pc-ou-dataset'
dataset_dir = 'save/ppg-dataset'

def modify_episode1(segmenter: ObjectSegmenter, episode_dir, index):
    try:
        episode_data = pickle.load(open(os.path.join(dataset_dir, episode_dir), 'rb'))
    except Exception as e:
        logging.info(e, "- Failed episode:", episode_dir)

    episode_data_list = []
    for data in episode_data:
        heightmap = data['state']
        object_masks = data['object_masks']

        object_masks, pred_mask, raw_masks, bboxes = segmenter.from_maskrcnn(data['color_obs'], bbox=True)

        new_masks = []
        masks = []
        new_bboxes = []
        for id, mask in enumerate(object_masks):
            mask = general_utils.resize_mask(transform, mask)
            masks.append(mask)
            new_masks.append(general_utils.extract_target_crop(mask, heightmap))

            new_bboxes.append(general_utils.resize_bbox(bboxes[id]))

        # get optimal nodes
        target_id = grasping.get_target_id(data['target_mask'], masks)
        objects_to_remove = grasping2.find_obstacles_to_remove(target_id, masks)
        # print(target_id, objects_to_remove[0])

        # show_images(masks, data['target_mask'], masks[objects_to_remove[0]], data['scene_mask'])

        traj_data = data['traj_data'][:20]
        if len(traj_data) == 0:
            print("len(traj_data):", len(traj_data))
            return

        transition = {
            'state': data['state'], 
            # 'target_mask': data['target_mask'], 
            'c_target_mask': general_utils.extract_target_crop(data['target_mask'], heightmap), 
            # 'scene_mask': data['scene_mask'],
            'c_object_masks': new_masks,
            # 'object_masks': object_masks,
            'action': data['action'],
            'optimal_nodes': objects_to_remove,
            'label': data['label'],
            'bboxes': new_bboxes,
            'target_id': target_id,
            'traj_data': traj_data,

            'actions': data['actions'],
        }
        episode_data_list.append(transition)

    memory.store_episode(episode_data_list)
    logging.info(f"{index} - Episode with dir {episode_dir} updated...")

def modify_transitions(memory: ReplayBuffer, transition_dir, idx):
    heightmap = cv2.imread(os.path.join(dataset_dir, transition_dir, 'heightmap.exr'), -1)
    target_mask = cv2.imread(os.path.join(dataset_dir, transition_dir, 'target_mask.png'), -1)
    action = pickle.load(open(os.path.join(dataset_dir, transition_dir, 'action'), 'rb'))
    full_state = action #pickle.load(open(os.path.join(dataset_dir, transition_dir, 'full_state'), 'rb'))

    color = cv2.imread(os.path.join(dataset_dir, transition_dir, 'color_0.png'), -1)
    depth = cv2.imread(os.path.join(dataset_dir, transition_dir, 'depth_0.exr'), -1)
    seg = cv2.imread(os.path.join(dataset_dir, transition_dir, 'seg_0.png'), -1)

    obs = {
        'color': [color],
        'depth': [depth],
        'seg': [seg],
        'full_state': full_state,
    }
    bounds = [[-0.25, 0.25], [-0.25, 0.25], [0.01, 0.3]]
    pxl_size = 0.005
    color_heightmap, depth_heightmap = general_utils.get_heightmap_(obs, cameras.RealSense.CONFIG, bounds, pxl_size)

    transition = {
        'state': heightmap,
        'target_mask': target_mask,
        'action': action,
        'obs': obs,
        'depth_heightmap': depth_heightmap,
        'color_heightmap': color_heightmap,
    }
    memory.store(transition)
    logging.info(f"{idx} - Episode with dir {transition_dir} updated...")

def show_images(obj_masks, target_mask, obstacle_mask, scene_mask):
    fig, ax = plt.subplots(len(obj_masks) + 3)

    ax[0].imshow(target_mask)
    ax[1].imshow(obstacle_mask)
    ax[2].imshow(scene_mask)

    k = 3
    for i in range(len(obj_masks)):
        obj_mask = obj_masks[i]

        ax[k].imshow(obj_mask)
        k += 1

    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    new_dir = 'save/ppg-dataset2'
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    memory = ReplayBuffer(new_dir)


    episode_dirs = os.listdir(dataset_dir)
    print("Total length:", len(episode_dirs))
    
    for file_ in episode_dirs:
        if not file_.startswith("episode"):
            print(file_)
            episode_dirs.remove(file_)

        # if not file_.startswith("transition"):
        #     episode_dirs.remove(file_)


    segmenter = ObjectSegmenter()
    for i, episode_dir in enumerate(episode_dirs):
        # modify_transitions(memory, episode_dir, i)

        modify_episode1(segmenter, episode_dir, i)

    logging.info(f"Dataset modified and saved in {new_dir}")
    