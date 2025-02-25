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

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

index = 0

def modify_episode(segmenter: ObjectSegmenter, episode_dir):
    global index
    try:
        episode_data = pickle.load(open(os.path.join(dataset_dir, episode_dir), 'rb'))

        episode_data_list = []
        for data in episode_data:
            heightmap = data['state']

            object_masks, pred_mask, raw_masks, bboxes = segmenter.from_maskrcnn(data['color_obs'], bbox=True)

            extracted_object_masks = []
            resized_object_masks = []
            resized_bboxes = []
            for id, mask in enumerate(object_masks):
                mask = general_utils.resize_mask(mask)
                resized_object_masks.append(mask)
                extracted_object_masks.append(general_utils.extract_target_crop(mask, heightmap))
                resized_bboxes.append(general_utils.resize_bbox(bboxes[id]))

            # get optimal nodes
            target_id = grasping.get_target_id(data['target_mask'], object_masks)
            objects_to_remove = grasping.find_obstacles_to_remove(target_id, object_masks)
            # print(target_id, objects_to_remove[0])

            transition = {
                'color_obs': data['color_obs'],
                'state': data['state'],
                'depth_heightmap': data['depth_heightmap'],
                'target_mask': data['target_mask'], 
                'c_target_mask': data['c_target_mask'], 
                'obstacle_mask': data['obstacle_mask'],
                'c_obstacle_mask': data['c_obstacle_mask'],
                'scene_mask': data['scene_mask'],
                'object_masks': object_masks,
                'c_object_masks': extracted_object_masks,
                'action': data['action'], 
                'label': data['label'],
                'bboxes': resized_bboxes,
                'target_id': target_id,
                'objects_to_remove': objects_to_remove,
            }
            episode_data_list.append(transition)

        memory.store_episode(episode_data_list)
        logging.info(f"{index} - Episode with dir {episode_dir} updated...")

        index += 1
        
    except Exception as e:
        logging.info(e, "- Failed episode:", episode_dir)

def modify_episode_act(episode_dir, index):
    try:
        episode_data = pickle.load(open(os.path.join(dataset_dir, episode_dir), 'rb'))
    except Exception as e:
        logging.info(e, "- Failed episode:", episode_dir)

    episode_data_list = []
    for data in episode_data:
        traj_data = data['traj_data']
        if len(traj_data) == 0:
            print("len(traj_data):", len(traj_data))
            return
        

        object_mask = general_utils.resize_mask(data['obstacle_mask'], new_size=(400, 400))
        cc_obstacle_mask = general_utils.extract_target_crop2(object_mask, data['color_obs'])
        transition = {
            'color_obs': data['color_obs'], 
            'depth_obs': data['depth_obs'], 
            'state': data['state'], 
            'depth_heightmap': data['depth_heightmap'],
            'target_mask': data['target_mask'], 
            'c_target_mask': data['c_target_mask'], 
            'obstacle_mask': data['obstacle_mask'],
            'c_obstacle_mask': data['c_obstacle_mask'], 
            'cc_obstacle_mask': cc_obstacle_mask,
            'scene_mask': data['scene_mask'],
            'action': data['action'], 
            'label': data['label'],
            'traj_data': data['traj_data'],
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

    extracted_target = general_utils.extract_target_crop(target_mask, heightmap)
    transition = {
        'state': heightmap,
        'target_mask': target_mask,
        'extracted_target': extracted_target,
        'action': action,
        'obs': obs,
        'depth_heightmap': depth_heightmap,
        'color_heightmap': color_heightmap,
    }
    memory.store(transition)
    logging.info(f"{idx} - Episode with dir {transition_dir} updated...")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    new_dir = "/home/e_chrisantus/Projects/grasping_in_clutter/object-unveiler/save/ppg-dataset-act672-2"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    memory = ReplayBuffer(new_dir)

    dataset_dir = "/home/e_chrisantus/Projects/grasping_in_clutter/object-unveiler/save/ppg-dataset-act672"


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

        modify_episode_act(episode_dir, i)
        # modify_episode(segmenter, episode_dir)

    logging.info(f"Dataset modified and saved in {new_dir}")
    