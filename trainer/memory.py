from random import Random
import shutil
from typing import Any
import numpy as np
import pickle
import os
import cv2
import utils.logger as logging
import utils.general_utils as general_utils
import env.cameras as cameras

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class ReplayBuffer:
    def __init__(self, save_dir, buffer_size=100000) -> None:
        self.buffer_size = buffer_size
        self.random = Random()
        self.count = 0
        self.save_dir = save_dir
        self.buffer_ids = []

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.z_fill = 6

    def __call__(self, index) -> Any:
        return self.buffer_ids[index]
    
    def load_episode(self, episode):
        try:
            episode_data = pickle.load(open(os.path.join(self.save_dir, episode), 'rb'))
        except Exception as e:
            logging.info(e, "- Failed episode:", episode)

        data_list = []
        for data in episode_data:
            heightmap = data['state']
            target_mask = data['target_mask']
            obstacle_mask = data['obstacle_mask']
            action = data['action']
            scene_color = data['color_obs']

            data_list.append((heightmap, scene_color, target_mask, obstacle_mask, action))

        return data_list
    
    def load_episode_attn(self, episode):
        try:
            episode_data = pickle.load(open(os.path.join(self.save_dir, episode), 'rb'))
        except Exception as e:
            logging.info(e, "- Failed episode:", episode)

        data_list = []        
        for data in episode_data:
            heightmap = data['state']
            c_target_mask = data['c_target_mask']
            c_obstacle_mask = data['obstacle_mask']
            action = data['action']

            c_object_masks = data['c_object_masks']
            scene_mask = data['scene_mask']

            objects_to_remove = data['objects_to_remove']

            bboxes = data['bboxes']
            target_id = data['target_id']

            data_list.append((heightmap, scene_mask, c_target_mask, c_obstacle_mask, c_object_masks, objects_to_remove, bboxes, target_id, action))

        return data_list
    
    def load_episode_sre(self, episode):
        try:
            episode_data = pickle.load(open(os.path.join(self.save_dir, episode), 'rb'))
        except Exception as e:
            logging.info(e, "- Failed episode:", episode)

        data = episode_data[0]
        c_target_mask = data['c_target_mask']
        c_object_masks = data['c_object_masks']
        scene_mask = data['scene_mask']
        objects_to_remove = data['objects_to_remove']
        bboxes = data['bboxes']

        return scene_mask, c_target_mask, c_object_masks, objects_to_remove, bboxes
    
    def load_episode_decoder(self, episode):
        try:
            episode_data = pickle.load(open(os.path.join(self.save_dir, episode), 'rb'))
        except Exception as e:
            logging.info(e, "- Failed episode:", episode)

        data = episode_data[0]
        heightmap = data['state']
        c_obstacle_mask = data['c_obstacle_mask']
        action = data['action']

        return heightmap, c_obstacle_mask, action
    
    def store_episode(self, transition):
        folder_name = os.path.join(self.save_dir, 'episode_' + str(self.count).zfill(5))
        pickle.dump(transition, open(folder_name, 'wb'))

        self.buffer_ids.append(self.count)
        if self.count < self.buffer_size:
            self.count += 1

    def store(self, transition):
        folder_name = os.path.join(self.save_dir, 'transition_' + str(self.count).zfill(5))

        if os.path.exists(folder_name):
            # Try to remove the tree; if it fails, throw an error using try...except.
            try:
                shutil.rmtree(folder_name)
            except OSError as e:
                pass

        os.mkdir(folder_name)

        cv2.imwrite(os.path.join(folder_name, 'heightmap.exr'), transition['state'])
        cv2.imwrite(os.path.join(folder_name, 'target_mask.png'), transition['target_mask'])

        # cv2.imwrite(os.path.join(folder_name, 'depth_heightmap.png'), transition['depth_heightmap'])
        # cv2.imwrite(os.path.join(folder_name, 'color_heightmap.png'), transition['color_heightmap'])
        cv2.imwrite(os.path.join(folder_name, 'extracted_target.png'), transition['extracted_target'])

        pickle.dump(transition['action'], open(os.path.join(folder_name, 'action'), 'wb'))

        # Save everything that obs contains
        for i in range(len(transition['obs']['color'])):
            cv2.imwrite(os.path.join(folder_name, 'color_' + str(i) + '.png'), transition['obs']['color'][i])
            cv2.imwrite(os.path.join(folder_name, 'depth_' + str(i) + '.exr'), transition['obs']['depth'][i])
            cv2.imwrite(os.path.join(folder_name, 'seg_' + str(i) + '.png'), transition['obs']['seg'][i])

        pickle.dump(transition['obs']['full_state'], open(os.path.join(folder_name, 'full_state'), 'wb'))

        self.buffer_ids.append(self.count)
        if self.count < self.buffer_size:
            self.count += 1

    def load(self, dir_ids, idx):
        try:
            heightmap = cv2.imread(os.path.join(self.save_dir, dir_ids[idx], 'heightmap.exr'), -1)
            # target_mask = cv2.imread(os.path.join(self.save_dir, dir_ids[idx], 'target_mask.png'), -1)
            target_mask = cv2.imread(os.path.join(self.save_dir, dir_ids[idx], 'extracted_target.png'), -1)
            action = pickle.load(open(os.path.join(self.save_dir, dir_ids[idx], 'action'), 'rb'))
        except Exception as e:
            logging.info(e)
            idx += 1

        return heightmap, target_mask, action

    def sample(self, given_batch_size=0): # authors did not use given_batch_size
        batch_size = self.count if self.count < given_batch_size else given_batch_size
        batch_id = self.random.sample(self.buffer_ids, 1)[0]

        folder_name = os.path.join(self.save_dir, 'transition_' + str(self.count).zfill(5))
        state = cv2.imread(os.path.join(folder_name, 'heightmap.exr'), -1)
        action = pickle.load(open(os.path.join(folder_name, 'action'), 'rb'))

        return state, action
    
    def clear(self):
        self.buffer_ids.clear()
        self.count = 0
    
    def size(self):
        return self.count
    
    def seed(self, seed):
        self.random.seed(seed)