import pickle
import random
import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython

from trainer.memory import ReplayBuffer
import utils.general_utils as general_utils

from skimage.transform import resize

e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad
    

class ACTUnveilerDataset(torch.utils.data.Dataset):
    def __init__(self, args, dir_ids, dataset_dir, camera_names, norm_stats):
        super(ACTUnveilerDataset, self).__init__()
        self.args = args
        self.dataset_dir = dataset_dir
        self.dir_ids = dir_ids
        self.camera_names = camera_names
        self.norm_stats = norm_stats

        self.memory = ReplayBuffer(self.dataset_dir)

        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def load_episode(self, episode):
        try:
            episode_data = pickle.load(open(os.path.join(self.dataset_dir, episode), 'rb'))
        except Exception as e:
            print(e, "- Failed episode:", episode)

        data_list = []        
        for data in episode_data:
            # heightmap = data['state']
            # c_target_mask = data['c_target_mask']
            # action = data['action']

            images = data['images_traj']['color'] # picks only the color top and side camera images
            qpos = data['joints_traj'][0][0] # picks only the first trajectory and only its joints pos
            action = data['action']
            c_target_mask = data['target_mask']

            data_list.append((images, qpos, action, c_target_mask))

        return data_list

    def __getitem__(self, id):
        episode_data = self.load_episode(self.dir_ids[id])

        images, qpos, action, c_target_mask = episode_data[-1] # images is a list containing the front and top camera images 

        sequence_len = self.args['policy_config']['num_queries']
        action = np.array(action, dtype=np.float32)
        action = action.reshape(1, action.shape[0])
        print("action.shape", action.shape)

        action_len = action.shape[0]
        padded_action = np.zeros((sequence_len, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(sequence_len)
        is_pad[action_len:] = 1

        image_dict = dict()
        for cam_name in self.camera_names:
            if cam_name == 'front':
                image_dict[cam_name] = images[0]
            elif cam_name == 'top':
                image_dict[cam_name] = images[1]
            elif cam_name == 'target':
                image_dict[cam_name] = process_target_image(c_target_mask)

        print("images[0].shape", images[0].shape)
        print("c_target_mask.shape", c_target_mask.shape, image_dict['target'].shape)

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        print("image_data.shape", image_data.shape)

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        print("image_data.shape", image_data.shape)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        return image_data, qpos_data, action_data, is_pad
    
    def __len__(self):
        return len(self.dir_ids)
    
    
def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats

def get_stats(dataset_dir, transition_dirs):
    episode_data = pickle.load(open(os.path.join(dataset_dir, transition_dirs[0]), 'rb'))
    all_action_data, all_qpos_data = [], []
    for data in episode_data:
        action = np.array(data['action'])
        all_action_data.append(torch.from_numpy(action))

        qpos = np.array(data['joints_traj'][0][0])
        all_qpos_data.append(torch.from_numpy(qpos))


    all_action_data = torch.stack(all_action_data)
    all_qpos_data = torch.stack(all_qpos_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze()}

    return stats

def load_data(args, dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')

    args['batch_size'] = batch_size_train
    transition_dirs = os.listdir(dataset_dir)
    for file_ in transition_dirs:
        if not file_.startswith("episode"):
            transition_dirs.remove(file_)

    # split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)

    transition_dirs = transition_dirs[:10000]

    split_index = int(args['split_ratio'] * len(transition_dirs))
    train_ids = transition_dirs[:split_index]
    val_ids = transition_dirs[split_index:]

    # this ensures that the split is done properly without causing input mismatch error
    data_length = (len(train_ids)//args['batch_size']) * args['batch_size']
    train_ids = train_ids[:data_length]

    data_length = (len(val_ids)//args['batch_size']) * args['batch_size']
    val_ids = val_ids[:data_length]

    norm_stats = get_stats(dataset_dir, transition_dirs)

    # construct dataset and dataloader
    train_dataset = ACTUnveilerDataset(args, train_ids, dataset_dir, camera_names, norm_stats)

    val_dataset = ACTUnveilerDataset(args, train_ids, dataset_dir, camera_names, norm_stats)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def process_target_image(target_image):
    # Resize the image
    resized_image = resize(target_image, (480, 640), anti_aliasing=True)

    # Add color channels
    return np.stack((resized_image,) * 3, axis=-1)