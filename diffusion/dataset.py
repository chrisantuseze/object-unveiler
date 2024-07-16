from act.utils_act import ACTUnveilerDataset
import torch
import numpy as np
import warnings
import random
import os
import torch

from act.utils_act import get_stats

class NormalizeDiffusionActionQpos:
    def __init__(self, norm_stats):
        # since the values of the qpos and action are tied together
        # (current position, goal position), we normalize them together
        self.action_min = norm_stats["action_min"]
        self.action_max = norm_stats["action_max"]
        self.qpos_min = norm_stats["qpos_min"]
        self.qpos_max = norm_stats["qpos_max"]
    
    def __call__(self, qpos, action):
        qpos = (qpos - self.qpos_min) / (self.qpos_max - self.qpos_min)
        action = (action - self.action_min) / (self.action_max - self.action_min)

        qpos = (qpos*2)-1
        action = (action*2)-1

        assert np.min(action) >= -1 and np.max(action) <= 1
        assert np.min(qpos) >= -1 and np.max(qpos) <= 1

        if np.min(action) < -1 or np.max(action) > 1:
            warnings.warn(f"outside bounds for action min or max, got min, max action: {np.min(action),np.max(action)}")
            
        if np.min(qpos) < -1 or np.max(qpos) > 1:
            warnings.warn(f"outside bounds for qpos min or max, got min max qpos: {np.min(qpos),np.max(qpos)}")

        return qpos, action
    
    def unnormalize(self, qpos, action):

        new_qpos = (qpos + 1)/2 
        new_qpos = (new_qpos*(self.qpos_max-self.qpos_min))+self.qpos_min

        print("max qpos:", self.qpos_max)
        print("min qpos:", self.qpos_min)
        print("max action:", self.action_max)
        print("min action:", self.action_min)

        new_action = (action+1)/2
        new_action = (new_action*(self.action_max-self.action_min))+self.action_min

        return new_qpos, new_action


class DiffusionEpisodicDataset(ACTUnveilerDataset):

    def __init__(self, args, dir_ids, dataset_dir, camera_names, norm_stats):
        self.gel_idx = None
        super().__init__(args, dir_ids, dataset_dir, camera_names, norm_stats)

        self.action_qpos_normalize = NormalizeDiffusionActionQpos(norm_stats)
        self.camera_names = camera_names

    def __getitem__(self, index):        

        if self.gel_idx == None:
            return super().__getitem__(index)
        
        all_cam_images, qpos_data, action_data, is_pad = super().__getitem__(index)
        # because we used the super init, everything is already normalized

        nsample = dict()

        # change the padding behavior so the robot stays in the same position at the end
        if any(is_pad): 
            last_idx = torch.where(is_pad==0)[0][-1]
            last_action = action_data[last_idx]

            action_data[last_idx+1:] = last_action
        
        # add all cameras
        for i, cam in enumerate(self.camera_names):
            nsample[cam] = torch.stack([all_cam_images[i],]) 

        nsample['agent_pos'] = torch.stack([qpos_data,])
        nsample['action'] = action_data

        return nsample
    

def load_data(args, dataset_dir, camera_names, batch_size_train, batch_size_val):
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
    train_dataset = DiffusionEpisodicDataset(args, train_ids, dataset_dir, camera_names, norm_stats)

    val_dataset = DiffusionEpisodicDataset(args, train_ids, dataset_dir, camera_names, norm_stats)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True,
        prefetch_factor=4
        # this leads to an error message about shutting down workers at the end but 
        # does not affect training/model output
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

