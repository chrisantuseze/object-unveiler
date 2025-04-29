#!/usr/bin/env python3
# import zipfile

# with zipfile.ZipFile("pc-ou-dataset-no-crop.zip", 'r') as zip_ref:
#     zip_ref.extractall("pc-ou-dataset-no-crop")


import argparse
import os

import cv2
from matplotlib import pyplot as plt
import torch
import yaml
from policy.policy import Policy
from trainer.memory import ReplayBuffer
from utils import general_utils

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', default='ae', type=str, help='')
    
    # args for eval_agent
    parser.add_argument('--ae_model', default='save/ae/ae_model_best.pt', type=str, help='')
    parser.add_argument('--sre_model', default='save/sre/sre_model_best.pt', type=str, help='')
    parser.add_argument('--fcn_model', default='save/fcn/fcn_model_best.pt', type=str, help='')
    parser.add_argument('--reg_model', default='downloads/reg_model.pt', type=str, help='')
    parser.add_argument('--seed', default=16, type=int, help='')
    parser.add_argument('--n_scenes', default=100, type=int, help='')
    parser.add_argument('--object_set', default='seen', type=str, help='')

    # args for trainer
    parser.add_argument('--dataset_dir', default='save/pc-ou-dataset', type=str, help='')
    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--lr', default=0.0001, type=float, help='')
    parser.add_argument('--batch_size', default=1, type=int, help='')
    parser.add_argument('--split_ratio', default=0.9, type=float, help='')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')

    parser.add_argument('--sequence_length', default=1, type=int, help='')
    parser.add_argument('--patch_size', default=64, type=int, help='')
    parser.add_argument('--num_patches', default=10, type=int, help='This should not be less than the maximum possible number of objects in the scene, which from list Environment.nr_objects is 9')
    parser.add_argument('--step', default=500, type=int, help='')

    # args for act
    parser.add_argument('--chunk_size', default=3, action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    return parser.parse_args()

def run_sre_policy():
    dataset_dir = "real_images/seg_data"

    transition_dirs = os.listdir(dataset_dir)
    
    for file_ in transition_dirs:
        if not file_.startswith("transition"):
            transition_dirs.remove(file_)
            
    memory = ReplayBuffer(dataset_dir)

    with open('yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)
        
    args = parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    policy = Policy(args, params)
    policy.load(ae_model=args.ae_model, reg_model=args.reg_model, sre_model=args.sre_model)

    for idx, transition_dir in enumerate(transition_dirs):
        print("Transition Directory:", transition_dir)
        scene_image, scene_mask, target_mask, bboxes, target_id, object_masks = memory.load_seg_data(transition_dirs, idx)
        scene_image = cv2.resize(scene_image, (400, 400)) 

        obstacle_id = policy.real_image_inference(target_mask, object_masks, bboxes)
        obstacle_mask = object_masks[obstacle_id]
 
        c_target_mask = general_utils.extract_target_crop2(target_mask, scene_image)
        c_obstacle_mask = general_utils.extract_target_crop2(obstacle_mask, scene_image)

        print("Target ID:", target_id)
        print("Obstacle ID:", obstacle_id)

        fig, ax = plt.subplots(2, 2)
        ax[0][0].imshow(scene_image)
        ax[0][0].set_title("Scene - Color")
        ax[0][0].axis("off")

        ax[0][1].imshow(scene_mask)
        ax[0][1].set_title("Scene - Grayscale")
        ax[0][1].axis("off")

        ax[1][0].imshow(c_target_mask)
        ax[1][0].set_title("Target")
        ax[1][0].axis("off")

        ax[1][1].imshow(c_obstacle_mask)
        ax[1][1].set_title("Obstacle")
        ax[1][1].axis("off")
        plt.show()


run_sre_policy()

