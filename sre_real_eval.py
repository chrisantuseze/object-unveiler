import argparse
import os

import cv2
from matplotlib import pyplot as plt
import numpy as np
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
    parser.add_argument('--sre_model', default='save/sre/sre_model_last.pt', type=str, help='')
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
    parser.add_argument('--chunk_size', default=5, action='store', type=int, help='chunk_size', required=False)
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
        if transition_dir[-2:] != "01":
            continue

        scene_image, scene_mask, target_mask, bboxes, target_id, object_masks = memory.load_seg_data(transition_dirs, idx)
        scene_image = cv2.resize(scene_image, (400, 400)) 

        obstacle_id = policy.real_image_inference(target_mask, object_masks, bboxes)
        obstacle_mask = object_masks[3]
 
        c_target_mask = general_utils.extract_target_crop2(target_mask, scene_image)
        sre_obstacle_mask = general_utils.extract_target_crop2(obstacle_mask, scene_image)

        clip_obstacle_mask = general_utils.extract_target_crop2(object_masks[2], scene_image)
        gpt_no_obstacle_mask = general_utils.extract_target_crop2(object_masks[obstacle_id], scene_image)

        cv2.imwrite('scene_image.png', scene_image)
        cv2.imwrite('c_target_mask.png', c_target_mask)
        cv2.imwrite('sre_obstacle_mask.png', sre_obstacle_mask)
        cv2.imwrite('clip_obstacle_mask.png', clip_obstacle_mask)
        cv2.imwrite('gpt_no_obstacle_mask.png', gpt_no_obstacle_mask)

        fig, ax = plt.subplots(1, len(object_masks))
        for i, mask in enumerate(object_masks):
            image = general_utils.extract_target_crop2(mask, scene_image)
            cv2.imwrite(f'object_{i}.png', image)

            ax[i].imshow(mask)
            ax[i].axis("off")
        plt.show()
        
        cv2.imwrite('scene_mask.png', scene_mask)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(scene_image)
        ax[1].imshow(scene_mask)
        plt.show()

        print("Target ID:", target_id)
        print("Obstacle ID:", obstacle_id)
        print()

        # fig, ax = plt.subplots(2, 2)
        # ax[0][0].imshow(scene_image)
        # ax[0][0].set_title("Scene - Color")
        # ax[0][0].axis("off")

        # ax[0][1].imshow(scene_mask)
        # ax[0][1].set_title("Scene - Grayscale")
        # ax[0][1].axis("off")

        # ax[1][0].imshow(c_target_mask)
        # ax[1][0].set_title("Target")
        # ax[1][0].axis("off")

        # ax[1][1].imshow(c_obstacle_mask)
        # ax[1][1].set_title("Obstacle")
        # ax[1][1].axis("off")
        # plt.show()

def make_video_visualization():
    dataset_dir = "real_images/video_seg_data"

    transition_dirs = os.listdir(dataset_dir)

    for file_ in transition_dirs:
        if not file_.startswith("transition"):
            transition_dirs.remove(file_)
            
    print("Transition Directories:", transition_dirs)

    transition_dirs = ['transition_00000', 'transition_00001', 'transition_00002']
    print("Transition Directories:", transition_dirs)
    memory = ReplayBuffer(dataset_dir)

    scenes = [
        { #0
            'target_id': 8,
            'obstacle_ids': [2, 8],
            'scene_image_dir': ['real_images/video_seg_data/0/1/scene_image.png', 'real_images/video_seg_data/0/2/scene_image.png'],
            'scene_mask_dir': ['real_images/video_seg_data/0/1/scene_mask.png', 'real_images/video_seg_data/0/2/scene_mask.png'],
        },
        { #1
            'target_id': 2,
            'obstacle_ids': [2],
            'scene_image_dir': ['real_images/video_seg_data/1/1/scene_image.png'],
            'scene_mask_dir': ['real_images/video_seg_data/1/1/scene_mask.png'],
        },
        { #2
            'target_id': 10,
            'obstacle_ids': [1, 5, 10],
            'scene_image_dir': ['real_images/video_seg_data/2/1/scene_image.png', 'real_images/video_seg_data/2/2/scene_image.png', 'real_images/video_seg_data/2/3/scene_image.png'],
            'scene_mask_dir': ['real_images/video_seg_data/2/1/scene_mask.png', 'real_images/video_seg_data/2/2/scene_mask.png', 'real_images/video_seg_data/2/3/scene_mask.png'],
        },
    ]

    for idx, transition_dir in enumerate(transition_dirs):
        scene_image, scene_mask, target_mask, bboxes, target_id, object_masks = memory.load_seg_data(transition_dirs, idx)
        scene_image = cv2.resize(scene_image, (400, 400)) 

        scene_dict = scenes[idx]
        target_mask = object_masks[scene_dict['target_id']]
        c_target_mask = general_utils.extract_target_crop2(target_mask, scene_image)

        for i, obstacle_id in enumerate(scene_dict['obstacle_ids']):
            obstacle_mask = object_masks[obstacle_id]
            c_obstacle_mask = general_utils.extract_target_crop2(obstacle_mask, scene_image)

            scene_image_ = cv2.imread(scene_dict['scene_image_dir'][i])
            scene_mask_ = cv2.imread(scene_dict['scene_mask_dir'][i], -1)
            scene_image_ = cv2.resize(scene_image_, (400, 400))

            fig, ax = plt.subplots(2, 2)
            ax[0][0].imshow(scene_image_)
            ax[0][0].set_title("Scene - Color")
            ax[0][0].axis("off")

            ax[0][1].imshow(scene_mask_)
            ax[0][1].set_title("Scene - Grayscale")
            ax[0][1].axis("off")

            ax[1][0].imshow(c_target_mask)
            ax[1][0].set_title("Target")
            ax[1][0].axis("off")

            ax[1][1].imshow(c_obstacle_mask)
            ax[1][1].set_title("Obstacle")
            ax[1][1].axis("off")

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    make_video_visualization()