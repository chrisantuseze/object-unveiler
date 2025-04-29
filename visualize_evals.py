import os
import shutil
import cv2
from trainer.memory import ReplayBuffer
from utils import general_utils

def get_obstacle_images():
    dataset_dir = "real_images/seg_data"

    transition_dirs = os.listdir(dataset_dir)

    for file_ in transition_dirs:
        if not file_.startswith("transition"):
            transition_dirs.remove(file_)
            
    memory = ReplayBuffer(dataset_dir)

    results = [
        {
            'target_id': 4,
            'gt': 4,
            'sre': 5,
            'clip': 0,
            'gpt-no': 2,
            'gpt-yes': 0,
            'image_id': '15'
        },
        {
            'target_id': 4,
            'gt': 2,
            'sre': 4,
            'clip': 0,
            'gpt-no': 2,
            'gpt-yes': 0,
            'image_id': '20'
        },
        {
            'target_id': 2,
            'gt': 1,
            'sre': 2,
            'clip': 0,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '03'
        },
        {
            'target_id': 5,
            'gt': 9,
            'sre': 5,
            'clip': 0,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '02'
        },
        {
            'target_id': 4,
            'gt': 5,
            'sre': 4,
            'clip': 0,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '13'
        },
        {
            'target_id': 3,
            'gt': 7,
            'sre': 3,
            'clip': 6,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '18'
        },
        {
            'target_id': 2,
            'gt': 2,
            'sre': 2,
            'clip': 2,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '05'
        },
        {
            'target_id': 5,
            'gt': 6,
            'sre': 5,
            'clip': 2,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '00'
        },
        {
            'target_id': 6,
            'gt': 6,
            'sre': 0,
            'clip': 6,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '12'
        },
        {
            'target_id': 1,
            'gt': 2,
            'sre': 1,
            'clip': 4,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '01'
        },
        {
            'target_id': 5,
            'gt': 2,
            'sre': 0,
            'clip': 0,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '09'
        },
        {
            'target_id': 3,
            'gt': 3,
            'sre': 3,
            'clip': 1,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '08'
        },
        {
            'target_id': 4,
            'gt': 2,
            'sre': 4,
            'clip': 0,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '07'
        },
        {
            'target_id': 3,
            'gt': 3,
            'sre': 3,
            'clip': 4,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '17'
        },
        {
            'target_id': 6,
            'gt': 4,
            'sre': 2,
            'clip': 2,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '14'
        },
        {
            'target_id': 1,
            'gt': 6,
            'sre': 0,
            'clip': 1,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '11'
        },
        {
            'target_id': 6,
            'gt': 4,
            'sre': 0,
            'clip': 3,
            'gpt-no': 0,
            'gpt-yes': 0,
            'image_id': '19'
        },
    ]

    for idx in range(len(results)):
        episode = results[idx]
        transition_dir = "transition_000" + episode['image_id']
        print("Transition Directory:", transition_dir)
        scene_image, scene_mask, target_mask, bboxes, target_id, object_masks = memory.load_seg_data(transition_dirs, idx)
        scene_image = cv2.resize(scene_image, (400, 400)) 

        print(len(object_masks), episode['target_id'], target_id)

        c_target_mask = general_utils.extract_target_crop2(object_masks[episode['target_id']], scene_image)
        sre_obstacle_mask = general_utils.extract_target_crop2(object_masks[episode['sre']], scene_image)
        clip_obstacle_mask = general_utils.extract_target_crop2(object_masks[episode['clip']], scene_image)
        gpt_no_obstacle_mask = general_utils.extract_target_crop2(object_masks[episode['gpt-no']], scene_image)
        gpt_yes_obstacle_mask = general_utils.extract_target_crop2(object_masks[episode['gpt-yes']], scene_image)

        save_dir = "real_images/results"

        folder_name = os.path.join(save_dir, transition_dir)
        if os.path.exists(folder_name):
            try:
                shutil.rmtree(folder_name)
            except OSError as e:
                pass
        os.makedirs(folder_name)

        cv2.imwrite(os.path.join(folder_name, 'scene_image.png'), scene_image)
        cv2.imwrite(os.path.join(folder_name, 'c_target_mask.png'), c_target_mask)
        cv2.imwrite(os.path.join(folder_name, 'sre_obstacle_mask.png'), sre_obstacle_mask)
        cv2.imwrite(os.path.join(folder_name, 'clip_obstacle_mask.png'), clip_obstacle_mask)
        cv2.imwrite(os.path.join(folder_name, 'gpt_no_obstacle_mask.png'), gpt_no_obstacle_mask)
        cv2.imwrite(os.path.join(folder_name, 'gpt_yes_obstacle_mask.png'), gpt_yes_obstacle_mask)


if __name__ == "__main__":
    get_obstacle_images()