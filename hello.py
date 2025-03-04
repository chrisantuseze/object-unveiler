import os
import pickle
import shutil

import numpy as np


# # dir = "/Users/chrisantuseze/Research/robot-learning/datasets/"
dir = ""
# Specify the path to the folder containing the files you want to rename

# folder_path = "save/pc-ou-dataset-new/"
# folder_path = "/home/e_chrisantus/Projects/grasping_in_clutter/using-pointcloud/single-target-grasping/target-ppg-using-9-12-objects/ppg-ou-dataset-9-12"
folder_path = "/home/e_chrisantus/Projects/grasping_in_clutter/object-unveiler/save/pc-ou-dataset2/"
id = 0


folder_path = dir + folder_path
# Loop through the files in the folder
for i, filename in enumerate(os.listdir(folder_path)):
    old_name = os.path.join(folder_path, filename)
    
    arr = filename.split("_")
    new_filename = arr[0] + "_" + str(id).zfill(5)

    # Rename the file
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
    id += 1

print(id)

def load_episode(dataset_dir, episode):
    # Ensure there's a valid image. If there's none, search through the timesteps
    try:
        # Load episode data
        episode_data = pickle.load(open(os.path.join(dataset_dir, episode), 'rb'))

        data = episode_data[0]
        c_target_mask = data['c_target_mask']
        c_object_masks = data['c_object_masks']
        scene_mask = data['scene_mask']
        objects_to_remove = data['objects_to_remove']
        bboxes = data['bboxes']
        return scene_mask, c_target_mask, c_object_masks, objects_to_remove, bboxes

    except Exception as e:
        print(f"{e} - Failed episode: {episode}")

    return None

def main():
    dataset_dir = "save/pc-ou-dataset-new"
    transition_dirs = os.listdir(dataset_dir)
    for file_ in transition_dirs:
        if not file_.startswith("episode"):
            transition_dirs.remove(file_)

    for episode in transition_dirs:
        scene_mask, c_target_mask, c_object_masks, objects_to_remove, bboxes = load_episode(dataset_dir, episode)

        if len(c_object_masks) == 0:
            print(f"Deleting {episode}...")
            episode_path = os.path.join(dataset_dir, episode)
            os.remove(episode_path)  # Delete the file

# main()