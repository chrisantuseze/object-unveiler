import os
import pickle
import shutil

import numpy as np


# # dir = "/Users/chrisantuseze/Research/robot-learning/datasets/"
dir = ""
# Specify the path to the folder containing the files you want to rename

# folder_path = "save/pc-ou-dataset-new/"
# folder_path = "/home/e_chrisantus/Projects/grasping_in_clutter/using-pointcloud/single-target-grasping/target-ppg-using-9-12-objects/ppg-ou-dataset-9-12"
folder_path = "/home/e_chrisantus/Projects/grasping_in_clutter/object-unveiler/save/pc-ou-dataset-new/"
id = 2330


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
        data = episode_data[-1]

        # Extract heightmap and mask
        heightmap = data['state']
        c_target_mask = None #data['c_target_mask']

        # Initialize storage for valid timesteps
        images, joint_pos = [], []

        # Iterate through trajectory data and validate each timestep
        traj_data = data['traj_data']
        for traj in traj_data:
            if len(traj[1]['color']) > 1:
                joint_pos.append(traj[0])  # Append joint positions
                images.append(traj[1])  # Append the valid image

        # If no valid images are found, raise an exception
        if not images:
            raise ValueError("No valid images found in the episode.")

        episode_len = len(traj_data)
        return images, joint_pos, heightmap, c_target_mask

    except Exception as e:
        print(f"{e} - Failed episode: {episode}")

    return None

def main():
    dataset_dir = "save/ppg-dataset2"
    transition_dirs = os.listdir(dataset_dir)
    for file_ in transition_dirs:
        if not file_.startswith("episode"):
            transition_dirs.remove(file_)

    for id in transition_dirs:
        print("id", id)

        images, joint_pos, heightmap, c_target_mask = load_episode(dataset_dir, id)
        images = images[0]
        color1 = images['color'][0]
        color2 = images['color'][1]

        print(color1)
        print(color2)

# main()