import os
import pickle
import shutil

import numpy as np
import cv2

from trainer.memory import ReplayBuffer


dir = ""
# Specify the path to the folder containing the files you want to rename

# folder_path = "save/pc-ou-dataset/"
# folder_path = "/home/e_chrisantus/Projects/grasping_in_clutter/using-pointcloud/single-target-grasping/target-ppg-using-9-12-objects/ppg-ou-dataset-9-12"
# folder_path = "/home/e_chrisantus/Projects/grasping_in_clutter/object-unveiler/save/pc-ou-dataset/"
# id = 5497


# folder_path = dir + folder_path
# # Loop through the files in the folder
# for i, filename in enumerate(os.listdir(folder_path)):
#     old_name = os.path.join(folder_path, filename)
    
#     arr = filename.split("_")
#     new_filename = arr[0] + "_" + str(id).zfill(5)

#     # Rename the file
#     os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
#     id += 1

# print(id)

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

def main1():
    dataset_dir = "save/ppg-dataset"
    transition_dirs = os.listdir(dataset_dir)
    
    for file_ in transition_dirs:
        if not file_.startswith("transition"):
            transition_dirs.remove(file_)

    print(len(transition_dirs))

    mem = ReplayBuffer(save_dir="save/ppg-dataset-clean")

    count = 0
    for transition in transition_dirs:
        path = os.path.join(dataset_dir, transition)
        if os.listdir(path):
            try:
                state = cv2.imread(os.path.join(path, 'heightmap.exr'), -1)
                action = pickle.load(open(os.path.join(path, 'action'), 'rb'))

                trans = {'state': state, 'action': action}

                mem.store(trans)
            except Exception as e:
                print(e)
            continue

        count += 1
        

    print(count)

    

def main():
    # dataset_dir = "/home/e_chrisantus/Projects/grasping_in_clutter/using-pointcloud/old-episodic-grasping/pc-ou-dataset2"
    dataset_dir = "save/pc-ou-dataset"
    transition_dirs = os.listdir(dataset_dir)
    for file_ in transition_dirs:
        if not file_.startswith("episode"):
            transition_dirs.remove(file_)

    # episode_data = pickle.load(open(os.path.join(dataset_dir, transition_dirs[0]), 'rb'))
    # data = episode_data[0]

    new_dir = "save/pc-ou-dataset1"
    memory = ReplayBuffer(new_dir)

    count = 0
    for episode in transition_dirs:
        # try:
        #     scene_mask, c_target_mask, c_object_masks, objects_to_remove, bboxes = load_episode(dataset_dir, episode)
        # except:
        #     count += 1

        try:
            with open(os.path.join(dataset_dir, episode), 'rb') as f:
                episode_data = pickle.load(f)
                memory.store_episode(episode_data)
                # first_bytes = f.read(100)

            # print(first_bytes)

        except Exception as e:
            print(f"Error loading {episode}: {e}")
            count += 1

    #     # if len(c_object_masks) == 0:
    #     #     print(f"Deleting {episode}...")
    #     #     episode_path = os.path.join(dataset_dir, episode)
    #     #     os.remove(episode_path)  # Delete the file

    #     # if len(objects_to_remove) == 1 and objects_to_remove[0] == -1:
    #     #     print(f"Deleting {episode}...", objects_to_remove[0])
    #     #     count += 1
    #     #     episode_path = os.path.join(dataset_dir, episode)
    #     #     os.remove(episode_path)  # Delete the file

    #     # if 10 in objects_to_remove:
    #     #     print(f"Deleting {episode}...", objects_to_remove)
    #     #     episode_path = os.path.join(dataset_dir, episode)
    #     #     os.remove(episode_path)  # Delete the file
    #     #     count += 1

    #     for obj_mask in c_object_masks:
    #         if isinstance(obj_mask, list):
    #             print(f"Deleting {episode}...")
    #             count += 1
    #             episode_path = os.path.join(dataset_dir, episode)
    #             os.remove(episode_path)  # Delete the file
    #             break

    print("Total count:", count)

main1()