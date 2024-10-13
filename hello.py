import os
import pickle
import shutil


# dir = "/Users/chrisantuseze/Research/robot-learning/datasets/"
dir = ""
# Specify the path to the folder containing the files you want to rename

folder_path = "save/ppg-dataset-extended/"
id = 0


folder_path = dir + folder_path
# Loop through the files in the folder
for i, filename in enumerate(os.listdir(folder_path)):
    old_name = os.path.join(folder_path, filename)
    
    arr = filename.split("_")
    new_filename = arr[0] + "_" + str(id).zfill(6)

    # Rename the file
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
    id += 1

print(id)

# folder_path = dir + folder_path
# # Loop through the files in the folder
# for i, filename in enumerate(os.listdir(folder_path)):
#     try:
#         dir = os.path.join(folder_path, filename)
#         episode_data = pickle.load(open(dir, 'rb'))
#     except Exception as e:
#         #logging.info(e, "- Failed episode:", episode_dir)
#         pass

#     data = episode_data[-1]
#     traj_data = data['traj_data'][:150]
#     if len(traj_data) == 0:
#         print(dir)