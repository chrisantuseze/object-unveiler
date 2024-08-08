import os
import shutil


# dir = "/Users/chrisantuseze/Research/robot-learning/datasets/"
dir = ""
# Specify the path to the folder containing the files you want to rename

folder_path = "ppg-dataset/"
id = 667


folder_path = dir + folder_path
# Loop through the files in the folder
for filename in os.listdir(folder_path):
    old_name = os.path.join(folder_path, filename)
    
    arr = filename.split("_")
    new_filename = arr[0] + "_" + str(id).zfill(5)

    # Rename the file
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
    id += 1