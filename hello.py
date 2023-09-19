import os

# Specify the path to the folder containing the files you want to rename
folder_path = "save/ppg-dataset-1/"

id = 845
# Loop through the files in the folder
for filename in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, filename)):
        # Construct the new name for the file (modify this as needed)

        arr = filename.split("_")
        new_filename = arr[0] + "_" + str(id).zfill(5)

        # Rename the file
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
        id += 1
