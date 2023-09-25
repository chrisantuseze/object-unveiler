import os
import shutil


dir = "/Users/chrisantuseze/Research/rl/unveiler-datasets/combined/"
# Specify the path to the folder containing the files you want to rename

# folder_path = "ppg-dataset-2/"
# id = 845

# folder_path = "ppg-dataset-edgar/"
# id = 1616

# folder_path = "ppg-dataset-mac/"
# id = 2441

# folder_path = "ppg-dataset-regan/"
# id = 3031


# folder_path = dir + folder_path
# # Loop through the files in the folder
# for filename in os.listdir(folder_path):
#     if os.path.isfile(os.path.join(folder_path, filename)):
#         # Construct the new name for the file (modify this as needed)

#         arr = filename.split("_")
#         new_filename = arr[0] + "_" + str(id).zfill(5)

#         # Rename the file
#         os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
#         id += 1


# Loop through all subdirectories in the source directory

source_dir = dir
destination_dir = "/Users/chrisantuseze/Research/rl/unveiler-datasets/consolidated/ppg-dataset"

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)


for root, _, files in os.walk(source_dir):
    for file in files:
        # Get the full path of the source file
        source_file = os.path.join(root, file)
        
        # Get the destination path for the file in the destination directory
        destination_file = os.path.join(destination_dir, file)
        
        try:
            # Move the file to the destination directory
            shutil.move(source_file, destination_file)
            print(f"Moved: {source_file} -> {destination_file}")
        except Exception as e:
            print(f"Error moving {source_file}: {str(e)}")
