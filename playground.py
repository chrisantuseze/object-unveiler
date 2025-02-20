#!/usr/bin/env python3
# import zipfile

with zipfile.ZipFile("pc-ou-dataset2.zip", 'r') as zip_ref:
    zip_ref.extractall("pc-ou-dataset2")

# import os
# import shutil

# # Set your folder path and number of copies
# folder_path = "/home/e_chrisantus/Projects/grasping_in_clutter/object-unveiler/save/pc-ou-dataset2"  # Change this to your actual folder
# N = 3  # Change this to the number of times you want to duplicate each file

# # Loop through all files in the folder
# for filename in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, filename)
    
#     # Make sure it's a file (not a folder)
#     if os.path.isfile(file_path):
#         file_name, file_ext = os.path.splitext(filename)
        
#         # Create N copies
#         for i in range(1, N + 1):
#             new_file = os.path.join(folder_path, f"{file_name}_copy{i}{file_ext}")
#             shutil.copy(file_path, new_file)

# print("Files duplicated successfully!")
