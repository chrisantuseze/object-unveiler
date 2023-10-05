#!/usr/bin/env python3
import sys
import zipfile
import pickle
import os

with zipfile.ZipFile("ppg-dataset-5k.zip", 'r') as zip_ref:
    zip_ref.extractall("ppg-dataset-5k")


# arr = [3,1,4,5,5,1,42,13,8,6]
# le = len(arr)

# f1 = le//3

# approx = f1 * 3
# split_index = int(0.9*approx)
# train_ids = arr[:split_index]
# val_ids = arr[split_index:]

# print(split_index, train_ids, val_ids)

# folder_path = "save/ppg-dataset"
# for filename in os.listdir(folder_path):
#     if os.path.isfile(os.path.join(folder_path, filename)):
# #         # Construct the new name for the file (modify this as needed)
#         try:
#             episode_data = pickle.load(open(os.path.join(folder_path, filename), 'rb'))
#         except:
#             os.remove(os.path.join(folder_path, filename))
#             print(filename)