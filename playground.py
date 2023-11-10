#!/usr/bin/env python3
import sys
import zipfile
import pickle
import os

with zipfile.ZipFile("ou-dataset-consolidated.zip", 'r') as zip_ref:
    zip_ref.extractall("ou-dataset-consolidated")


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


# directory = "/Users/chrisantuseze/Research/robot-learning/ppg-datasets/ppg-dataset/"
# for root, dirs, files in os.walk(directory, topdown=False):
#     for dir in dirs:
#         dir_path = os.path.join(root, dir)
#         if not os.listdir(dir_path):
#             os.rmdir(dir_path)
#             print(f"Removed empty directory: {dir_path}")



# import numpy as np

# # Sample list of tuples, each containing three tensors
# tensor_tuple_list = [(np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])),
#                     (np.array([1, 2, 3, 4]), np.array([5, 6, 7]), np.array([8, 9, 10]))]

# # Get the size of the tensors in the first tuple to compare with others
# first_tuple_sizes = [len(tensor) for tensor in tensor_tuple_list[0]]

# # Initialize a list to store the indices of tuples with different sizes
# indices_of_different_sizes = []

# for i, tuple_elem in enumerate(tensor_tuple_list):
#     if not all(len(tensor) == size for tensor, size in zip(tuple_elem, first_tuple_sizes)):
#         indices_of_different_sizes.append(i)

# if not indices_of_different_sizes:
#     print("All tuples have tensors with equal sizes.")
# else:
#     print("Tuples with different-sized tensors:")
#     for idx in indices_of_different_sizes:
#         print(f"Tuple at index {idx}: {tensor_tuple_list[idx]}")

