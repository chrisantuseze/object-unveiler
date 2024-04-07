#!/usr/bin/env python3
# import zipfile

# with zipfile.ZipFile("pc-ou-dataset.zip", 'r') as zip_ref:
#     zip_ref.extractall("pc-ou-dataset")

import numpy as np

# Assuming 'arr' is your N x 4 array
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# Pad the array to 8 x 4
padded_arr = np.pad(arr, ((0, 8 - arr.shape[0]), (0, 0)), mode='constant')

print(padded_arr)