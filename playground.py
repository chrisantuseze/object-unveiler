#!/usr/bin/env python3
import sys
import zipfile

with zipfile.ZipFile("ppg-dataset-consolidated.zip", 'r') as zip_ref:
    zip_ref.extractall("ppg-dataset")


# arr = [3,1,4,5,5,1,42,13,8,6]
# le = len(arr)

# f1 = le//3

# approx = f1 * 3
# split_index = int(0.9*approx)
# train_ids = arr[:split_index]
# val_ids = arr[split_index:]

# print(split_index, train_ids, val_ids)