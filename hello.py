# A = [3, 1, 2, 2, 1, 4, 5]
# B = [1, 2, 2, 3, 3, 3, 4, 4, 5]

# # Create a sorting key function that uses the count of each item in B
# def sorting_key(item):
#     return B.count(item)

# # Sort list A using the custom sorting key
# A_sorted = sorted(A, key=sorting_key)

# print("Sorted A based on occurrences in B:", A_sorted)

# list_ = [3, 2, 5, 7]

# for i, item in enumerate(list_):
#     print(i, item)

import numpy as np

# Assuming you have your original data as 'original_data' with shape (1, 224, 224)
original_data = np.random.rand(1, 2, 2)
data = np.array([original_data, original_data])
s, c, h, w = data.shape
# print(data.shape, data)

# Number of copies you want (in this case, 4)
num_copies = 4

padding_needed = num_copies - s

# Create a tiled version of the original data
# padded_data = np.tile(data, (num_copies, 1, 1, 1))
padded_data = np.pad(data, ((padding_needed, 0), (0, 0), (0, 0), (0, 0)), mode='constant')


# 'padded_data' now contains 4 copies of the original data along the first dimension.
# Its shape will be (4, 224, 224).
print(padded_data.shape, padded_data)