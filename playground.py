#!/usr/bin/env python3
import sys
import zipfile
import pickle
import os

# with zipfile.ZipFile("ou-dataset-consolidated2.zip", 'r') as zip_ref:
#     zip_ref.extractall("ou-dataset-consolidated2")

# import torch
# import torch.nn as nn

# # Example usage of nn.Softmax(dim=-1)
# softmax = nn.Softmax(dim=-1)
# input_tensor = torch.randn(3, 4, 5)  # Example tensor with shape (3, 4, 5)
# print(input_tensor)
# output = softmax(input_tensor)
# print("================")
# print(output)

# import torch
# import torch.nn as nn

# # Example usage of nn.Softmax(dim=1)
# softmax = nn.Softmax(dim=1)
# input_tensor = torch.randn(3, 4, 5)  # Example tensor with shape (3, 4, 5)
# output = softmax(input_tensor)
# print("================")
# print(output)


# one_hot_encoded_vectors = [
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#     # Add more vectors as needed
# ]

# one_hot_encoded_vectors = [[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
#          [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]

# one_hot_encoded_vectors = [[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], 
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]

# # one_hot_encoded_vectors = one_hot_encoded_vectors[0]

# original_values = []

# for vector in one_hot_encoded_vectors:
#     index_of_one = vector.index(1)
#     original_values.append(index_of_one)

# print("Original values:", original_values)

# import torch
# def one_hot_encoding_tensor(tensor, max_value):
#         # Create an identity matrix with size (max_value + 1)
#         identity_matrix = torch.eye(max_value + 1)

#         # Use tensor indexing to get the one-hot encoded representation
#         one_hot_encoded_tensor = identity_matrix[tensor]

#         return one_hot_encoded_tensor

# # Example usage:
# input_tensor = torch.tensor([2, 0, 5, 11])
# max_value = 13
# result_tensor = one_hot_encoding_tensor(input_tensor, max_value)

# print(result_tensor)


# import torch

# def decode_one_hot_tensor(one_hot_tensor):
#     # Use torch.argmax to find the index of the maximum value along the second axis
#     decoded_tensor = torch.argmax(one_hot_tensor, dim=1)

#     # Convert the tensor to a Python list
#     decoded_list = decoded_tensor.tolist()

#     return decoded_list

# # Example usage:
# # one_hot_encoded_tensor = torch.tensor([
# #     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #     [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
# #     [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
# #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]
# # ])

# d = torch.tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
#          [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

#         [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]],

#         [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
#          [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

#         [[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#          [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]]])

# decoded_list = decode_one_hot_tensor(d)

# print(decoded_list)

import torch

# Assuming your input tensor is a torch tensor
# input_tensor = torch.tensor([2, 0, 5, 11])

input_tensor = torch.tensor([[ 0, 10, 11,  2, 12],
        [ 0,  6,  1, 10,  9]])

# Define the maximum possible value
max_value = 14

# Use torch.nn.functional.one_hot to perform one-hot encoding
one_hot_encoded = torch.tensor(torch.nn.functional.one_hot(input_tensor, num_classes=max_value), dtype=torch.float, requires_grad=True)

print(one_hot_encoded)


# import torch

# Assuming you have a one-hot encoded tensor
# one_hot_encoded = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

# o = torch.tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]])


for item in one_hot_encoded:
    # Use torch.argmax to find the indices of the maximum values along the specified axis
    original_indices = torch.argmax(item, dim=1)

    # Convert the tensor back to a Python list
    original_list = original_indices.tolist()

    print(original_list)
