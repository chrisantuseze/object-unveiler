#!/usr/bin/env python3
import zipfile

with zipfile.ZipFile("new.zip", 'r') as zip_ref:
    zip_ref.extractall("new")


# import torch

# # A has shape (2, 12, 256) 
# A = torch.rand(2, 12, 256)  

# # B has shape (2, 12, 100, 100)
# B = torch.rand(2, 12, 100, 100)

# # Set some entries of B to 0 to simulate padding
# B[0, 5:8] = 0  
# B[1, 2:3] = 0

# # Assuming B is a mask indicating which entries are padded (containing zeros)
# padding_masks = (B.sum(dim=(2, 3)) == 0)
# print(padding_masks)
# print()

# # Expand the mask to match the shape of A
# padding_mask_expanded = padding_masks.unsqueeze(-1).expand_as(A)

# # Zero out the corresponding entries in A using the mask
# A = A.masked_fill_(padding_mask_expanded, 0)

# print(A)