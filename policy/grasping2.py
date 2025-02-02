import cv2
import numpy as np
from scipy.ndimage import center_of_mass

from policy.grasping import get_distance

def find_obstacles_to_remove(target_index, segmentation_masks):
    if len(segmentation_masks) <= 3:
        return [target_index]
    
    distances_to_edge = get_distances_to_edge(segmentation_masks)

    min_dist = min(distances_to_edge)
    min_index = distances_to_edge.index(min_dist)
    if min_index == target_index:
        return [target_index]
    
    # Identify the target mask
    target_mask = segmentation_masks[target_index]

    # Find obstacles overlapping with the target object
    target_obj_distances = []
    for object_idx, mask in enumerate(segmentation_masks):
        dist = get_distance(center_of_mass(target_mask), center_of_mass(segmentation_masks[object_idx]))
        target_obj_distances.append(dist)
    
    normalized_periphery_dists = normalize(distances_to_edge)
    normalized_target_obj_dists = normalize(target_obj_distances)
    combined_distances = [d1 + d2 for d1, d2 in zip(normalized_periphery_dists, normalized_target_obj_dists)]
    
    sorted_indices = sorted(range(len(combined_distances)), key=lambda k: combined_distances[k])

    if sorted_indices[0] == target_index:
        sorted_indices = sorted_indices[1:]
    
    return sorted_indices

def normalize(distances):
    max_distance = max(distances)
    normalized_distances = [distance / max_distance for distance in distances]
    return normalized_distances

def get_distances_to_edge(segmentation_masks):
    distances_list = []

    # Iterate over each segmentation mask
    for i, mask in enumerate(segmentation_masks):
        min_distance = find_centroid_distance(mask)[0]
        # print(i, "-", min_distance)
        distances_list.append(min_distance)

    return distances_list

def find_centroid_distance(segmentation_mask):
    # Find unique labels in the segmentation mask
    unique_labels = np.unique(segmentation_mask)

    # Remove background label if present
    unique_labels = unique_labels[unique_labels != 0]

    # Initialize an array to store the minimum distances
    min_distances = []

    # Iterate through each object
    for obj_label in unique_labels:
        # Create a binary mask for the current object
        obj_mask = segmentation_mask == obj_label

        # Find the centroid of the object
        centroid = np.array(center_of_mass(obj_mask))

        # Compute distances to the four edges
        distances_to_edges = [
            centroid[0],                      # Distance to top edge
            segmentation_mask.shape[0] - centroid[0],  # Distance to bottom edge
            centroid[1],                      # Distance to left edge
            segmentation_mask.shape[1] - centroid[1]   # Distance to right edge
        ]

        # Append the minimum distance to the array
        min_distances.append(min(distances_to_edges))

    return min_distances