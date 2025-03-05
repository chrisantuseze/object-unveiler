from copy import deepcopy
import os
import math
from typing import List
import torch
import cv2
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import pdist, squareform

import numpy as np
from skimage.morphology import dilation, disk
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import utils.object_comparison as compare
import utils.logger as logging

def find_obstacles_to_remove(target_index, segmentation_masks):
    if len(segmentation_masks) <= 3:
        return [target_index if target_index >= 0 else 0]
    
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

def get_object_centroid(mask):
    # Calculate the centroid (center of mass)
    M = cv2.moments(mask)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        # Handle the case when the contour is empty
        cx, cy = 0, 0

    return [cx, cy]

def find_target(processed_masks, old_target_mask): 
    valid_objs = {}

    for id, mask in enumerate(processed_masks):
        if compare.object_compare(mask, old_target_mask):
            valid_objs[id] = (mask, get_object_centroid(mask))

    # no target is found in the scene
    if not valid_objs:
        print("No target is found in the scene")
        return -1, None

    # pick the object with the closest distance to the old target as the new target
    min_dist = float('inf')
    target_point = get_object_centroid(old_target_mask)
    # logging.info("target_point:", target_point)

    id = -1
    mask = None

    dist_threshold = 100 # TODO: this is subject to change
    for key, value in valid_objs.items():
        point = value[1]
        dist = get_distance(target_point, point)
        # logging.info("point:", point, "dist:", dist)

        if dist < min_dist and dist < dist_threshold: # update only when the object isn't too far away from the previous target
            min_dist = dist
            id = key
            mask = value[0]
        
    logging.info("New target id:", id)
    return id, mask

def get_distance(point1, point2):
    # Calculate the differences

    '''
    d = √((x2 - x1)^2 + (y2 - y1)^2)
    '''

    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]

    # Calculate the squared differences
    delta_x_squared = delta_x ** 2
    delta_y_squared = delta_y ** 2

    # Sum of squared differences
    sum_of_squared_diff = delta_x_squared + delta_y_squared

    # Calculate the distance
    distance = math.sqrt(sum_of_squared_diff)

    # print("Distance between the two points:", distance)
    return distance

def episode_status(grasping_status, is_target_grasped):
    if not is_target_grasped:
        return False
    
    return not any(item is False for item in grasping_status)

def get_target_id(target, processed_masks):
    indices = []
    for index, mask in enumerate(processed_masks):
        dist = get_distance(get_object_centroid(target), get_object_centroid(mask))
        if dist < 20:
            indices.append((index, dist))

    indices.sort(key=lambda x: x[1])
    if len(indices) > 0:
        return indices[0][0]
    
    return -1

@DeprecationWarning
def get_grasped_object(processed_masks, action):
    dists = []
    print(action[0], action[1], "\n")
    for id, mask in enumerate(processed_masks):
        cx, cy = get_object_centroid(mask)
        print(cx, cy)

        dist = get_distance([cx, cy], [action[0], action[1]])
        dists.append((id, dist))

    print(dists)
    dists.sort(key=lambda x: x[1])
    if len(dists) > 0:
        return dists[0][0]
    return -1

def is_target(target_mask, object_mask):
    dist = get_distance(get_object_centroid(target_mask), get_object_centroid(object_mask))
    print(dist)
    return dist < 150

def find_central_object(segmentation_masks):
    """
    Find the object with the centroid closest to the average centroid of all objects.

    Args:
        segmentation_masks (list): A list of binary segmentation masks, where 1 represents the object pixels, and 0 represents the background.

    Returns:
        int: The index of the object with the centroid closest to the average centroid of all objects.
    """
    num_objects = len(segmentation_masks)
    object_centroids = []

    # Calculate the centroid for each object's segmentation mask
    for mask in segmentation_masks:
        centroid = center_of_mass(mask)
        object_centroids.append(centroid)

    # Calculate the average centroid of all objects' centroids
    avg_centroid_x = sum(centroid[0] for centroid in object_centroids) / num_objects
    avg_centroid_y = sum(centroid[1] for centroid in object_centroids) / num_objects
    avg_centroid = (avg_centroid_x, avg_centroid_y)

    # Find the object with the centroid closest to the average centroid
    min_distance = float('inf')
    central_object_index = None

    for i, centroid in enumerate(object_centroids):
        distance = np.sqrt((centroid[0] - avg_centroid[0])**2 + (centroid[1] - avg_centroid[1])**2)
        if distance < min_distance:
            min_distance = distance
            central_object_index = i

    print("Central object id is:", central_object_index)
    return central_object_index

def compute_singulation(before_masks, after_masks):
    """
    Measures the degree of clutter reduction after an object is removed.

    Parameters:
        before_masks (list of np.array): List of binary masks for objects in the scene before removal.
        after_masks (list of np.array): List of binary masks for objects after removal.

    Returns:
        dict: Clutter reduction metrics including average centroid distance change and density change.
    """
    
    def get_centroids(masks):
        """Compute centroids of object masks."""
        centroids = []
        for mask in masks:
            y, x = np.where(mask > 0)
            if len(x) > 0 and len(y) > 0:
                centroids.append((np.mean(x), np.mean(y)))
        return np.array(centroids)
    
    def bounding_box_area(masks):
        """Compute the bounding box area that contains all objects."""
        all_y, all_x = np.where(np.any(np.stack(masks), axis=0))
        if len(all_x) > 0 and len(all_y) > 0:
            width = max(all_x) - min(all_x)
            height = max(all_y) - min(all_y)
            return width * height
        return 0

    # Get centroids
    before_centroids = get_centroids(before_masks)
    after_centroids = get_centroids(after_masks)
    
    # Compute average pairwise distance
    def avg_pairwise_distance(centroids):
        return np.mean(pdist(centroids)) if len(centroids) > 1 else 0

    avg_dist_before = avg_pairwise_distance(before_centroids)
    avg_dist_after = avg_pairwise_distance(after_centroids)

    # Compute bounding box area
    bbox_area_before = bounding_box_area(before_masks)
    bbox_area_after = bounding_box_area(after_masks)

    # Clutter metrics
    clutter_reduction = avg_dist_after - avg_dist_before
    density_reduction = (bbox_area_before - bbox_area_after) / bbox_area_before if bbox_area_before > 0 else 0

    # return {
    #     "clutter_reduction": clutter_reduction,  # Positive means better singulation
    #     "density_reduction": density_reduction,  # Positive means reduced density
    # }
    return clutter_reduction