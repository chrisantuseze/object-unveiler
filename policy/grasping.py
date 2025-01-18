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

def get_object_centroid_old(segmentation_mask):
    # Find the contours in the segmentation mask
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the center of mass and maximum contour area
    max_contour_area = 0
    center_of_mass = None

    # Iterate through all the contours to find the largest one and its center of mass
    for contour in contours:
        # Calculate the area of the current contour
        contour_area = cv2.contourArea(contour)

        # Update the maximum contour area and center of mass if a larger contour is found
        if contour_area > max_contour_area:
            max_contour_area = contour_area
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                center_of_mass = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

    center_of_mass = list(center_of_mass)
    return center_of_mass

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

def calculate_iou(mask1, mask2):
    """Calculates the intersection over union (IoU) between two object masks."""
    intersection = torch.sum(mask1 * mask2)
    union = torch.sum(mask1) + torch.sum(mask2) - intersection
    return intersection / union

def add_edge(relationships, i, j):
    for edge in relationships:
        if edge[0] == j and edge[1] == i:
            return relationships

    relationships.append((i, j))
    return relationships

def extract_relationships(object_masks, threshold_iou=0.0001):
    relationships = []

    for i, mask_i in enumerate(object_masks):
        for j, mask_j in enumerate(object_masks):
            if i != j:  # Avoid self-comparison
                mask_i = torch.Tensor.float(mask_i)
                mask_j = torch.Tensor.float(mask_j)

                iou = calculate_iou(mask_i, mask_j)

                if iou >= threshold_iou:
                    add_edge(relationships, i, j)

    return relationships

def check_occlusion(target_bbox, other_bboxes, overlap_threshold=0.5):
    # Check if the target object is occluded by other objects
    for bbox in other_bboxes:
        overlap_area = calculate_overlap(target_bbox, bbox)
        if overlap_area / calculate_area(target_bbox) > overlap_threshold:
            return True
    return False

def calculate_overlap(bbox1, bbox2):
    # Calculate the overlap area between two bounding boxes
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return x_overlap * y_overlap

def calculate_area(bbox):
    # Calculate the area of a bounding box
    _, _, width, height = bbox
    return width * height

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
        
    logging.info("new target id:", id)
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

def is_target_neighbor(target, action, threshold):
    dist = get_distance(get_object_centroid(target), (action[0], action[1]))
    # print(dist)
    return dist < threshold

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

def evaluate_actions(actions, target_mask):
    new_actions = []
    for action in actions:
        if is_target_neighbor(target_mask, action, threshold=100):
            new_actions.append(action)
            print(action)

    return new_actions

def get_closest_neighbor(actions, target_mask):
    new_actions = []
    for action in actions:
        new_actions.append((get_distance(get_object_centroid(target_mask), (action[0], action[1])), action))

    return min(new_actions, key=lambda x: x[0])[1]

def get_grasped_object(processed_masks, action):

    for id, mask in enumerate(processed_masks):
        dist = get_distance(get_object_centroid(mask), (action[0], action[1]))
        print(dist)
        if dist < 250:
            print("grasped object:", id, dist)
            return id, mask

    return -1, None

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

def find_topmost_right_object(segmentation_masks, top_weight=2, right_weight=1):
    best_score = float('-inf')
    best_object = None
    
    height, width = segmentation_masks[0].shape
    
    for i, mask in enumerate(segmentation_masks):
        non_zero = np.nonzero(mask)
        if len(non_zero[0]) > 0:
            # Find the topmost and rightmost points
            top = np.min(non_zero[0])
            right = np.max(non_zero[1])
            
            # Calculate a score favoring top position more than right position
            # We invert top because lower y values are higher in the image
            score = (top_weight * (height - top) / height) + (right_weight * right / width)
            
            if score > best_score:
                best_score = score
                best_object = i

    print("Top-right object id is:", best_object)
    return best_object

def measure_singulation(target_id, masks: List, dilation_radius=5):
    seg_masks = deepcopy(masks)
    target_mask = seg_masks[target_id]
    # Dilate the target mask
    dilated_target = dilation(target_mask, disk(dilation_radius))

    seg_masks.pop(target_id)
    
    # Create a combined mask of all other objects
    other_objects = np.any(seg_masks, axis=0)
    
    # Calculate the overlap between dilated target and other objects
    overlap = np.logical_and(dilated_target, other_objects)
    
    # Calculate singulation score (1 - overlap ratio)
    overlap_ratio = np.sum(overlap) / np.sum(dilated_target)
    singulation_score = 1 - overlap_ratio
    
    return singulation_score

def measure_clutter_segmentation(segmentation_masks, k=3):
    # Calculate the centroid for each object's segmentation mask
    positions = []
    for mask in segmentation_masks:
        centroid = center_of_mass(mask)
        positions.append(centroid)

    n = len(positions)
    positions = np.array(positions)
    
    # k-NN regression to estimate x_hat_i and y_hat_i
    # nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(positions)
    nbrs = NearestNeighbors(n_neighbors=n-1, algorithm='auto').fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    
    # Remove the self-distance (distance to itself which is 0)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    x_hat = np.array([np.mean(positions[indices[i], 0]) for i in range(n)])
    y_hat = np.array([np.mean(positions[indices[i], 1]) for i in range(n)])
    
    # Calculate the average Euclidean distance
    euclidean_distances = np.sqrt((positions[:, 0] - x_hat)**2 + (positions[:, 1] - y_hat)**2)
    avg_distance = np.mean(euclidean_distances)
    
    # Calculate the clutter coefficient
    clutter_coefficient = -np.log(avg_distance)
    
    return clutter_coefficient