import os
import math
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils.object_comparison as compare
import utils.general_utils as general_utils
import utils.logger as logging

def get_object_centroid(segmentation_mask):
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

def get_grasping_angle(segmentation_masks, target_id):
    centroids = []

    for id, mask in enumerate(segmentation_masks):
        if id == target_id:
            continue
        centroids.append(get_object_centroid(mask))

    target_centroid = get_object_centroid(segmentation_masks[target_id])
    
    # check if > 0.5 segmentations masks have x & y > target x & y. If so, tilt angle 315 degs anticlockwise
    objects_count = {
        'A': 0,
        'B': 0,
        'C': 0,
        'D': 0
    }
    for centroid in centroids:
        if centroid[0] > target_centroid[0] and centroid[1] > target_centroid[1]:
            objects_count['A'] += 1

        elif centroid[0] < target_centroid[0] and centroid[1] < target_centroid[1]:
            objects_count['C'] += 1

        elif centroid[0] < target_centroid[0] and centroid[1] > target_centroid[1]:
            objects_count['B'] += 1

        elif centroid[0] > target_centroid[0] and centroid[1] < target_centroid[1]:
            objects_count['D'] += 1

    logging.info("objects_count:", objects_count)

    percentage = 0.75
    if objects_count['A'] >= int(percentage * len(segmentation_masks)):
        return 315
    
    if objects_count['C'] >= int(percentage * len(segmentation_masks)):
        return 135
    
    if objects_count['B'] >= int(percentage * len(segmentation_masks)):
        return 225
    
    if objects_count['D'] >= int(percentage * len(segmentation_masks)):
        return 45
    
    return 0


def compute_grasping_point_for_object1(segmentation_masks, object_id, aperture_limits, rotations, rng):
    # Return the computed grasping point (center of mass)
    center_of_mass = get_object_centroid(segmentation_masks[object_id])

    # Print the computed grasping point
    # logging.info("Grasping Point:", center_of_mass)

    # sample aperture uniformly
    aperture = rng.uniform(aperture_limits[0], aperture_limits[1])

    p1 = center_of_mass 
    
    theta = -np.arctan2(p1[1], p1[0])
    step_angle = 2 * np.pi / rotations
    discrete_theta = round(theta / step_angle) * step_angle

    discrete_theta = general_utils.rad_to_deg(discrete_theta)

    p1[0] *= 0.225 #0.22 #0.175 #0.175
    p1[1] *= 0.215 #0.21 #0.19 #0.18

    grasping_angle = get_grasping_angle(segmentation_masks, object_id)
    if grasping_angle > 0:
        discrete_theta = grasping_angle

    else:

        if p1[0] <= 35 and p1[1] <= 35:
            # discrete_theta -= np.pi * 0.25      # tilt hand to face 315 degs left (anti-clockwise)
            pass

        elif p1[0] < 35 and p1[1] < 60:
            # discrete_theta += np.pi * 0.225      # tilt hand to face ~45 degs left (anti-clockwise) -67.5 + x = -27; x = 40.5
            discrete_theta += 40.5

        elif p1[0] > 35 and p1[1] < 35:
            # discrete_theta -= np.pi * 0.75      # tilt hand to face 225 degs left (anti-clockwise) -45 - x = -180; x = 135
            discrete_theta -= 135

        else:
            p1[0] *= 1.33 #(0.28/0.21)
            p1[1] *= 1.33

            # discrete_theta += np.pi             # tilt hand to face 135 degs left (anti-clockwise) -22.5 + x = 157.5; x = 180
            discrete_theta += 180
        
    # logging.info("discrete_theta:", discrete_theta)

    discrete_theta = general_utils.deg_to_rad(discrete_theta)

    action = np.zeros((4,))
    action[0] = p1[0]
    action[1] = p1[1]
    action[2] = discrete_theta
    action[3] = aperture

    # print("action_:", action)

    return action

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

def build_graph(object_masks):
    nodes = []
    edges = []

    # Add nodes (objects) to the graph
    for idx, mask in enumerate(object_masks):
        nodes.append(idx)

    relationships = extract_relationships(object_masks)
    # print("relationships:", relationships)

    # Add edges (relationships) to the graph
    for rel in relationships:
        edges.append((rel[0], rel[1]))

    return nodes, edges

def get_optimal_target_path(representations, target_id):
    '''
    representations -> [(0,2), (0,3), (0,5), (3,1), (5,1), (2,3), (5,4)]
    target_id -> 1
    '''
    adj_nodes = []
    for rep in representations:
        if rep[0] == target_id:
            adj_nodes.append(rep[1])
        elif rep[1] == target_id:
            adj_nodes.append(rep[0])

    '''
    add possible paths to dictionary. Nodes are 3 and 5 -> 
    (3,1) -> (0,3), (2,3)
    (5,1) -> (0,5), (5,4)
    '''
    path_dic = {}
    for rep in representations:
        for node in adj_nodes:
            if node in rep:
                if node not in path_dic.keys():
                    path_dic[node] = []
                path_dic[node].append(rep)

    # print("path_dic:", path_dic)
    if len(path_dic) < 1:
        return []
    
    '''
    get node with smallest path (items in list)
    Picks the first one (3, 1) since both have same number of unique items
    '''
    optimal_node_id = key_with_least_unique_items(path_dic)
    optimal_nodes = set([node for tuple_ in path_dic[optimal_node_id] for node in tuple_])
    optimal_nodes = list(optimal_nodes) # returns -> [0,2,3]

    # order based on the node with the least number of edges
    return sort_by_num_edges(representations, optimal_nodes)

def key_with_least_unique_items(dictionary):
    min_unique_count = float('inf')
    key_with_min_unique = None
    
    for key, value in dictionary.items():
        unique_count = len(set(value))
        if unique_count < min_unique_count:
            min_unique_count = unique_count
            key_with_min_unique = key
    
    return key_with_min_unique

def sort_by_num_edges(representations, optimal_nodes):
    edges_as_list = [node for tuple_ in representations for node in tuple_]
    # print("edges_as_list:", edges_as_list)

    # Create a sorting key function that uses the count of each item in edges_as_list
    def sorting_key(item):
        return edges_as_list.count(item)

    # Sort list optimal_nodes using the custom sorting key
    return sorted(optimal_nodes, key=sorting_key)

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

def get_obstacle_id(raw_masks, target_id, prev_node_id):
    _, edges = build_graph(raw_masks)
    if len(edges) > 0:
        optimal_nodes = get_optimal_target_path(edges, target_id)
        # print("optimal_nodes:", optimal_nodes)

        if len(optimal_nodes) > 0:
            node_id = optimal_nodes[0]
            if prev_node_id == node_id and len(optimal_nodes) > 1:
                node_id = optimal_nodes[1]
                
        else: # if target is not occluded
            node_id = target_id

    else: # if target is not occluded
        print("Object is not occluded")
        node_id = target_id

    # print("node_id:", node_id)
    prev_node_id = node_id

    return node_id, prev_node_id