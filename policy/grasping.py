import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def compute_grasping_point1(self, segmentation_mask):
    # # Find the contours in the segmentation mask
    # contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Initialize variables to store the center of mass and maximum contour area
    # max_contour_area = 0
    # center_of_mass = None

    # # Iterate through all the contours to find the largest one and its center of mass
    # for contour in contours:
    #     # Calculate the area of the current contour
    #     contour_area = cv2.contourArea(contour)

    #     # Update the maximum contour area and center of mass if a larger contour is found
    #     if contour_area > max_contour_area:
    #         max_contour_area = contour_area
    #         moments = cv2.moments(contour)
    #         if moments["m00"] != 0:
    #             center_of_mass = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

    # # Return the computed grasping point (center of mass)

    # # Print the computed grasping point
    # # print("Grasping Point:", center_of_mass)

    # # sample aperture uniformly
    # aperture = self.rng.uniform(self.aperture_limits[0], self.aperture_limits[1])

    # p1 = center_of_mass
    # discrete_theta = 1.18

    # action = np.zeros((4,))
    # action[0] = p1[0] * 0.0092
    # action[1] = p1[1] * 0.4198
    # action[2] = discrete_theta
    # action[3] = aperture

    # print("action_:", action)

    # return action
    pass


def compute_grasping_point_for_object(segmentation_mask, aperture_limits, rotations, rng):
    """Computes the grasping point of an object in a scene given a segmentation mask.

    Args:
        segmentation_mask: A binary mask of the scene.

    Returns:
        The grasping point of the object in the scene.
    """

    x_centers, y_centers = np.where(segmentation_mask >= 250)
    x_center = np.mean(x_centers) * 0.3
    y_center = np.mean(y_centers) * 0.3
    center_of_mass = (x_center, y_center)

    # sample aperture uniformly
    aperture = rng.uniform(aperture_limits[0], aperture_limits[1])

    p1 = center_of_mass

    theta = -np.arctan2(p1[1], p1[0])
    step_angle = 2 * np.pi / rotations
    discrete_theta = round(theta / step_angle) * step_angle

    if p1[0] <= 25 and p1[1] <= 25:
        discrete_theta -= np.pi * 0.25
    elif p1[0] >= 25 and p1[1] <= 25:
        discrete_theta -= np.pi * 0.75
    else:
        discrete_theta += np.pi
    
    action = np.zeros((4,))
    action[0] = p1[0]
    action[1] = p1[1]
    action[2] = discrete_theta
    action[3] = aperture

    print("action_:", action)

    return action

def calculate_iou(mask1, mask2):
    """Calculates the intersection over union (IoU) between two object masks."""
    intersection = torch.sum(mask1 * mask2)
    union = torch.sum(mask1) + torch.sum(mask2) - intersection
    return intersection / union

def calculate_overlap(mask1, mask2):
    return (mask1 & mask2).sum() / mask1.sum()

def add_edge(relationships, i, j):
    for edge in relationships:
        if edge[0] == j and edge[1] == i:
            return relationships

    relationships.append((i, j))
    return relationships

def extract_relationships(object_masks, overlap_threshold=0.2):
    relationships = []

    for i, mask_i in enumerate(object_masks):
        for j, mask_j in enumerate(object_masks):
            if i != j:  # Avoid self-comparison
                # overlap = self.calculate_overlap(mask_i, mask_j)
                # print(f"overlap ({i}, {j}): {overlap}")
                # if overlap > overlap_threshold:
                #     relationships.append((i, j))

                threshold_iou=0.0001

                mask_i = torch.Tensor.float(torch.tensor(mask_i))
                mask_j = torch.Tensor.float(torch.tensor(mask_j))

                iou = calculate_iou(mask_i, mask_j)
                # print(f"({i}, {j}) - iou:", iou)

                if iou >= threshold_iou:
                    add_edge(relationships, i, j)

    return relationships

def calculate_bounding_box(mask):
    # Calculate the bounding box of a segmentation mask
    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        bbox = cv2.boundingRect(contours[0])
        return bbox
    return None

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

def check_middle_placement(target_bbox, other_bboxes, distance_threshold=50):
    # Check if the target object is in the middle of other objects
    target_center = np.array([(target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2])
    for bbox in other_bboxes:
        other_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        distance = np.linalg.norm(target_center - other_center)

        print("distance:", distance)
        if distance < distance_threshold:
            return True
    return False

def build_graph( object_masks):
    nodes = []
    edges = []
    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes (objects) to the graph
    for idx, mask in enumerate(object_masks):
        G.add_node(idx, mask=mask)  # You can add more attributes to nodes here
        nodes.append(idx)

    overlap_threshold = 0.0001
    relationships = extract_relationships(object_masks, overlap_threshold=overlap_threshold)
    # print("relationships:", relationships)

    # Add edges (relationships) to the graph
    for rel in relationships:
        G.add_edge(rel[0], rel[1])
        edges.append((rel[0], rel[1]))

    # # Choose a target object (node) to focus on
    # target_object = 2

    # # Visualize the graph
    # pos = nx.spring_layout(G)  # Position nodes using spring layout
    # nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1000)
    # plt.title(f"Graph of Object Relationships (Target: {target_object})")
    # plt.show()

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

    '''
    get node with smallest path (items in list)
    '''
    optimal_node = key_with_least_unique_items(path_dic)
    return path_dic[optimal_node]

def key_with_least_unique_items(dictionary):
    min_unique_count = float('inf')
    key_with_min_unique = None
    
    for key, value in dictionary.items():
        unique_count = len(set(value))
        if unique_count < min_unique_count:
            min_unique_count = unique_count
            key_with_min_unique = key
    
    return key_with_min_unique

        