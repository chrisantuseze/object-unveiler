import cv2
import numpy as np
from scipy.ndimage import center_of_mass

from policy.grasping import get_distance, get_object_centroid

# def find_obstacles_to_remove(target_index, segmentation_masks):
#     if len(segmentation_masks) <= 3:
#         return [target_index]
    
#     distances_to_edge = get_distances_to_edge(segmentation_masks)
    
#     sorted_indices = base_find_obstacles_to_remove(target_index, segmentation_masks, distances_to_edge)
#     best_obstacle_id = sorted_indices[0]

#     print("sorted_indices:", sorted_indices)

#     sorted_indices = base_find_obstacles_to_remove(best_obstacle_id, segmentation_masks, distances_to_edge)
#     return sorted_indices

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

def find_obstacles_to_remove1(target_index, segmentation_masks):
    # Convert masks to binary images
    binary_masks = [mask > 0 for mask in segmentation_masks]

    # Calculate distance transform for the periphery
    periphery_mask = np.zeros_like(binary_masks[0])
    periphery_mask[:, 0] = 1  # Left edge
    periphery_mask[:, -1] = 1  # Right edge
    periphery_mask[0, :] = 1  # Top edge
    periphery_mask[-1, :] = 1  # Bottom edge
    periphery_distance_map = cv2.distanceTransform(np.uint8(periphery_mask), cv2.DIST_L2, 5)

    # Check if the target is the closest object to the periphery
    objects_periphery_dist = []
    for obstacle_index, mask in enumerate(segmentation_masks):
        obstacle_mask = binary_masks[obstacle_index]
        min_distance = np.min(obstacle_mask.astype(np.float32) * periphery_distance_map)
        objects_periphery_dist.append((obstacle_index, min_distance, mask))

    # Sort obstacles based on distance to periphery
    objects_periphery_dist.sort(key=lambda x: x[1])
    
    if objects_periphery_dist[0][0] == target_index:
        return [target_index]

    # Identify the target mask
    target_mask = segmentation_masks[target_index]

    # Find obstacles overlapping the target mask
    overlapping_obstacles = []
    for i, obstacle_mask in enumerate(segmentation_masks):
        dist = get_distance(get_object_centroid(target_mask), get_object_centroid(obstacle_mask))
        print(dist)
        if dist < 100:
            overlapping_obstacles.append(i)

    # Find closest overlapping obstacles to the periphery
    closest_overlapping_obstacles = []
    for obstacle_index in overlapping_obstacles:
        obstacle_mask = binary_masks[obstacle_index]
        min_distance = np.min(obstacle_mask.astype(np.float32) * periphery_distance_map)
        closest_overlapping_obstacles.append((obstacle_index, min_distance))

    if len(closest_overlapping_obstacles) == 0:
        return [target_index]
    
    print(closest_overlapping_obstacles)
    
    # Sort overlapping obstacles based on distance to periphery
    closest_overlapping_obstacles.sort(key=lambda x: x[1])
    print(closest_overlapping_obstacles)
    
    return [id for id, dist in closest_overlapping_obstacles]
    
def find_obstacles_to_remove2(target_index, segmentation_masks):
    # Convert masks to binary images
    binary_masks = [mask > 0 for mask in segmentation_masks]

    # Calculate distance transforms
    distance_maps = [cv2.distanceTransform(np.uint8(mask), cv2.DIST_L2, 5) for mask in binary_masks]

    # Find closest obstacles to the target mask
    target_distance_map = distance_maps[target_index]
    closest_obstacles = []
    for i, distance_map in enumerate(distance_maps):
        if i != target_index:
            min_distance = np.min(target_distance_map + distance_map)
            closest_obstacles.append((i, min_distance))
    closest_obstacles.sort(key=lambda x: x[1])

    # Identify the periphery of the scene
    periphery_mask = np.zeros_like(binary_masks[0])
    periphery_mask[:, 0] = 1  # Left edge
    periphery_mask[:, -1] = 1  # Right edge
    periphery_mask[0, :] = 1  # Top edge
    periphery_mask[-1, :] = 1  # Bottom edge
    periphery_distance_map = cv2.distanceTransform(np.uint8(periphery_mask), cv2.DIST_L2, 5)

    # Find obstacles closest to the periphery
    closest_to_periphery = []
    for i, distance_map in enumerate(distance_maps):
        min_distance = np.min(distance_map + periphery_distance_map)
        closest_to_periphery.append((i, min_distance))
    closest_to_periphery.sort(key=lambda x: x[1])

    if len(closest_to_periphery) == 0:
        return [target_index]
    
    return [id for id, dist in closest_to_periphery]