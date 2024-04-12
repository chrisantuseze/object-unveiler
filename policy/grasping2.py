import numpy as np
from scipy import ndimage

def find_obstacles_to_remove(target_mask, all_masks):
    # Calculate distances of each object to the periphery
    # distances = calculate_distances_to_periphery(all_masks, workspace_boundary)

    distances = get_distances_to_edge(all_masks)

    # Sort objects based on distance
    sorted_objects = sort_objects_by_distance(distances)

    # Find obstacles overlapping with the target object
    obstacles_to_remove = []
    for object_idx in sorted_objects:
        if mask_is_close_to_target(object_idx, all_masks[object_idx], target_mask):
            obstacles_to_remove.append(object_idx)

    return obstacles_to_remove

def get_target_objects_distance(target_mask, all_masks):
    weights = []

    for object_mask in all_masks:
        weights.append(get_target_object_distance(object_mask, target_mask))

    # Find the minimum and maximum values in the tensor
    min_value = np.min(weights)
    max_value = np.max(weights)

    # Invert the list
    inverted_weights = [min_value + max_value - x for x in weights]

    inverted_weights -= min_value
    
    epsilon = 1e-6  # Small positive value to avoid division by zero
    normalized_weights = inverted_weights / (max_value - min_value + epsilon)

    # Step 4: Add a small positive value to ensure the values are above 0
    normalized_weights += epsilon

    return np.array(normalized_weights)

# Implement the helper functions as needed

def sort_objects_by_distance(distances):
    # Create a list of indices and sort them based on distances
    sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])

    return sorted_indices

def mask_overlaps_with_target(object_mask, target_mask):
    # Check if there is any overlap between the object mask and the target mask
    overlap = np.sum(np.logical_and(object_mask, target_mask)) > 0

    return overlap

def get_target_object_distance(object_mask, target_mask):
    # Calculate the centroids of the object and target masks
    object_centroid = calculate_centroid(object_mask)
    target_centroid = calculate_centroid(target_mask)

    # Calculate the distance between centroids
    distance = np.linalg.norm(object_centroid - target_centroid)
    return distance

def mask_is_close_to_target(object_idx, object_mask, target_mask, distance_threshold=95):
    distance = get_target_object_distance(object_mask, target_mask)
    # print("object_idx:", object_idx, "distance:", distance)

    # Check if the distance is below the threshold
    is_close = distance < distance_threshold

    return is_close

def calculate_centroid(mask):
    # Assuming mask is a binary image, find the indices of non-zero pixels
    object_pixels = np.transpose(np.nonzero(mask))

    # Calculate centroid of the object
    centroid = np.mean(object_pixels, axis=0)

    return centroid

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
        centroid = np.array(ndimage.measurements.center_of_mass(obj_mask))

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