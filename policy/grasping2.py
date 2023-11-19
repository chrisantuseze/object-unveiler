import numpy as np

def find_obstacles_to_remove(target_mask, all_masks, workspace_boundary = (100, 100)):
    # Calculate distances of each object to the periphery
    distances = calculate_distances_to_periphery(all_masks, workspace_boundary)

    # Sort objects based on distance
    sorted_objects = sort_objects_by_distance(distances)

    # Find obstacles overlapping with the target object
    obstacles_to_remove = []
    for object_idx in sorted_objects:
        print("object_idx:", object_idx)
        if mask_is_close_to_target(all_masks[object_idx], target_mask):
            obstacles_to_remove.append(object_idx)

    # Remove obstacles outside the heap
    # obstacles_to_remove = filter_obstacles_inside_heap(obstacles_to_remove, target_mask)

    return obstacles_to_remove

# Implement the helper functions as needed

def calculate_distances_to_periphery(all_masks, workspace_boundary):
    distances = []

    for mask in all_masks:
        centroid = calculate_centroid(mask)

        # Calculate the distance to the nearest point on the workspace boundary
        distance_to_periphery = calculate_distance_to_periphery(centroid, workspace_boundary)

        distances.append(distance_to_periphery)

    return distances

def calculate_distance_to_periphery(centroid, workspace_boundary):
    # Find the nearest point on the workspace boundary
    nearest_point = find_nearest_point_on_boundary(centroid, workspace_boundary)

    # Calculate Euclidean distance
    distance = np.linalg.norm(centroid - nearest_point)

    return distance

def find_nearest_point_on_boundary(point, boundary):
    # Find the nearest point on the boundary to the given point
    x, y = point
    boundary_x, boundary_y = boundary

    # Clip the coordinates to ensure they are within the workspace boundary
    x = np.clip(x, 0, boundary_x - 1)
    y = np.clip(y, 0, boundary_y - 1)

    # Find the nearest point
    nearest_x = min(max(0, x), boundary_x - 1)
    nearest_y = min(max(0, y), boundary_y - 1)

    return np.array([nearest_x, nearest_y])

def sort_objects_by_distance(distances):
    # Create a list of indices and sort them based on distances
    sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])

    return sorted_indices

def mask_overlaps_with_target(object_mask, target_mask):
    # Check if there is any overlap between the object mask and the target mask
    overlap = np.sum(np.logical_and(object_mask, target_mask)) > 0

    return overlap

def mask_is_close_to_target(object_mask, target_mask, distance_threshold=95):
    # Calculate the centroids of the object and target masks
    object_centroid = calculate_centroid(object_mask)
    target_centroid = calculate_centroid(target_mask)

    # Calculate the distance between centroids
    distance = np.linalg.norm(object_centroid - target_centroid)
    print("distance:", distance)

    # Check if the distance is below the threshold
    is_close = distance < distance_threshold

    return is_close

def calculate_centroid(mask):
    # Assuming mask is a binary image, find the indices of non-zero pixels
    object_pixels = np.transpose(np.nonzero(mask))

    # Calculate centroid of the object
    centroid = np.mean(object_pixels, axis=0)

    return centroid

def filter_obstacles_inside_heap(obstacles, target_mask):
    # Remove obstacles that are outside the target object's mask or heap
    # Return a filtered list of obstacles
    pass

def distance_to_edge(mask, workspace_size):
    rows, cols = np.where(mask)
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    top = min_row
    bottom = workspace_size - max_row - 1
    left = min_col 
    right = workspace_size - max_col - 1
    
    return min(top, bottom, left, right)

# def get_distances_to_edge(masks, workspace_size=100):
#     print("Method 2")
#     for i, mask in enumerate(masks):
#         print(i, "-", distance_to_edge(mask, workspace_size))


# def get_distances_to_edge(segmentation_masks, workspace_size=(100, 100)):
#     distances = []
#     print("Method 2")

#     # Iterate over each segmentation mask
#     for i, mask in enumerate(segmentation_masks):
#         # Find the indices of non-zero elements in the mask
#         non_zero_indices = np.nonzero(mask)

#         # Calculate the bounding box coordinates
#         min_x, min_y = np.min(non_zero_indices, axis=1)
#         max_x, max_y = np.max(non_zero_indices, axis=1)

#         # Calculate the distance from the object to each edge
#         distance_top = min(min_y, workspace_size[1] - max_y)
#         distance_bottom = min(max_y, workspace_size[1] - min_y)
#         distance_left = min(min_x, workspace_size[0] - max_x)
#         distance_right = min(max_x, workspace_size[0] - min_x)

#         # Choose the minimum distance to any edge
#         min_distance = min(distance_top, distance_bottom, distance_left, distance_right)

#         distances.append(min_distance)
#         print(i, "-", min_distance)

def get_distances_to_edge(segmentation_masks, workspace_size=(100, 100)):
    distances_list = []
    print(segmentation_masks[0].shape)
    print("Method 2")

    for i, mask in enumerate(segmentation_masks):
        # Find the coordinates of the pixels in the object
        object_pixels = np.transpose(np.nonzero(mask))

        # Calculate distances to the closest edges
        distances_top = object_pixels[:, 0]
        distances_bottom = workspace_size[0] - object_pixels[:, 0] - 1
        distances_left = object_pixels[:, 1]
        distances_right = workspace_size[1] - object_pixels[:, 1] - 1

        # Find the minimum distance for each pixel in the object
        min_distances = np.min([distances_top, distances_bottom, distances_left, distances_right], axis=0)

        # Find the overall minimum distance for the object
        min_distance = np.min(min_distances)

        distances_list.append(min_distance)
        print(i, "-", min_distance)