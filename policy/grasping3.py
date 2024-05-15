import cv2
import numpy as np

from policy.grasping import get_distance, get_object_centroid

def find_obstacles_to_remove(target_index, segmentation_masks):
    # Convert masks to binary images
    binary_masks = [mask > 0 for mask in segmentation_masks]

    # Calculate distance transform for the periphery
    periphery_mask = np.zeros_like(binary_masks[0])
    periphery_mask[:, 0] = 1  # Left edge
    periphery_mask[:, -1] = 1  # Right edge
    periphery_mask[0, :] = 1  # Top edge
    periphery_mask[-1, :] = 1  # Bottom edge
    periphery_distance_map = cv2.distanceTransform(np.uint8(periphery_mask), cv2.DIST_L2, 5)

    # Find closest overlapping obstacles to the periphery
    objects_periphery_dist = []
    for obstacle_index, mask in enumerate(segmentation_masks):
        obstacle_mask = binary_masks[obstacle_index]
        min_distance = np.min(obstacle_mask.astype(np.float32) * periphery_distance_map)
        objects_periphery_dist.append((obstacle_index, min_distance, mask))

    # Sort overlapping obstacles based on distance to periphery
    objects_periphery_dist.sort(key=lambda x: x[1])
    if objects_periphery_dist[0][0] == target_index:
        return [target_index]
    
    # Identify the target mask
    target_mask = segmentation_masks[target_index]

    # Find obstacles overlapping the target mask
    closest_overlapping_obstacles = []
    for i, dist, obstacle_mask in objects_periphery_dist:
        if i != target_index:
            dist = get_distance(get_object_centroid(target_mask), get_object_centroid(obstacle_mask))
            closest_overlapping_obstacles.append((i, dist))

    if len(closest_overlapping_obstacles) == 0:
        return [target_index]
    
    closest_overlapping_obstacles.sort(key=lambda x: x[1])
    
    return [id for id, dist in closest_overlapping_obstacles]

def find_obstacles_to_remove1(target_index, segmentation_masks):
    # Convert masks to binary images
    binary_masks = [mask > 0 for mask in segmentation_masks]

    # Identify the target mask
    target_mask = segmentation_masks[target_index]

    # Find obstacles overlapping the target mask
    overlapping_obstacles = []
    for i, obstacle_mask in enumerate(segmentation_masks):
        if i != target_index:
            # overlap = cv2.bitwise_and(target_mask, obstacle_mask)
            # if cv2.countNonZero(overlap) > 0:
            #     overlapping_obstacles.append(i)

            dist = get_distance(get_object_centroid(target_mask), get_object_centroid(obstacle_mask))
            if dist < 100:
                overlapping_obstacles.append(i)

    # Calculate distance transform for the periphery
    periphery_mask = np.zeros_like(binary_masks[0])
    periphery_mask[:, 0] = 1  # Left edge
    periphery_mask[:, -1] = 1  # Right edge
    periphery_mask[0, :] = 1  # Top edge
    periphery_mask[-1, :] = 1  # Bottom edge
    periphery_distance_map = cv2.distanceTransform(np.uint8(periphery_mask), cv2.DIST_L2, 5)

    # Find closest overlapping obstacles to the periphery
    closest_overlapping_obstacles = []
    for obstacle_index in overlapping_obstacles:
        obstacle_mask = binary_masks[obstacle_index]
        min_distance = np.min(obstacle_mask.astype(np.float32) * periphery_distance_map)
        closest_overlapping_obstacles.append((obstacle_index, min_distance))

    # Sort overlapping obstacles based on distance to periphery
    closest_overlapping_obstacles.sort(key=lambda x: x[1])

    if len(closest_overlapping_obstacles) == 0:
        return [target_index]
    
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