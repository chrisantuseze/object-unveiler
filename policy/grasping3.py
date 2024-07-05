import numpy as np
from typing import List, Tuple
from scipy.ndimage import center_of_mass
from policy.grasping import get_distance

class Object:
    def __init__(self, mask: np.ndarray, id: int):
        self.mask = mask
        self.id = id
        self.position = self.calculate_position()
        self.accessible = self.is_accessible()

    def calculate_position(self) -> Tuple[int, int]:
        y, x = np.where(self.mask)
        return (int(np.mean(y)), int(np.mean(x)))

    def is_accessible(self) -> bool:
        y, x = self.position
        height, width = self.mask.shape
        
        # Define inaccessible regions
        if x > 0.4 * width and y < 0.3 * height:  # Top-right corner
            return False
        if x > 0.7 * width and y < 0.7 * height:  # Bottom-right corner
            return False
        
        # Check if object is on the edge accessible to the robot
        # if y == height - 1 or x == 0 or (x == width - 1 and y > 0.1 * height):
        #     return True
        
        # return False
        return True

def find_removal_sequence(objects: List[Object], target: Object) -> List[int]:
    accessible_objects = [obj for obj in objects if obj.accessible]

    # while not target.accessible:
    #     if not accessible_objects:
    #         raise ValueError("No valid removal sequence found")

    #     # Choose the object to remove (here, we're using a simple heuristic)
    #     obj_to_remove = min(accessible_objects, key=lambda obj: 
    #                         abs(obj.position[0] - target.position[0]) + 
    #                         abs(obj.position[1] - target.position[1]))

    #     removal_sequence.append(obj_to_remove.id)
    #     objects.remove(obj_to_remove)
    #     accessible_objects.remove(obj_to_remove)

    #     # Update accessibility of remaining objects
    #     for obj in objects:
    #         obj.accessible = obj.is_accessible()
        
    #     accessible_objects = [obj for obj in objects if obj.accessible]
        
    #     # Check if target is now accessible
    #     target.accessible = target.is_accessible()

    # return removal_sequence

    distances_to_edge = get_distances_to_edge(accessible_objects)

    # Find obstacles overlapping with the target object
    target_obj_distances = []
    for object_idx, object in enumerate(accessible_objects):
        dist = get_distance(center_of_mass(target.mask), center_of_mass(accessible_objects[object_idx].mask))
        target_obj_distances.append(dist)
    
    normalized_periphery_dists = normalize(distances_to_edge)
    normalized_target_obj_dists = normalize(target_obj_distances)
    combined_distances = [d1 + d2 for d1, d2 in zip(normalized_periphery_dists, normalized_target_obj_dists)]
    
    sorted_indices = sorted(range(len(combined_distances)), key=lambda k: combined_distances[k])

    return sorted_indices

def normalize(distances):
    max_distance = max(distances)
    normalized_distances = [distance / max_distance for distance in distances]
    return normalized_distances

def get_distances_to_edge(segmentation_masks: List[Object]):
    distances_list = []

    # Iterate over each segmentation mask
    for i, mask in enumerate(segmentation_masks):
        min_distance = find_centroid_distance(mask.mask)[0]
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

def find_obstacles_to_remove(target_index, segmentation_masks):
    objects = [Object(mask, i) for i, mask in enumerate(segmentation_masks)]
    target = objects[target_index]

    objects = objects[:target_index] + objects[target_index+1:]
    removal_sequence = find_removal_sequence(objects, target)
    print(f"Objects should be removed in this order: {removal_sequence}")
    return removal_sequence