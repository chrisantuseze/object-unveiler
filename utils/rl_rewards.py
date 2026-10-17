"""
Reward functions for RL fine-tuning of the Spatial Relationship Encoder.

This module defines various reward components for training the SRE with RL:
- Distance-based rewards (encouraging progress toward target)
- Stability penalties (discouraging object disturbances)
- Efficiency bonuses (rewarding fewer steps)
- Success rewards (reaching the target)
"""

import numpy as np
import torch
import cv2
from scipy.spatial import distance


def compute_target_distance(target_mask, object_masks, removed_idx):
    """
    Compute the distance from the removed object to the target.
    Lower distance = better removal choice (closer to target)
    
    Args:
        target_mask: Binary mask of target object [H, W]
        object_masks: Masks of all objects [N, H, W]
        removed_idx: Index of the object that was removed
    
    Returns:
        distance: Normalized distance (0-1, lower is better)
    """
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.cpu().numpy()
    if isinstance(object_masks, torch.Tensor):
        object_masks = object_masks.cpu().numpy()
    
    # Get centroids
    target_centroid = get_centroid(target_mask.squeeze())
    removed_mask = object_masks[removed_idx].squeeze()
    removed_centroid = get_centroid(removed_mask)
    
    if target_centroid is None or removed_centroid is None:
        return 1.0  # Max distance penalty
    
    # Compute Euclidean distance
    dist = np.linalg.norm(np.array(target_centroid) - np.array(removed_centroid))
    
    # Normalize by image diagonal
    H, W = target_mask.shape[-2:]
    max_dist = np.sqrt(H**2 + W**2)
    normalized_dist = dist / max_dist
    
    return normalized_dist


def compute_stability_penalty(obs_before, obs_after, removed_object_id=None):
    """
    Penalize removals that cause other objects to fall or move significantly.
    
    Args:
        obs_before: Observation before removal (contains full_state)
        obs_after: Observation after removal
        removed_object_id: ID of the removed object
    
    Returns:
        penalty: Stability penalty (0-1, higher = more instability)
    """
    if 'full_state' not in obs_before or 'full_state' not in obs_after:
        return 0.0
    
    penalty = 0.0
    objects_before = {obj.body_id: obj for obj in obs_before['full_state']}
    objects_after = {obj.body_id: obj for obj in obs_after['full_state']}
    
    # If removed_object_id does not match any body_id, ignore it
    if removed_object_id is not None and removed_object_id not in objects_before:
        removed_object_id = None

    # Check each object that remains in the scene
    for body_id, obj_before in objects_before.items():
        if removed_object_id is not None and body_id == removed_object_id:
            continue
        
        if body_id not in objects_after:
            # Object disappeared (fell off table or was removed accidentally)
            penalty += 1.0
            continue
        
        obj_after = objects_after[body_id]
        
        # Compute position change
        pos_change = np.linalg.norm(
            np.array(obj_before.pos) - np.array(obj_after.pos)
        )
        
        # Penalize significant movements (threshold: 5cm)
        if pos_change > 0.05:
            penalty += pos_change * 2.0
    
    # Normalize by number of objects
    if len(objects_before) > 1:
        penalty /= (len(objects_before) - 1)
    
    return min(penalty, 1.0)  # Cap at 1.0


def compute_path_clearance_reward(target_mask, object_masks, removed_idx):
    """
    Reward removing objects that block the path to the target.
    Uses ray-casting from target to removed object.
    
    Args:
        target_mask: Binary mask of target [H, W]
        object_masks: Masks of all objects [N, H, W]
        removed_idx: Index of removed object
    
    Returns:
        reward: Path clearance reward (0-1, higher = better path clearance)
    """
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.cpu().numpy()
    if isinstance(object_masks, torch.Tensor):
        object_masks = object_masks.cpu().numpy()
    
    target_centroid = get_centroid(target_mask.squeeze())
    removed_centroid = get_centroid(object_masks[removed_idx].squeeze())
    
    if target_centroid is None or removed_centroid is None:
        return 0.0
    
    # Count how many other objects lie on the line between target and removed object
    blocking_objects = 0
    for i, mask in enumerate(object_masks):
        if i == removed_idx:
            continue
        
        if mask.sum() == 0:  # Skip padding
            continue
        
        centroid = get_centroid(mask.squeeze())
        if centroid is None:
            continue
        
        # Check if this object is between target and removed object
        if is_on_line(target_centroid, removed_centroid, centroid, threshold=20):
            blocking_objects += 1
    
    # Higher reward if removed object was blocking other objects
    reward = blocking_objects / max(len(object_masks) - 1, 1)
    
    return reward


def compute_grasp_accessibility_reward(object_masks, removed_idx):
    """
    Reward removing objects that are easy to grasp (accessible from top).
    
    Args:
        object_masks: Masks of all objects [N, H, W]
        removed_idx: Index of removed object
    
    Returns:
        reward: Accessibility reward (0-1, higher = more accessible)
    """
    if isinstance(object_masks, torch.Tensor):
        object_masks = object_masks.cpu().numpy()
    
    removed_mask = object_masks[removed_idx].squeeze()
    
    # Compute how many other objects overlap with the removed object
    overlaps = 0
    for i, mask in enumerate(object_masks):
        if i == removed_idx or mask.sum() == 0:
            continue
        
        overlap = np.logical_and(removed_mask, mask.squeeze()).sum()
        if overlap > 0:
            overlaps += 1
    
    # More overlap = less accessible = lower reward
    max_overlaps = len(object_masks) - 1
    accessibility = 1.0 - (overlaps / max(max_overlaps, 1))
    
    return accessibility


def compute_episode_reward(
    target_mask,
    object_masks,
    removed_idx,
    grasp_success,
    collision,
    step_num,
    max_steps,
    target_reached=False,
    target_id=None,
    num_objects=None
):
    """
    Compute the total reward for a single step in the episode.
    
    Simplified reward focusing on what matters most:
    1. Did we reach the target? (huge bonus)
    2. Did we correctly select the target when it was accessible? (big bonus)
    3. Did we pick an obstacle close to the target? (shaping)
    4. Small step penalty to encourage efficiency
    
    Args:
        target_mask: Binary mask of target object
        object_masks: Masks of all objects before removal
        removed_idx: Index of the object that was removed
        grasp_success: Whether the grasp was successful
        collision: Whether a collision occurred
        step_num: Current step number in episode
        max_steps: Maximum steps allowed
        target_reached: Whether the target was successfully retrieved
        target_id: Index of the target in object_masks
        num_objects: Number of actual (non-padded) objects
    
    Returns:
        reward: Total reward for this step
        reward_components: Dictionary of individual reward components
    """
    reward_components = {}
    
    if num_objects is None:
        num_objects = len(object_masks)
    
    # ====== 1. TARGET REACHED (highest priority) ======
    if target_reached:
        # Big bonus, scaled by how quickly we got here
        efficiency_bonus = (max_steps - step_num) / max_steps
        reward_components['target_reached'] = 20.0
        reward_components['speed_bonus'] = efficiency_bonus * 10.0
    else:
        reward_components['target_reached'] = 0.0
        reward_components['speed_bonus'] = 0.0
    
    # ====== 2. TARGET ACCESSIBILITY — select target when graspable ======
    # Use actual mask overlap to determine if target is accessible,
    # instead of a hard object-count threshold.
    selected_target = (target_id is not None and removed_idx == target_id)
    target_accessible = _is_target_accessible(target_mask, object_masks, target_id)
    
    if selected_target and target_accessible:
        # Correctly chose to grasp the target when it was accessible
        reward_components['target_selection'] = 10.0
    elif not selected_target and target_accessible:
        # Should have grasped the target but chose an obstacle instead
        reward_components['target_selection'] = -5.0
    elif selected_target and not target_accessible:
        # Tried to grasp target when it's buried — bad idea
        reward_components['target_selection'] = -3.0
    else:
        reward_components['target_selection'] = 0.0
    
    # ====== 3. OCCLUSION-BASED SHAPING (only for obstacle removal) ======
    # Reward removing objects that actually overlap/occlude the target
    # and that clear the path to the target — not just centroid proximity.
    if not selected_target:
        overlap_reward = _compute_overlap_with_target(target_mask, object_masks, removed_idx)
        clearance_reward = compute_path_clearance_reward(target_mask, object_masks, removed_idx)
        # Combine: overlap matters most, path clearance is secondary
        reward_components['occlusion'] = overlap_reward * 2.5 + clearance_reward * 1.5
    else:
        reward_components['occlusion'] = 0.0
    
    # ====== 4. GRASP OUTCOME (light feedback) ======
    if grasp_success:
        reward_components['grasp'] = 1.0
    else:
        reward_components['grasp'] = -1.0
    
    # ====== 5. STEP PENALTY (encourage fewer steps) ======
    reward_components['step_penalty'] = -0.1
    
    # Sum all components
    total_reward = sum(reward_components.values())
    
    return total_reward, reward_components


def _is_target_accessible(target_mask, object_masks, target_id):
    """
    Determine if the target is accessible for grasping by checking how much
    of its surface is occluded by other objects.
    
    Uses mask overlap ratio instead of a hard object-count threshold.
    The target is considered accessible if less than 15% of its pixels
    are overlapped by other object masks.
    
    Args:
        target_mask: Binary mask of target [H, W]
        object_masks: Masks of all objects [N, H, W] (may include padding)
        target_id: Index of the target in object_masks
    
    Returns:
        bool: True if target is accessible for direct grasping
    """
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.cpu().numpy()
    if isinstance(object_masks, torch.Tensor):
        object_masks = object_masks.cpu().numpy()
    
    target_flat = target_mask.squeeze().astype(bool)
    target_area = target_flat.sum()
    
    if target_area == 0:
        return False
    
    # Accumulate overlap from all non-target objects
    total_overlap = 0
    for i, mask in enumerate(object_masks):
        if i == target_id:
            continue
        m = mask.squeeze().astype(bool)
        if m.sum() == 0:  # skip padding
            continue
        total_overlap += np.logical_and(target_flat, m).sum()
    
    overlap_ratio = total_overlap / target_area
    # Target is accessible when less than 15% occluded
    return overlap_ratio < 0.15


def _compute_overlap_with_target(target_mask, object_masks, removed_idx):
    """
    Compute how much the removed object overlaps (occludes) the target.
    Higher overlap means removing it directly helps expose the target.
    
    Returns:
        reward: Normalized overlap reward in [0, 1]
    """
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.cpu().numpy()
    if isinstance(object_masks, torch.Tensor):
        object_masks = object_masks.cpu().numpy()
    
    target_flat = target_mask.squeeze().astype(bool)
    removed_flat = object_masks[removed_idx].squeeze().astype(bool)
    
    target_area = target_flat.sum()
    if target_area == 0:
        return 0.0
    
    overlap = np.logical_and(target_flat, removed_flat).sum()
    # Normalize by target area: what fraction of the target does this object cover?
    return min(float(overlap) / float(target_area), 1.0)


def get_centroid(mask):
    """
    Compute the centroid of a binary mask.
    
    Args:
        mask: Binary mask [H, W]
    
    Returns:
        centroid: (y, x) coordinates of centroid, or None if empty
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if mask.sum() == 0:
        return None
    
    # Use moments to find centroid
    moments = cv2.moments(mask.astype(np.uint8))
    if moments['m00'] == 0:
        return None
    
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    
    return (cy, cx)


def is_on_line(p1, p2, p3, threshold=10):
    """
    Check if point p3 is approximately on the line segment between p1 and p2.
    
    Args:
        p1, p2, p3: Points as (y, x) tuples
        threshold: Distance threshold in pixels
    
    Returns:
        bool: True if p3 is on the line between p1 and p2
    """
    # Compute distance from p3 to line segment p1-p2
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    
    # Vector from p1 to p2
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    
    if line_len == 0:
        return False
    
    # Normalized direction vector
    line_dir = line_vec / line_len
    
    # Vector from p1 to p3
    p1_to_p3 = p3 - p1
    
    # Project p3 onto line
    proj_length = np.dot(p1_to_p3, line_dir)
    
    # Check if projection is within line segment
    if proj_length < 0 or proj_length > line_len:
        return False
    
    # Compute perpendicular distance
    proj_point = p1 + proj_length * line_dir
    perp_dist = np.linalg.norm(p3 - proj_point)
    
    return perp_dist < threshold


def compute_dense_reward_simple(
    prev_distance_to_target,
    curr_distance_to_target,
    grasp_success,
    collision,
    target_reached=False
):
    """
    Simplified dense reward function for quick testing.
    
    Args:
        prev_distance_to_target: Distance to target before action
        curr_distance_to_target: Distance to target after action
        grasp_success: Whether grasp was successful
        collision: Whether collision occurred
        target_reached: Whether target was retrieved
    
    Returns:
        reward: Total reward
    """
    reward = 0.0
    
    # Reward for reducing distance to target
    if prev_distance_to_target is not None:
        distance_delta = prev_distance_to_target - curr_distance_to_target
        reward += distance_delta * 5.0
    
    # Grasp success/failure
    if grasp_success:
        reward += 2.0
    else:
        reward -= 1.0
    
    # Collision penalty
    if collision:
        reward -= 2.0
    
    # Target reached (big reward)
    if target_reached:
        reward += 20.0
    
    return reward
