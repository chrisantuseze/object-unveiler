import random
from collections import deque
import torch

from policy.grasping import calculate_iou, extract_relationships
import utils.logger as logging

# Example graph representation
# graph = {
#     'A': ['B', 'C'],
#     'B': ['A', 'D', 'E'],
#     'C': ['A', 'F'],
#     'D': ['B'],
#     'E': ['B', 'F'],
#     'F': ['C', 'E'],
# }

def is_neighbor(graph, node1, node2):
    # Check if node2 is a neighbor of node1
    return node2 in graph.get(node1, [])

def path_from_target_to_neighbor(graph, target_node, neighbor_node):
    # Breadth-first search to find the shortest path from target_node to neighbor_node
    queue = deque()
    visited = set()
    parent_map = {}

    queue.append(target_node)
    visited.add(target_node)

    while queue:
        current_node = queue.popleft()

        if current_node == neighbor_node:
            # Found the neighbor_node, reconstruct the path
            path = []
            while current_node != target_node:
                path.append(current_node)
                current_node = parent_map[current_node]
            path.append(target_node)
            return list(reversed(path))

        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                parent_map[neighbor] = current_node

    # If no path exists, return None
    return None

def get_neighbors(graph, node):
    # Get the neighbors of a node
    return graph.get(node, [])

# Example graph representation
# graph = {
#     'A': ['B', 'C'],
#     'B': ['A', 'D', 'E'],
#     'C': ['A', 'F'],
#     'D': ['B'],
#     'E': ['B', 'F'],
#     'F': ['C', 'E'],
# }

def build_graph(object_masks):
    graph = {}
    # Add nodes (objects) to the graph
    for idx, mask in enumerate(object_masks):
        graph[idx] = []

    for i, mask_i in enumerate(object_masks):
        for j, mask_j in enumerate(object_masks):
            if i == j:  # Avoid self-comparison
                continue

            threshold_iou=0.0001

            mask_i = torch.Tensor.float(torch.tensor(mask_i))
            mask_j = torch.Tensor.float(torch.tensor(mask_j))

            iou = calculate_iou(mask_i, mask_j)

            if iou >= threshold_iou:
                graph[i].append(j)

    return graph

def shortest_path_to_neighbor(segmentation_masks, target_node):
    graph = build_graph(segmentation_masks)
    logging.info(graph)

    queue = deque(graph)
    queue.append(target_node)
    shortest_path = None

    while queue:
        current_node = queue.popleft()

        # Check if current_node is a neighbor of the target node
        if is_neighbor(graph, current_node, target_node):
            shortest_path = path_from_target_to_neighbor(graph, target_node, current_node)
            break

        # Enqueue all children (neighbors) of the current_node
        for neighbor in get_neighbors(graph, current_node):
            queue.append(neighbor)

    if shortest_path:
        shortest_path.remove(target_node)
        shortest_path.append(target_node)
        
    return shortest_path

# Randomly select a target node from the graph
# graph = {}  # Replace with your graph structure
# target_node = random.choice(list(graph.keys()))

# # Call the function to find the shortest path
# shortest_path = shortest_path_to_neighbor(graph, target_node)

# logging.info("Shortest path:", shortest_path)

# Example usage:
# target_node = 'A'
# neighbor_node = 'F'
# if is_neighbor(target_node, neighbor_node):
#     shortest_path = path_from_target_to_neighbor(target_node, neighbor_node)
#     logging.info("Shortest path from", target_node, "to", neighbor_node, ":", shortest_path)
# else:
#     logging.info(target_node, "and", neighbor_node, "are not neighbors.")
