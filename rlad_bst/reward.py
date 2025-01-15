import numpy as np
from numpy.typing import NDArray


def find_path(index):
    """Finds the path from a node to the root in the binary tree."""
    path = []
    while index >= 0:
        path.append(index)
        index = (index - 1) // 2  # Move to the parent
    return path


def compute_edge_distance(idx1, idx2):
    """Computes the edge distance between two indices in a binary tree."""
    path1 = find_path(idx1)
    path2 = find_path(idx2)

    # Find the Lowest Common Ancestor (LCA)
    path1_set = set(path1)
    for ancestor in path2:
        if ancestor in path1_set:
            lca = ancestor
            break

    # Distance = (Distance from idx1 to LCA) + (Distance from idx2 to LCA)
    distance = path1.index(lca) + path2.index(lca)
    return distance


def get_distance_matrix(data_len: int) -> NDArray[np.int64]:
    edge_distance_matrix = np.zeros((data_len, data_len), dtype=np.int64)
    for i in range(data_len):
        for j in range(data_len):
            edge_distance_matrix[i, j] = compute_edge_distance(i, j)
    return edge_distance_matrix


def calculate_reward(
    candidate_arr: NDArray[np.int64],
    edge_distance_matrix: NDArray[np.int64],
    max_penalty: int,
    visited: NDArray[np.bool],
    correct_positions: NDArray[np.int64],
    data_len: int,
) -> int:
    reward = 0
    for i, node in enumerate(candidate_arr):
        if node == 0:
            reward -= max_penalty
        else:
            reward -= edge_distance_matrix[correct_positions[node], i]
    worst_case = max_penalty * data_len
    reward += visited.sum() * 0.1
    reward = (reward + worst_case) / (worst_case + data_len * 0.1)
    return reward
