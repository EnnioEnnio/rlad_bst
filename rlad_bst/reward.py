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
    for i in range(data_len):
        edge_distance_matrix[i, i] -= edge_distance_matrix.max() + 1
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
    # We normalize by adding the maximum negative penalty to make it positive
    # Then we divide by the maximum positive value, which is everything
    # visited => data_len * 0.1 + all correct worst_case + worst_case
    reward = (reward + worst_case) / (2 * worst_case + data_len * 0.1)
    return reward


"""
For comparison reasons the following section will contain
our old reward function and its corresponding classes.
"""


class TreeNode:
    def __init__(self, val: int) -> None:
        self.val: int = val
        self.left: TreeNode | None = None
        self.right: TreeNode | None = None


class Tree:
    def __init__(self, root: TreeNode | None = None) -> None:
        self.root: TreeNode | None = root

    @classmethod
    def from_array(cls, arr: list[int | None]) -> "Tree":
        """Builds a Tree from a given array representation."""

        def build_tree_from_array(index: int) -> TreeNode | None:
            if index < len(arr) and arr[index] is not None:
                node = TreeNode(arr[index])
                node.left = build_tree_from_array(2 * index + 1)
                node.right = build_tree_from_array(2 * index + 2)
                return node
            return None

        return cls(root=build_tree_from_array(0))


def compare_trees(root_t1: TreeNode, root_t2: TreeNode) -> int:
    """
    Compare two trees and return a 'difference score'.
    - High score means high difference (which is bad in our case)
    The difference score is calculated as:
    - If both nodes exist, difference = (0 if values match, else 1) + difference(left) + difference(right)
    - If one node exists and the other doesn't, difference = 1 + (check the existing node's children as mismatches)
    """  # noqa: E501
    if root_t1 is None and root_t2 is None:
        return 0  # Both empty, no difference
    if root_t1 is None and root_t2 is not None:
        # Entire subtree t2 is extra
        return 1 + count_subtree(root_t2)
    if root_t1 is not None and root_t2 is None:
        # Entire subtree t1 is extra
        return 1 + count_subtree(root_t1)

    # Both nodes exist
    diff = 0
    if root_t1.val != root_t2.val:
        diff += 1
    # Compare children
    diff += compare_trees(root_t1.left, root_t2.left)
    diff += compare_trees(root_t1.right, root_t2.right)
    return diff


def count_subtree(node: TreeNode) -> int:
    """
    Count the number of nodes in a subtree.
    This is used to penalize completely missing subtrees.
    """
    if node is None:
        return 0
    return 1 + count_subtree(node.left) + count_subtree(node.right)


def calculate_old_reward(
    solution_arr: list[int], candidate_arr: list[int]
) -> int:
    sol_tree = Tree.from_array(solution_arr)
    cand_tree = Tree.from_array(candidate_arr)

    difference_score = compare_trees(sol_tree.root, cand_tree.root)

    # Reward could simply be the negative of the difference score
    # The fewer the differences, the higher the reward.
    reward = -difference_score
    return reward
