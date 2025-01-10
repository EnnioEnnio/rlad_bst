from typing import Optional

import numpy as np
from numpy.typing import NDArray


class TreeNode:
    def __init__(self, val: int) -> None:
        self.val: int = val
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None


class Tree:
    def __init__(self, root: Optional[TreeNode] = None) -> None:
        self.root: Optional[TreeNode] = root

    @classmethod
    def from_array(cls, arr: list[Optional[int]]) -> "Tree":
        """Builds a Tree from a given array representation."""

        def build_tree_from_array(index: int) -> Optional[TreeNode]:
            if index < len(arr) and arr[index] is not None:
                node = TreeNode(arr[index])
                node.left = build_tree_from_array(2 * index + 1)
                node.right = build_tree_from_array(2 * index + 2)
                return node
            return None

        return cls(root=build_tree_from_array(0))


def compare_trees(root_t1: TreeNode, root_t2: TreeNode) -> int:
    """
    Compare two trees and return a reward score.
    Reward calculation:
    - Matching values get +1 reward.
    - Missing or mismatched values get a penalty:
        - Incorrect value: -1 per node.
        - Missing node: -2 per node.
    Returns: int, the higher the greater is the similarity (+1 for each match)
    """
    if root_t1 is None and root_t2 is None:
        return 0  # Both empty, no penalty or reward
    if root_t1 is None:
        # Penalty for extra nodes in the candidate tree
        return 2 * -count_subtree(root_t2)
    if root_t2 is None:
        # Penalty for missing nodes in the candidate tree
        return 2 * -count_subtree(root_t1)

    # Both nodes exist
    reward = 0
    if root_t1.val == root_t2.val:
        reward += 1  # Reward for matching node values
    else:
        reward -= 1  # Penalty for mismatched node values

    # Recursively compare children
    reward += compare_trees(root_t1.left, root_t2.left)
    reward += compare_trees(root_t1.right, root_t2.right)
    return reward


def count_subtree(node: TreeNode) -> int:
    """
    Count the number of nodes in a subtree.
    This is used to penalize completely missing subtrees.
    """
    if node is None:
        return 0
    return 1 + count_subtree(node.left) + count_subtree(node.right)


def calculate_reward(
    solution_arr: NDArray[np.int64], candidate_arr: NDArray[np.int64]
) -> int:
    # Since we pad the candidate array with 0
    # we need to replace them with Nones
    stripped_cand_array = np.where(candidate_arr == 0, None, candidate_arr)
    sol_tree = Tree.from_array(solution_arr)
    cand_tree = Tree.from_array(stripped_cand_array)

    return compare_trees(sol_tree.root, cand_tree.root)


# Example usage
if __name__ == "__main__":
    solution = [8, 4, 12, 2, 6, 10, 14]
    candidate = [8, 4, 12, 6, 2, 14, 10]

    reward = calculate_reward(solution, candidate)
    print(f"Correct Solution: {solution}")
    print(f"Model Solution: {candidate}")
    print("Reward:", reward)
