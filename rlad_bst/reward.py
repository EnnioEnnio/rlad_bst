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


def calculate_reward(
    solution_arr: NDArray[np.int64], candidate_arr: NDArray[np.int64]
) -> int:
    # Since we pad the candidate array with 0
    # we need to replace them with Nones
    stripped_cand_array = np.where(candidate_arr == 0, None, candidate_arr)
    sol_tree = Tree.from_array(solution_arr)
    cand_tree = Tree.from_array(stripped_cand_array)

    difference_score = compare_trees(sol_tree.root, cand_tree.root)

    # Reward could simply be the negative of the difference score
    # The fewer the differences, the higher the reward.
    reward = -difference_score
    return reward


# Example usage
if __name__ == "__main__":
    solution = [8, 4, 12, 2, 6, 10, 14]
    candidate = [8, 4, 12, 6, 2, 14, 10]

    reward = calculate_reward(solution, candidate)
    print("Reward:", reward)
