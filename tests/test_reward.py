import pytest

from rlad_bst.reward import Tree, calculate_reward, compare_trees

# Note: Replace `your_module_name` with the actual module name where
# the Tree and compare_trees implementations are defined.


@pytest.mark.parametrize(
    "arr1, arr2, expected_diff",
    [
        # Identical trees
        ([8, 4, 12, 2, 6, 10, 14], [8, 4, 12, 2, 6, 10, 14], 0),
        # One tree empty, the other not
        (
            [],
            [1],
            2,
        ),
        # Both empty trees
        ([], [], 0),
        # Trees differ in one node value (same structure)
        ([8, 4, 12], [8, 4, 13], 1),
        # Trees differ in structure: second tree missing a subtree
        (
            [8, 4, 12, 2],
            [8, 4, 12],
            1 + 1,
        ),
        # Trees differ in structure: first tree missing a subtree
        ([8, 4, 12], [8, 4, 12, 2], 1 + 1),
        # More complex difference: multiple nodes differ
        ([8, 4, 12, 2, 6, 10, 14], [8, 4, 12, 1, 7, 9, 15], 4),
    ],
)
def test_compare_trees(arr1, arr2, expected_diff):
    tree1 = Tree.from_array(arr1)
    tree2 = Tree.from_array(arr2)

    diff = compare_trees(tree1.root, tree2.root)
    assert diff == expected_diff


def test_compare_trees_none_roots():
    # Explicitly test when both roots are None
    # Although covered by empty arrays, this is a direct test on None
    diff = compare_trees(None, None)
    assert diff == 0

    # One None, one not
    t = Tree.from_array([1])
    diff = compare_trees(None, t.root)
    # Difference: 1 + size_of_subtree(t.root)
    # t.root subtree size = 1
    # diff = 1 + 1 = 2
    assert diff == 2


@pytest.mark.parametrize(
    "solution_arr, candidate_arr, expected_reward",
    [
        # Identical trees => difference = 0, reward = -0 = 0
        ([8, 4, 12, 2, 6, 10, 14], [8, 4, 12, 2, 6, 10, 14], 0),
        # One tree empty, the other not
        ([], [1], -2),
        # Both empty trees => difference = 0, reward = 0
        ([], [], 0),
        # Trees differ in one node value (same structure)
        ([8, 4, 12], [8, 4, 13], -1),
        # Trees differ in structure: second tree missing a subtree
        ([8, 4, 12, 2], [8, 4, 12], -2),
        # Trees differ in structure: first tree missing a subtree
        ([8, 4, 12], [8, 4, 12, 2], -2),
        # More complex difference: multiple nodes differ
        ([8, 4, 12, 2, 6, 10, 14], [8, 4, 12, 1, 7, 9, 15], -4),
    ],
)
def test_calculate_reward(solution_arr, candidate_arr, expected_reward):
    reward = calculate_reward(solution_arr, candidate_arr)
    assert (
        reward == expected_reward
    ), f"Expected reward {expected_reward}, got {reward}"
