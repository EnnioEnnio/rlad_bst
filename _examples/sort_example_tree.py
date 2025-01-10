import numpy as np

from _examples.playground import Command, SortMachine, getcode


def wait_for_debugger(port: int = 5678):
    """
    Pauses the program until a remote debugger is attached.
    Should only be called on rank0.
    """

    import debugpy

    debugpy.listen(("0.0.0.0", port))
    print(f"Waiting for client to attach on port {port}... ")
    debugpy.wait_for_client()


# wait_for_debugger()

recursion = np.array(
    [
        Command.mark,
        Command.leftchildempty,
        Command.leftchild,
        Command.istreeend,
        Command.write,
        Command.istreeend,
        Command.parent,
        Command.leftchildempty,
        Command.jump,
        Command.nodeempty,
        Command.write,
        Command.rightchildempty,
        Command.rightchild,
        Command.nodeempty,
        Command.jump,
        Command.mark,
        Command.isnottreestart,
        Command.parent,
        Command.rightchildempty,
        Command.drop,
        Command.isnottreestart,
        Command.jump,
        Command.rightchildempty,
        Command.jump,
        Command.nodeempty,
        Command.jump,
        Command.giveresult,
    ]
)

move_to_right = np.array(
    [
        Command.mark,  # 1
        Command.right,
        Command.isnotend,
        Command.jump,
        Command.drop,
    ]
)

outer_loop_beginning = np.array(
    [
        Command.mark,  # 6
        Command.push,
        Command.mark,
        Command.left,
        Command.isnotstart,
        Command.jump,
        Command.drop,
    ]
)

inner_loop = np.array(
    [
        Command.mark,  # 13
        Command.compareright,
        Command.swapright,
        Command.right,
        Command.isnotequal,
        Command.jump,
        Command.drop,
    ]
)

outer_loop_ending = np.array(
    [
        Command.pop,  # 20
        Command.left,
        Command.isnotstart,
        Command.jump,
        Command.drop,
    ]
)

test_code = recursion
encoded = np.array([getcode(cmd) for cmd in test_code])
data = list(range(1, 30))
# random.shuffle(data)
print(data)
temp = data
sm = SortMachine(encoded, data, 0)
print(list(sm.data))


def _make_binary_tree(x):
    def in_order(index, sorted_tree, result):
        if index >= len(result):
            return
        # Left child
        in_order(2 * index + 1, sorted_tree, result)
        # Root
        result[index] = sorted_tree.pop(0)
        # Right child
        in_order(2 * index + 2, sorted_tree, result)

    output_tree = [None] * len(x)
    sorted_tree = list(x)
    in_order(0, sorted_tree, output_tree)
    return output_tree


data = temp
print(_make_binary_tree(data))
