import random

import numpy as np

from .sortmachine import Command, SortMachine, getcode


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

test_code = np.concatenate(
    (move_to_right, outer_loop_beginning, inner_loop, outer_loop_ending)
)
encoded = np.array([getcode(cmd) for cmd in test_code])
data = list(range(7))
random.shuffle(data)
sm = SortMachine(encoded, data, 0)
print(sm.data)
