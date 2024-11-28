from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

register(id="rlad/bst-v0", entry_point="sort_machine_env:SortingMachine")


class SortingMachine(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, data_len, program_len, verbosity=1, render_mode=None):
        self.data_len = data_len
        self.render_mode = render_mode
        self.verbosity = verbosity

        self._initial_machine_state()
        self.correct_tree = self._make_binary_tree()

        # Each command gets his own number
        self.action_space = spaces.Discrete(14)

        self._action_to_command = {
            0: self.right,
            1: self.left,
            2: self.push,
            3: self.pop,
            4: self.compare,
            5: self.mark,
            6: self.jump,
            7: self.swap,
            8: self.isnotend,
            9: self.isnotstart,
            10: self.isnotequal,
            11: self.drop,
            12: self.swapright,
            13: self.compareright,
        }

        # The programm can observe the
        self.observation_space = gym.spaces.Dict(
            {
                "program": spaces.Sequence(
                    spaces.Box(
                        low=-np.inf, high=np.inf, shape=(), dtype=np.float64
                    ),
                    stack=True,
                ),
                "data": spaces.Sequence(
                    spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64),
                    stack=True,
                ),
                "pointers": spaces.Sequence(
                    spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int64),
                    stack=True,
                ),
                "stack": spaces.Sequence(
                    spaces.Box(
                        low=-np.inf, high=np.inf, shape=(), dtype=np.float64
                    ),
                    stack=True,
                ),
                "skipflag": spaces.Discrete(2),
                "commandpointer": spaces.Box(
                    low=0, high=np.inf, shape=(), dtype=np.int64
                ),
                "lastcommand": spaces.MultiBinary(1),
                "lastconditional": spaces.Discrete(
                    3, start=-1
                ),  # None (-1), True (1), or False (0)
                "execcost": spaces.Box(
                    low=0, high=np.inf, shape=(), dtype=np.int64
                ),
                "storage": spaces.Box(
                    low=-1, high=np.inf, shape=(), dtype=np.int64
                ),
            }
        )

    def _initial_machine_state(self):
        self.program: np.array = np.array([])
        self.data: list = np.array(list(range(self.data_len)))
        self.pointers: list = [0]
        self.stack: list = []
        self.skipflag: bool = False
        self.commandpointer: int = 0
        self.lastcommand: np.array = np.zeros((1), dtype=bool)
        self.lastconditional: int = -1
        self.execcost: int = 0
        self.storage: int = -1

    def _make_binary_tree(self):
        def in_order(index, sorted_tree, result):
            if index >= len(result):
                return
            # Left child
            in_order(2 * index + 1, sorted_tree, result)
            # Root
            result[index] = sorted_tree.pop(0)
            # Right child
            in_order(2 * index + 2, sorted_tree, result)

        output_tree = [None] * len(self.data)
        sorted_tree = list(self.data)
        in_order(0, sorted_tree, output_tree)
        return np.array(output_tree)

    def _get_obs(self) -> dict:
        return {
            "program": self.program,
            "data": self.data,
            "pointers": self.pointers,
            "stack": self.stack,
            "skipflag": self.skipflag,
            "commandpointer": self.commandpointer,
            "lastcommand": self.lastcommand,
            "lastconditional": self.lastconditional,
            "execcost": self.execcost,
            "storage": self.storage,
        }

    def _get_info(self):
        return {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> [dict, None]:
        super().reset(seed=seed)
        self._initial_machine_state()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.lastcommand = action
        self.commandpointer += 1

        self.program = np.append(self.program, action)

        self.lastconditional = -1  # Set to None
        if self.skipflag:
            self.skipflag = False
            return self._get_obs(), 0, False, False, self._get_info()
        self.execcost += 1
        self._action_to_command[action]()

        terminated = np.array_equal(self.data, self.correct_tree)
        reward = terminated  # TODO: Better reward function
        truncated = False  # Used to limit steps

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._get_info(),
        )

    def render(self):
        print("---")
        if self.verbosity >= 2:
            print("data: ", self.highlight_nth_element())
        print("pointers: ", self.pointers)
        print("stack: ", self.stack)
        print("skipflag: ", self.skipflag)
        print("commandpointer: ", self.commandpointer)
        print("lastcommand: ", self.lastcommand)
        print("lastconditional: ", self.lastconditional)
        print("execcost: ", self.execcost)
        print("---")

    def highlight_nth_element(self):
        # Convert the integer array into a string (space-separated)
        arr_str = " ".join(map(str, self.data))

        # Get the nth element to highlight
        nth_element = str(self.data[self.pointers[-1]])

        # Highlight the nth element by adding square brackets around it
        highlighted_str = arr_str.replace(nth_element, f"[{nth_element}]")

        return highlighted_str

    # Available commands #

    def right(self):
        if self.pointers[-1] < len(self.data) - 1:
            self.pointers[-1] += 1

    def left(self):
        if self.pointers[-1] > 0:
            self.pointers[-1] -= 1

    def push(self):
        self.pointers.append(self.pointers[-1])

    def pop(self):
        if len(self.pointers) > 1:
            self.pointers.pop()

    def compare(self):
        self.skipflag = self.data[self.pointers[-1]] <= self.storage

    def mark(self):
        self.stack.append(self.commandpointer - 1)

    def jump(self):
        if len(self.stack) != 0:
            self.commandpointer = self.stack.pop()

    def swap(self):
        temp = self.storage
        self.storage = self.data[self.pointers[-1]]
        self.data[self.pointers[-1]] = temp

    def isnotend(self):
        self.skipflag = self.pointers[-1] >= len(self.data) - 1

    def isnotstart(self):
        self.skipflag = self.pointers[-1] == 0

    def isnotequal(self):
        if len(self.pointers) > 1:
            self.skipflag = self.pointers[-1] == self.pointers[-2]

    def drop(self):
        if len(self.stack) > 0:
            self.stack.pop()
        self.skipflag = False

    def swapright(self):
        temp = self.data[self.pointers[-1] + 1]
        self.data[self.pointers[-1] + 1] = self.data[self.pointers[-1]]
        self.data[self.pointers[-1]] = temp

    def compareright(self):
        self.skipflag = (
            self.data[self.pointers[-1]] <= self.data[self.pointers[-1] + 1]
        )


if __name__ == "__main__":
    # Test env
    def wait_for_debugger(port: int = 5678):
        """
        Pauses the program until a remote debugger is attached.
        Should only be called on rank0.
        """

        import debugpy

        debugpy.listen(("0.0.0.0", port))
        print(f"Waiting for client to attach on port {port}... ")
        debugpy.wait_for_client()

    wait_for_debugger()
    env = gym.make(
        "rlad/bst-v0", render_mode="human", data_len=10, program_len=100
    )

    check_env(env.unwrapped)

    obs = env.reset()[0]
    print(obs)

    for _ in range(10):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(rand_action, obs, reward, terminated)
