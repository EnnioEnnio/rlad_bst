from typing import Optional

import gymnasium as gym
import numpy as np


class SortingMachine(gym.Env):
    def __init__(self, data_len, program_length):
        self.data_len = data_len

        self._initial_machine_state()
        self.correct_tree = self._make_binary_tree()

        # Each command gets his own number
        self.action_space = gym.spaces.Discrete(14)

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
            14: self.compareright,
        }

        # The programm can observe the
        self.observation_space = gym.spaces.Dict(
            {
                "program": gym.spaces.Discrete(
                    program_length
                ),  # TODO: Do I need to limit this??
                "data": gym.spaces.Discrete(data_len),
                "pointers": gym.spaces.Discrete(data_len),
                "stack": gym.spaces.Discrete(data_len),
                "skipflag": gym.spaces.Discrete(1),
                "commandpointer": gym.spaces.Discrete(1),
                "lastcommand": gym.spaces.Discrete(1),
                "lastconditional": gym.spaces.Discrete(1),
                "execcost": gym.spaces.Discrete(1),
                "storage": gym.spaces.Discrete(1),
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
        self.lastconditional: Optional[bool] = None
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
        in_order(0, self.data, output_tree)
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
        return

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> [dict, None]:
        super().reset(seed=seed)
        self._initial_machine_state()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.lastconditional = None
        if self.skipflag:
            self.skipflag = False
            return self._get_obs(), 0, False, False, self._get_info()
        self.execcost += 1
        self._action_to_command[action]()

        terminated = np.array_equal(self.data, self.correct_tree)
        reward = terminated  # TODO: Better reward function
        truncated = False  # TODO: Whats that?

        return self._get_obs(), reward, terminated, truncated, self._get_info()

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
