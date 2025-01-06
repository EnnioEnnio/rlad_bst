from typing import Optional

import gymnasium as gym
import numpy as np
import wandb
from gymnasium import spaces
from gymnasium.envs.registration import register

from rlad_bst.model import get_model
from rlad_bst.reward import calculate_reward

register(id="rlad/bst-v0", entry_point="sort_machine_env:SortingMachine")


class SortingMachine(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        data_len,
        program_len,
        maximum_exec_cost,
        verbosity=1,
        render_mode=None,
    ):
        self.data_len = data_len
        self.program_len = program_len
        self.render_mode = render_mode
        self.verbosity = verbosity
        self.maximum_exec_cost = maximum_exec_cost

        # Each command gets his own number
        self._action_to_command = {
            0: self.right,
            1: self.left,
            2: self.push,
            3: self.pop,
            4: self.mark,
            5: self.jump,
            6: self.isnotend,
            7: self.isnotstart,
            8: self.isnotequal,
            9: self.drop,
            10: self.swapright,
            11: self.compareright,
            12: self.leftchild,
            13: self.rightchild,
            14: self.parent,
            15: self.leftchildempty,
            16: self.rightchildempty,
            17: self.write,
            18: self.nodeempty,
            19: self.istreeend,
            20: self.isnottreestart,
        }

        # Inverse dict to make it human readable
        self._command_to_action = {
            cmd.__name__: action
            for action, cmd in self._action_to_command.items()
        }

        self._conditional_actions = [
            self._command_to_action["isnotend"],
            self._command_to_action["isnotstart"],
            self._command_to_action["isnotequal"],
            self._command_to_action["compareright"],
            self._command_to_action["leftchildempty"],
            self._command_to_action["rightchildempty"],
            self._command_to_action["nodeempty"],
            self._command_to_action["istreeend"],
            self._command_to_action["isnottreestart"],
        ]

        self.action_space = spaces.Discrete(len(self._action_to_command))

        self.pad = 0

        self.observation_space = spaces.Dict(
            {
                "program": spaces.Box(
                    low=0,
                    high=len(self._action_to_command),
                    shape=(program_len,),
                    dtype=np.int64,
                ),  # np.array of size program_len
                "data": spaces.Box(
                    low=0, high=np.inf, shape=(data_len,), dtype=np.int64
                ),  # np.array of size data_len
                "pointers": spaces.Box(
                    low=0, high=data_len + 1, shape=(data_len,), dtype=np.int64
                ),  # np.array of size data_len
                "stack": spaces.Box(
                    low=0,
                    high=program_len + 1,
                    shape=(data_len,),
                    dtype=np.int64,
                ),  # np.array of size data_len
                "skipflag": spaces.Box(
                    low=0, high=2, shape=(1,), dtype=np.int64
                ),  # np.int64 (0 or 1)
                "commandpointer": spaces.Box(
                    low=0, high=program_len + 1, shape=(1,), dtype=np.int64
                ),  # np.int64
                "last_action": spaces.Box(
                    low=0,
                    high=len(self._action_to_command),
                    shape=(1,),
                    dtype=np.int64,
                ),  # np.int64
                "execcost": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.int64
                ),  # np.int64
                "result": spaces.Box(
                    low=0, high=np.inf, shape=(data_len,), dtype=np.int64
                ),  # np.array of size data_len
                "pointersresult": spaces.Box(
                    low=0, high=data_len + 1, shape=(data_len,), dtype=np.int64
                ),  # np.array of size data_len
            }
        )

        self._initial_machine_state()
        self.correct_tree = self._make_binary_tree()

    def _initial_machine_state(self):
        self.program: np.array = np.array([], dtype=np.int64)
        self.data: np.array = np.array(list(range(1, self.data_len + 1)))
        self.pointers: list = [0]
        self.stack: list = []
        self.skipflag: bool = False
        self.commandpointer: int = 0
        self.last_action: int = len(self._action_to_command)
        self.execcost: int = 0
        self.result: np.array = np.ones_like(self.data, dtype=int) * self.pad
        self.pointersresult: list = [0]

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
        program = np.concatenate(
            (
                self.program,
                np.ones((self.program_len - len(self.program)))
                * len(self._action_to_command),
            )
        ).astype(np.int64)
        pointers = np.concatenate(
            (
                self.pointers,
                np.ones((self.data_len - len(self.pointers))) * self.data_len,
            )
        ).astype(np.int64)
        pointersresult = np.concatenate(
            (
                self.pointersresult,
                np.ones((self.data_len - len(self.pointersresult)))
                * self.data_len,
            )
        ).astype(np.int64)
        stack = np.concatenate(
            (
                self.stack,
                np.ones((self.data_len - len(self.stack))) * self.program_len,
            )
        ).astype(np.int64)

        return {
            "program": program,
            "data": self.data,
            "pointers": pointers,
            "stack": stack,
            "skipflag": np.array([self.skipflag]),
            "commandpointer": np.array([self.commandpointer]),
            "last_action": np.array([self.last_action]),
            "execcost": np.array([self.execcost]),
            "result": self.result,
            "pointersresult": pointersresult,
        }

    def _get_info(self):
        return {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> list[dict, None]:
        super().reset(seed=seed)
        self._initial_machine_state()
        return self._get_obs(), self._get_info()

    def _check_terminated(self):
        # return np.array_equal(self.data, self.correct_tree)
        return np.array_equal(self.result, self.correct_tree)

    def _check_trucated(self):
        return (
            self.commandpointer >= self.program_len
            or self.execcost == self.maximum_exec_cost
        )

    def step(self, action):
        """
        This function should run the program until the next input is expected.
        We apply actions as long as:
        1. We have not terminated, which can happend when:
            1.1 The solution is correct
            1.2 The program has reached its max length
            1.3 The program has reached its maximum execution cost
        2. The machine awaits a new instruction
        """

        self.program = np.append(self.program, action)
        terminated = self._check_terminated()
        truncated = self._check_trucated()
        while not (
            terminated or truncated or self.commandpointer >= len(self.program)
        ):
            self._transition_once()
            if self.verbosity >= 1:
                self.render()
            terminated = self._check_terminated()
            truncated = self._check_trucated()

        reward = calculate_reward(
            solution_arr=self.correct_tree, candidate_arr=self.result
        )

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._get_info(),
        )

    def _transition_once(self):
        cmd = self.program[self.commandpointer]
        self.commandpointer += 1

        if self.skipflag:
            self.skipflag = False
        else:
            self.execcost += 1
            self._action_to_command[cmd]()
        self.last_action = cmd

    def render(self):
        print("---")
        if self.verbosity >= 2:
            print("data: ", self.highlight_nth_element())
        print("pointers: ", self.pointers)
        print("stack: ", self.stack)
        print("skipflag: ", self.skipflag)
        print("commandpointer: ", self.commandpointer)
        print("last_action: ", self.last_action)
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

    def action_masks(self) -> np.array:
        # If we are at the start we only allow right, push, mark, swapright
        if self.last_action == len(self._action_to_command):
            mask = np.zeros(len(self._action_to_command))
            mask[
                [
                    self._command_to_action["right"],
                    self._command_to_action["push"],
                    self._command_to_action["mark"],
                    self._command_to_action["compareright"],
                    self._command_to_action["write"],
                    self._command_to_action["leftchild"],
                    self._command_to_action["rightchild"],
                ]
            ] = 1
            return mask

        mask = np.ones(len(self._action_to_command))
        # We do not allow the same action twice in a row
        mask[self.last_action] = 0

        # If the last action was a conditional
        # the current action can not be a conditional
        if self.last_action in self._conditional_actions:
            mask[self._conditional_actions] = 0

        # Check for individual actions
        if self.pointers[-1] == 0:
            mask[[self._command_to_action["left"]]] = 0

        elif self.pointers[-1] == len(self.data) - 1:
            mask[
                [
                    self._command_to_action["right"],
                    self._command_to_action["compareright"],
                ]
            ] = 0

        if len(self.pointersresult) == self.data_len - 1:
            mask[[self._command_to_action["push"]]] = 0

        if (
            len(self.pointersresult) == 1
            or self.last_action == self._command_to_action["push"]
        ):
            mask[[self._command_to_action["pop"]]] = 0

        if len(self.stack) == self.data_len - 1:
            mask[[self._command_to_action["mark"]]] = 0

        # Check that prior to a jump a conditional is checked
        if (
            len(self.stack) == 0
            or self.last_action not in self._conditional_actions
        ):
            mask[[self._command_to_action["jump"]]] = 0

        if len(self.pointers) == 1:
            mask[[self._command_to_action["isnotequal"]]] = 0

        if (
            len(self.stack) == 0
            or self.last_action == self._command_to_action["mark"]
        ):
            mask[[self._command_to_action["drop"]]] = 0

        if self.last_action != self._command_to_action["compareright"]:
            mask[[self._command_to_action["swapright"]]] = 0

        if self.pointersresult[-1] == 0:
            mask[[self._command_to_action["parent"]]] = 0

        return mask

    # Available commands #

    def right(self):
        if self.pointers[-1] < len(self.data) - 1:
            self.pointers[-1] += 1

    def left(self):
        if self.pointers[-1] > 0:
            self.pointers[-1] -= 1

    def push(self):
        if len(self.pointersresult) < self.data_len - 1:
            self.pointersresult.append(self.pointersresult[-1])

    def pop(self):
        if len(self.pointersresult) > 1:
            self.pointersresult.pop()

    def mark(self):
        if len(self.stack) < self.data_len - 1:
            self.stack.append(self.commandpointer - 1)
        else:
            self.invalid_action = True

    def jump(self):
        if len(self.stack) != 0:
            self.commandpointer = self.stack.pop()

    def isnotend(self):
        self.skipflag = self.pointers[-1] > len(self.data) - 1

    def isnotstart(self):
        self.skipflag = self.pointers[-1] == 0

    def isnotequal(self):
        if len(self.pointers) > 1:
            self.skipflag = self.pointers[-1] == self.pointers[-2]

    def drop(self):
        if (
            len(self.stack) > 0
            and self.last_action != self._command_to_action["mark"]
        ):
            self.stack.pop()
            self.skipflag = False

    def swapright(self):
        # Before swapping you need to compare
        try:
            temp = self.data[self.pointers[-1] + 1]
            self.data[self.pointers[-1] + 1] = self.data[self.pointers[-1]]
            self.data[self.pointers[-1]] = temp
        except IndexError:
            pass

    def compareright(self):
        try:
            self.skipflag = (
                self.data[self.pointers[-1]]
                <= self.data[self.pointers[-1] + 1]
            )
        except IndexError:
            pass

    def leftchild(self):
        left_child = self.pointersresult[-1] * 2 + 1
        if left_child < len(self.data):
            self.pointersresult[-1] = left_child

    def rightchild(self):
        right_child = self.pointersresult[-1] * 2 + 2
        if right_child < len(self.data):
            self.pointersresult[-1] = right_child

    def leftchildempty(self):
        left_child = self.pointersresult[-1] * 2 + 1
        if left_child >= len(self.data):
            self.skipflag = 1
        else:
            self.skipflag = self.result[left_child] != 0

    def rightchildempty(self):
        right_child = self.pointersresult[-1] * 2 + 2
        if right_child >= len(self.data):
            self.skipflag = 1
        else:
            self.skipflag = self.result[right_child] != 0

    def nodeempty(self):
        self.skipflag = self.result[self.pointersresult[-1]] != 0

    def parent(self):
        if self.pointersresult[-1] != 0:
            self.pointersresult[-1] = int((self.pointersresult[-1] - 1) / 2)

    def write(self):
        self.result[self.pointersresult[-1]] = self.data[self.pointers[-1]]
        self.right()

    def istreeend(self):
        self.skipflag = self.pointersresult[-1] * 2 + 1 <= len(self.data) - 1

    def isnottreestart(self):
        self.skipflag = self.pointersresult[-1] == 0
