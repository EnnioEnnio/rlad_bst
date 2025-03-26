from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from print_on_steroids import logger

from rlad_bst.reward import (
    calculate_old_reward,
    calculate_reward,
    get_distance_matrix,
)

register(id="rlad/bst-v0", entry_point="bst_sort_machine:SortingMachine")


class SortingMachine(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        max_data_len,
        start_data_len,
        start_program_len_factor,
        max_program_len_factor,
        max_exec_cost_factor,
        do_action_masking,
        correct_reward_scale,
        incremental_reward=False,
        verbosity=1,
        render_mode=None,
        reward_function="new",
        naive=False,
    ):
        self.max_data_len = max_data_len
        self.current_data_len = start_data_len
        self.max_program_len_factor = max_program_len_factor
        self.program_len_factor = start_program_len_factor
        self.render_mode = render_mode
        self.verbosity = verbosity
        self.do_action_masking = do_action_masking
        self.max_exec_cost_factor = max_exec_cost_factor
        self.overall_max_program_len = (
            max_program_len_factor * self.max_data_len
        )
        self.reward_function = reward_function
        self.correct_reward_scale = correct_reward_scale
        self.incremental_reward = incremental_reward
        self.naive = naive

        self.pad_value = -1

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
        self._cmd_name_to_action_nr = {
            cmd.__name__: action
            for action, cmd in self._action_to_command.items()
        }

        self._action_nr_to_cmd_name = {
            action_nr: cmd_name
            for cmd_name, action_nr in self._cmd_name_to_action_nr.items()
        }

        self._conditional_actions = [
            self._cmd_name_to_action_nr["isnotend"],
            self._cmd_name_to_action_nr["isnotstart"],
            self._cmd_name_to_action_nr["isnotequal"],
            self._cmd_name_to_action_nr["compareright"],
            self._cmd_name_to_action_nr["leftchildempty"],
            self._cmd_name_to_action_nr["rightchildempty"],
            self._cmd_name_to_action_nr["nodeempty"],
            self._cmd_name_to_action_nr["istreeend"],
            self._cmd_name_to_action_nr["isnottreestart"],
        ]

        self._valid_first_actions = [
            self._cmd_name_to_action_nr["right"],
            self._cmd_name_to_action_nr["push"],
            self._cmd_name_to_action_nr["mark"],
            self._cmd_name_to_action_nr["compareright"],
            self._cmd_name_to_action_nr["leftchild"],
            self._cmd_name_to_action_nr["rightchild"],
        ]

        # To solve the problem for a fixed size you only need a few commands
        self._naive_actions = [
            self._cmd_name_to_action_nr["leftchild"],
            self._cmd_name_to_action_nr["rightchild"],
            self._cmd_name_to_action_nr["write"],
            self._cmd_name_to_action_nr["parent"],
        ]

        self.action_space = spaces.Discrete(len(self._action_to_command))

        self.observation_space = spaces.Dict(
            {
                "program": spaces.Box(
                    low=-1,
                    high=len(self._action_to_command),
                    shape=(self.overall_max_program_len,),
                    dtype=np.int64,
                ),  # np.array of size program_len
                "data": spaces.Box(
                    low=-1,
                    high=max_data_len,
                    shape=(max_data_len,),
                    dtype=np.int64,
                ),  # np.array of size max_data_len
                "pointers": spaces.Box(
                    low=-1,
                    high=max_data_len,
                    shape=(max_data_len,),
                    dtype=np.int64,
                ),  # np.array of size max_data_len
                "stack": spaces.Box(
                    low=-1,
                    high=self.overall_max_program_len,
                    shape=(max_data_len,),
                    dtype=np.int64,
                ),  # np.array of size data_len
                "skipflag": spaces.Box(
                    low=0, high=2, shape=(1,), dtype=np.int64
                ),  # np.int64 (0 or 1)
                "commandpointer": spaces.Box(
                    low=0,
                    high=self.overall_max_program_len + 1,
                    shape=(1,),
                    dtype=np.int64,
                ),  # np.int64
                "last_action": spaces.Box(
                    low=0,
                    high=len(self._action_to_command),
                    shape=(1,),
                    dtype=np.int64,
                ),  # np.int64
                "execcost": spaces.Box(
                    low=0,
                    high=max_exec_cost_factor * max_data_len,
                    shape=(1,),
                    dtype=np.int64,
                ),  # np.int64
                "result": spaces.Box(
                    low=-1,
                    high=max_data_len,
                    shape=(max_data_len,),
                    dtype=np.int64,
                ),  # np.array of size data_len
                "pointersresult": spaces.Box(
                    low=-1,
                    high=max_data_len,
                    shape=(max_data_len,),
                    dtype=np.int64,
                ),  # np.array of size data_len
            }
        )

        self._initial_machine_state()

        logger.info(
            f"""Environment initialized, with: 
                    action masking: {self.do_action_masking}, 
                    reward function: {self.reward_function}, 
                    naive: {self.naive}, 
                    maximum data length: {self.max_data_len}, 
                    maximum program length: {self.overall_max_program_len}, 
                    maximum exec cost: {self.maximum_exec_cost}"""
        )

    def _initial_machine_state(self):
        self.program: np.array = np.array([], dtype=np.int64)
        self.data: np.array = np.array(
            list(range(1, self.current_data_len + 1))
        )
        self.pointers: list = [0]
        self.stack: list = []
        self.skipflag: bool = False
        self.commandpointer: int = 0
        self.last_action: int = len(self._action_to_command)
        self.execcost: int = 0
        self.result: np.array = np.zeros_like(self.data, dtype=int)
        self.pointersresult: list = [0]

        self.correct_tree = self._make_binary_tree()
        self.edge_distance_matrix = get_distance_matrix(self.current_data_len)
        self.correct_positions = {
            value: i for i, value in enumerate(self.correct_tree)
        }
        self.max_penalty = self.edge_distance_matrix.max() + 1

        self.written_numbers = []
        self.visited = np.zeros_like(self.data)
        self.last_reward = 0.0
        self.maximum_exec_cost = (
            self.max_exec_cost_factor * self.current_data_len
        )
        self.program_len = self.program_len_factor * self.current_data_len

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

    def _pad_ob(self, ob, max_length):
        return np.concatenate(
            (
                ob,
                np.ones((max_length - len(ob))) * self.pad_value,
            )
        ).astype(np.int64)

    def _get_obs(self) -> dict:
        program = self._pad_ob(self.program, self.overall_max_program_len)
        pointers = self._pad_ob(self.pointers, self.max_data_len)
        pointersresult = self._pad_ob(self.pointersresult, self.max_data_len)
        stack = self._pad_ob(self.stack, self.max_data_len)
        data = self._pad_ob(self.data, self.max_data_len)
        result = self._pad_ob(self.result, self.max_data_len)

        return {
            "program": program,
            "data": data,
            "pointers": pointers,
            "stack": stack,
            "skipflag": np.array([self.skipflag]).astype("int"),
            "commandpointer": np.array([self.commandpointer]),
            "last_action": np.array([self.last_action]),
            "execcost": np.array([self.execcost]),
            "result": result,
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

    def increase_data_len(self):
        if self.current_data_len < self.max_data_len:
            self.current_data_len += 1
        return self.reset()

    def increase_program_len(self):
        if self.program_len_factor < self.max_program_len_factor:
            self.program_len_factor += 1
        return self.reset()

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

        """
        Calculate Reward based on function set in config (not beautiful, sorry)
        """
        if self.reward_function == "new":
            new_reward = calculate_reward(
                self.result,
                self.edge_distance_matrix,
                self.max_penalty,
                self.visited,
                self.correct_positions,
                self.current_data_len,
            )
        else:
            new_reward = calculate_old_reward(
                solution_arr=self.correct_tree, candidate_arr=self.result
            )

        if self.incremental_reward:
            reward = new_reward - self.last_reward
            self.last_reward = new_reward
        else:
            reward = new_reward

        # If we terminate we give a bigger reward to
        # compensate for the early stop
        if terminated:
            reward = self.correct_reward_scale * (
                self.program_len - len(self.program) + 1
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
        print(
            "last_action: ",
            (
                self._action_nr_to_cmd_name[self.last_action]
                if self.last_action != len(self._action_to_command)
                else "PAD"
            ),
        )
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
        if not self.do_action_masking:
            return np.ones(len(self._action_to_command))
        elif self.naive:
            mask = np.zeros(len(self._action_to_command))
            mask[self._naive_actions] = 1
            return mask

        if self.last_action == len(self._action_to_command):
            mask = np.zeros(len(self._action_to_command))
            mask[self._valid_first_actions] = 1
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
            mask[[self._cmd_name_to_action_nr["left"]]] = 0

        elif self.pointers[-1] == len(self.data) - 1:
            mask[
                [
                    self._cmd_name_to_action_nr["right"],
                    self._cmd_name_to_action_nr["compareright"],
                ]
            ] = 0

        if len(self.pointersresult) == self.current_data_len - 1:
            mask[[self._cmd_name_to_action_nr["push"]]] = 0

        if (
            len(self.pointersresult) == 1
            or self.last_action == self._cmd_name_to_action_nr["push"]
        ):
            mask[[self._cmd_name_to_action_nr["pop"]]] = 0

        if len(self.stack) == self.current_data_len - 1:
            mask[[self._cmd_name_to_action_nr["mark"]]] = 0

        # Check that prior to a jump a conditional is checked
        if (
            len(self.stack) == 0
            or self.last_action not in self._conditional_actions
        ):
            mask[[self._cmd_name_to_action_nr["jump"]]] = 0

        if len(self.pointers) == 1:
            mask[[self._cmd_name_to_action_nr["isnotequal"]]] = 0

        if (
            len(self.stack) == 0
            or self.last_action == self._cmd_name_to_action_nr["mark"]
        ):
            mask[[self._cmd_name_to_action_nr["drop"]]] = 0

        if self.last_action != self._cmd_name_to_action_nr["compareright"]:
            mask[[self._cmd_name_to_action_nr["swapright"]]] = 0

        if self.pointersresult[-1] == 0:
            mask[[self._cmd_name_to_action_nr["parent"]]] = 0

        if self.pointersresult[-1] * 2 + 1 >= len(self.data):
            mask[
                [
                    self._cmd_name_to_action_nr["leftchild"],
                    self._cmd_name_to_action_nr["leftchildempty"],
                    self._cmd_name_to_action_nr["rightchild"],
                    self._cmd_name_to_action_nr["rightchildempty"],
                ]
            ] = 0

        elif self.pointersresult[-1] * 2 + 2 >= len(self.data):
            mask[
                [
                    self._cmd_name_to_action_nr["rightchild"],
                    self._cmd_name_to_action_nr["rightchildempty"],
                ]
            ] = 0

        if (
            self.result[self.pointersresult[-1]] != 0
            or self.data[self.pointers[-1]] in self.written_numbers
        ):
            mask[[self._cmd_name_to_action_nr["write"]]] = 0

        return mask

    # Available commands #

    def right(self):
        if self.pointers[-1] < len(self.data) - 1:
            self.pointers[-1] += 1
            self.visited[self.pointers[-1]] = 1

    def left(self):
        if self.pointers[-1] > 0:
            self.pointers[-1] -= 1

    def push(self):
        if len(self.pointersresult) < self.current_data_len - 1:
            self.pointersresult.append(self.pointersresult[-1])

    def pop(self):
        if len(self.pointersresult) > 1:
            self.pointersresult.pop()

    def mark(self):
        if len(self.stack) < self.current_data_len - 1:
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
            and self.last_action != self._cmd_name_to_action_nr["mark"]
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
        self.written_numbers.append(self.data[self.pointers[-1]])
        self.right()

    def istreeend(self):
        self.skipflag = self.pointersresult[-1] * 2 + 1 <= len(self.data) - 1

    def isnottreestart(self):
        self.skipflag = self.pointersresult[-1] == 0
