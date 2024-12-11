from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO

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
            14: self.leftchild,
            15: self.rightchild,
            16: self.parent,
            17: self.leftchildempty,
            18: self.rightchildempty,
            19: self.write,
            20: self.nodeempty,
            21: self.istreeend,
            22: self.isnottreestart,
            23: self.giveresult,
        }

        self.action_space = spaces.Discrete(len(self._action_to_command))

        self.pad = 0

        self.observation_space = gym.spaces.Dict(
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
                    low=0, high=data_len, shape=(data_len,), dtype=np.int64
                ),  # np.array of size data_len
                "stack": spaces.Box(
                    low=0, high=program_len, shape=(data_len,), dtype=np.int64
                ),  # np.array of size data_len
                "skipflag": spaces.Discrete(2),  # np.int64 (0 or 1)
                "commandpointer": spaces.Discrete(program_len),  # np.int64
                "lastcommand": spaces.Discrete(
                    len(self._action_to_command) + 1
                ),  # np.int64
                "lastconditional": spaces.Discrete(
                    3
                ),  # np.int64 None (2), True (1), or False (0)
                "execcost": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.int64
                ),  # np.int64
                "storage": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.int64
                ),  # np.int64
                "result": spaces.Box(
                    low=0, high=np.inf, shape=(data_len,), dtype=np.int64
                ),  # np.array of size data_len
                "pointersresult": spaces.Box(
                    low=0, high=data_len, shape=(data_len,), dtype=np.int64
                ),  # np.array of size data_len
            }
        )

        self._initial_machine_state()
        self.correct_tree = self._make_binary_tree()

    def _initial_machine_state(self):
        self.program: np.array = np.array([], dtype=np.int64)
        self.data: np.array = np.array(list(range(self.data_len)))
        self.pointers: list = [0]
        self.stack: list = []
        self.skipflag: bool = False
        self.commandpointer: int = 0
        self.lastcommand: int = len(self._action_to_command)
        self.lastconditional: int = 2
        self.execcost: int = 0
        self.storage: int = self.pad
        self.result: np.array = np.zeros((len(self.data)), dtype=int)
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
            "skipflag": int(self.skipflag),
            "commandpointer": self.commandpointer,
            "lastcommand": self.lastcommand,
            "lastconditional": self.lastconditional,
            "execcost": np.array([self.execcost]),
            "storage": np.array([self.storage]),
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
        return (
            np.array_equal(self.data, self.correct_tree)
            or self.commandpointer == self.program_len
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
        while not (
            self._check_terminated()
            or self.commandpointer >= len(self.program)
        ):
            self._transition_once()
            if self.verbosity >= 1:
                self.render()

        reward = calculate_reward(
            solution_arr=self.correct_tree, candidate_arr=self.data
        )
        terminated = self._check_terminated()
        truncated = False  # Used to limit steps

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._get_info(),
        )

    def _transition_once(self):
        cmd = self.program[self.commandpointer]
        self.lastcommand = cmd
        self.commandpointer += 1

        self.lastconditional = 2  # Set to None
        if self.skipflag:
            self.skipflag = False
        else:
            self.execcost += 1
            self._action_to_command[cmd]()

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
        print("push:", self.pointersresult[-1])
        if len(self.pointersresult) < self.data_len - 1:
            self.pointersresult.append(self.pointersresult[-1])

    def pop(self):
        if len(self.pointersresult) > 1:
            self.pointersresult.pop()

    def pointernotempty(self):
        self.skipflag = len(self.pointers) <= 1

    def compare(self):
        self.skipflag = self.data[self.pointers[-1]] <= self.storage

    def mark(self):
        if len(self.stack) < self.data_len - 1:
            self.stack.append(self.commandpointer - 1)

    def jump(self):
        print("jump")
        if len(self.stack) != 0:
            self.commandpointer = self.stack.pop()

    def swap(self):
        temp = self.storage
        self.storage = self.data[self.pointers[-1]]
        self.data[self.pointers[-1]] = temp

    def isnotend(self):
        self.skipflag = self.pointers[-1] > len(self.data) - 1

    def isend(self):
        self.skipflag = self.pointers[-1] <= len(self.data) - 1

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
        print("left")
        if left_child < len(self.data):
            self.pointersresult[-1] = left_child

    def rightchild(self):
        right_child = self.pointersresult[-1] * 2 + 2
        print("right")
        if right_child < len(self.data):
            self.pointersresult[-1] = right_child

    def leftchildempty(self):
        left_child = self.pointersresult[-1] * 2 + 1
        if left_child >= len(self.data):
            self.skipflag = 1
        else:
            print("leftempty", left_child)
            self.skipflag = self.result[left_child] != 0

    def rightchildempty(self):
        right_child = self.pointersresult[-1] * 2 + 2
        if right_child >= len(self.data):
            self.skipflag = 1
        else:
            print("rightempty", right_child)
            self.skipflag = self.result[right_child] != 0

    def nodeempty(self):
        self.skipflag = self.result[self.pointersresult[-1]] != 0

    def parent(self):
        print("parent", int((self.pointersresult[-1] - 1) / 2))
        if self.pointersresult[-1] != 0:
            self.pointersresult[-1] = int((self.pointersresult[-1] - 1) / 2)

    def write(self):
        print("write", self.data[self.pointers[-1]])
        self.result[self.pointersresult[-1]] = self.data[self.pointers[-1]]
        self.right()

    def istreeend(self):
        self.skipflag = self.pointersresult[-1] * 2 + 1 <= len(self.data) - 1

    def isnottreestart(self):
        print(self.result)
        print("isnottreestart", self.pointersresult[-1])
        self.skipflag = self.pointersresult[-1] == 0

    def giveresult(self):
        print("here")
        self.data = self.result


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

    # wait_for_debugger()

    env = gym.make(
        "rlad/bst-v0",
        render_mode="human",
        data_len=10,
        program_len=100,
        maximum_exec_cost=100,
        verbosity=0,
    )

    check_env(env.unwrapped)

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10, progress_bar=True)

    print("Done")
    # TODO: Does not make much sense before we know our model can learn
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render()

    env.close()
