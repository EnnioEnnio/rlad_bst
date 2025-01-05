from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

import wandb
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
            # 23: self.giveresult,
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
                    low=0, high=data_len + 1, shape=(data_len,), dtype=np.int64
                ),  # np.array of size data_len
                "stack": spaces.Box(
                    low=0,
                    high=program_len + 1,
                    shape=(data_len,),
                    dtype=np.int64,
                ),  # np.array of size data_len
                "skipflag": spaces.Discrete(2),  # np.int64 (0 or 1)
                "commandpointer": spaces.Discrete(program_len + 1),  # np.int64
                "lastcommand": spaces.Discrete(
                    len(self._action_to_command) + 1
                ),  # np.int64
                # "lastconditional": spaces.Discrete(
                #     3
                # ),  # np.int64 None (2), True (1), or False (0)
                "execcost": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.int64
                ),  # np.int64
                # "storage": spaces.Box(
                #     low=0, high=np.inf, shape=(1,), dtype=np.int64
                # ),  # np.int64
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
        self.lastcommand: int = len(self._action_to_command)
        # self.lastconditional: int = 2
        self.execcost: int = 0
        # self.storage: int = self.pad
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
            "skipflag": int(self.skipflag),
            "commandpointer": self.commandpointer,
            "lastcommand": self.lastcommand,
            # "lastconditional": self.lastconditional,
            "execcost": np.array([self.execcost]),
            # "storage": np.array([self.storage]),
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

        reward = (
            calculate_reward(
                solution_arr=self.correct_tree, candidate_arr=self.result
            )
            * self.execcost
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
        self.lastcommand = cmd
        self.commandpointer += 1

        # self.lastconditional = 2  # Set to None
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
        # print("lastconditional: ", self.lastconditional)
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
        if len(self.pointersresult) < self.data_len - 1:
            self.pointersresult.append(self.pointersresult[-1])

    def pop(self):
        if len(self.pointersresult) > 1:
            self.pointersresult.pop()

    def pointernotempty(self):
        self.skipflag = len(self.pointers) <= 1

    # def compare(self):
    #     self.skipflag = self.data[self.pointers[-1]] <= self.storage

    def mark(self):
        if len(self.stack) < self.data_len - 1:
            self.stack.append(self.commandpointer - 1)

    def jump(self):
        if len(self.stack) != 0:
            self.commandpointer = self.stack.pop()

    # def swap(self):
    #     temp = self.storage
    #     self.storage = self.data[self.pointers[-1]]
    #     self.data[self.pointers[-1]] = temp

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

    # def giveresult(self):
    #     self.data = self.result


# TODO: Would be nice in a different file:
# 1 file for env, 1 for training, 1 for config
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

    # TODO: Make args
    config = {
        "data_len": 7,
        "program_len": 64,  # Ours needs 51
        "maximum_exec_cost": 128,  # Ours needs for 7 data points 82
        "verbosity": 0,
        "total_timesteps": 250000,
        "gradient_save_freq": 100,
        "offline": True,
        "debug": False,
        "model_checkpoint": None,  # "models/x1vzt59t/model.zip"
    }
    if config["debug"]:
        wait_for_debugger()

    if config["offline"]:
        import os

        os.environ["WANDB_MODE"] = "dryrun"

    env = gym.make(
        "rlad/bst-v0",
        render_mode="human",
        data_len=config["data_len"],
        program_len=config["program_len"],
        maximum_exec_cost=config["maximum_exec_cost"],
        verbosity=config["verbosity"],
    )

    check_env(env.unwrapped)

    if not config["model_checkpoint"]:
        run = wandb.init(
            project="sb3",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,
            save_code=True,
        )
        env = Monitor(env)

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=config["verbosity"],
            tensorboard_log=f"runs/{run.id}",
        )
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=WandbCallback(
                gradient_save_freq=config["gradient_save_freq"],
                model_save_path=f"models/{run.id}",
                verbose=config["verbosity"],
            ),
        )

        run.finish()

    else:
        model = PPO.load(config["model_checkpoint"])
        obs, info = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
        print(obs["program"])

        env.close()
