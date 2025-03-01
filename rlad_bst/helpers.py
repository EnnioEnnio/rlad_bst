import os
import warnings
from typing import Union

import gymnasium as gym
import numpy as np
import wandb
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
)


class GrowDataLenCallback(BaseCallback):
    """
    The callback evaluates the model every n steps.
    If grow data is activated and solve the problem we increase the data_len.
    If we switch data len or get worse we save the model.
    When evaluating the model we log interesting metrics.
    """

    def __init__(
        self,
        grow_data: bool,
        grow_program_len: int,
        n_steps: int,
        eval_env: Union[gym.Env, VecEnv],
        patience: int,
        delta: float,
        checkpoint_path: str,
    ):
        super().__init__()
        # When to trigger
        self.n_steps = n_steps
        self.last_time_trigger = 0

        # Evaluation in an early stopping fashion
        self.data_mean_reward = -np.inf
        self.program_mean_reward = -np.inf
        self.last_save_reward = -np.inf
        self.patience = patience
        self.delta = delta
        self.data_wait = 0
        self.program_wait = 0
        self.eval_env = eval_env
        self._action_nr_to_cmd_name = eval_env.unwrapped._action_nr_to_cmd_name
        self.action_names = list(self._action_nr_to_cmd_name.values())
        self.first_step_action_names = [
            self._action_nr_to_cmd_name[action]
            for action in eval_env.unwrapped._valid_first_actions
        ]
        self.env_does_action_masking = eval_env.unwrapped.do_action_masking

        self.action_table = wandb.Table(
            columns=["event_step", "reward", "program", "results"]
        )

        # Checkpoint saving
        self.checkpoint_path = checkpoint_path
        self.checkpoint_buffer = None
        self.first_step_logs = []

        self.grow_data = grow_data
        self.grow_program_len = grow_program_len

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps

            # Save model
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)

            reward, episodes_terminated, log_dict = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=5,
                return_episode_terminated=True,
            )
            mean_reward = np.mean(reward)
            self._delete_previous_checkpoint_if_needed(mean_reward)
            if self.grow_data and self._check_if_increase_data_len(
                mean_reward, episodes_terminated
            ):
                self.model.get_env().env_method("increase_data_len")
                self.eval_env.unwrapped.increase_data_len()
                print(
                    "Data len increased to: ",
                    self.model.get_env().get_attr("current_data_len"),
                )
            if self.grow_program_len and self._check_if_increase_program_len(
                mean_reward, episodes_terminated
            ):
                self.model.get_env().env_method("increase_program_len")
                self.eval_env.unwrapped.increase_program_len()
                print(
                    "Program len increased to: ",
                    self.model.get_env().get_attr("program_len"),
                )
            self.first_step_logs.append(log_dict["first_step_log"])
            self._log_data(log_dict)
            self.checkpoint_buffer = model_path

        return True

    def _on_training_end(self) -> None:
        wandb.log({"val/action_table": self.action_table})

    def _log_data(self, log_dict):
        wandb_log = {
            f"val/action_{action}": log_dict["actions"][i].item()
            for i, action in enumerate(self.action_names)
        }
        wandb_log["event_step"] = self.num_timesteps
        if self.env_does_action_masking:
            wandb_log.update(
                {
                    f"val/first_action_{action}": self.first_step_logs[-1][
                        "temp_probs"
                    ][self.first_step_logs[-1]["temp_probs"] != 0][i].item()
                    for i, action in enumerate(self.first_step_action_names)
                }
            )
        else:
            wandb_log.update(
                {
                    f"val/first_action_{action}": self.first_step_logs[-1][
                        "temp_probs"
                    ][i].item()
                    for i, action in enumerate(self.action_names)
                }
            )
        if len(self.first_step_logs) > 1:
            wandb_log.update(
                {
                    "val/L2_distance_between_first_probs": (
                        (
                            self.first_step_logs[-1]["temp_probs"]
                            - self.first_step_logs[-2]["temp_probs"]
                        )
                        ** 2
                    )
                    .sum()
                    .cpu(),
                    "val/L2_distance_first_latent_pi": (
                        (
                            self.first_step_logs[-1]["latent_pi"]
                            - self.first_step_logs[-2]["latent_pi"]
                        )
                        ** 2
                    )
                    .sum()
                    .cpu(),
                }
            )
        wandb.log(wandb_log)

        program = [
            self._action_nr_to_cmd_name[cmd]
            for cmd in log_dict["first_step_log"]["program"]
            if cmd > 0
        ]
        self.action_table.add_data(
            self.num_timesteps,
            self.first_step_logs[-1]["episode_reward"],
            ", ".join(program),
            str(self.first_step_logs[-1]["final_result"]),
        )

    def _delete_previous_checkpoint_if_needed(
        self, mean_reward: float
    ) -> None:
        if mean_reward + self.delta > self.last_save_reward:
            # The previous checkpoint was worse so we delete it
            if self.checkpoint_buffer is not None:
                print("Deleting previous checkpoint: ", self.checkpoint_buffer)
                os.remove(self.checkpoint_buffer)
            self.last_save_reward = mean_reward

    def _check_if_increase_data_len(
        self, mean_reward: float, episodes_terminated: float
    ) -> bool:
        """
        We increase the data lenght if all episodes terminated
        and the reward is not getting better.
        """
        if not len(episodes_terminated) == sum(episodes_terminated):
            print("Not all episodes terminated: ", episodes_terminated)
            return False
        if mean_reward > self.data_mean_reward + self.delta:
            self.data_mean_reward = mean_reward
            self.data_wait = 0
            print("New best mean reward: ", mean_reward)
        else:
            self.data_wait += 1
            print("Data wait: ", self.data_wait)
            if self.data_wait >= self.patience:
                self.data_wait = 0
                self.data_mean_reward = mean_reward
                return True
        return False

    def _check_if_increase_program_len(
        self, mean_reward: float, episodes_terminated: float
    ) -> bool:
        """
        We increase the program length if the model has solved the subproblem,
        meaning the reward is not getting better but also not worse.
        """
        if (
            mean_reward > self.program_mean_reward - self.delta
            and mean_reward < self.program_mean_reward + self.delta
        ):
            self.program_wait += 1
            print("Program wait: ", self.program_wait)
            if self.program_wait >= self.patience:
                self.program_wait = 0
                return True
        else:
            self.program_mean_reward = mean_reward
            self.program_wait = 0
            print("New best mean reward: ", mean_reward)
        return False

    def _checkpoint_path(
        self, checkpoint_type: str = "", extension: str = ""
    ) -> str:
        """
        Taken from CheckpointCallback
        Helper to get checkpoint path for each type of checkpoint.
        """
        return os.path.join(
            self.checkpoint_path,
            f"{checkpoint_type}{self.num_timesteps}_steps.{extension}",
        )


def evaluate_policy(
    model: MaskablePPO,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = False,
    render: bool = False,
    return_episode_terminated: bool = False,
    warn: bool = True,
) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
    """
    Modified from https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/maskable/evaluation.py # noqa: E501
    """

    is_monitor_wrapped = False

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper."
            " This may result in reporting modified episode lengths and "
            "rewards, if other wrappers happen to modify these."
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )
    episode_rewards = []
    episode_lengths = []
    episodes_terminated = []
    log_dicts = []

    current_reward = 0.0
    current_length = 0.0
    episode_count = 0
    observations = env.reset()
    states = None
    episode_start = True
    first_step_log = None
    per_episode_log_dicts = []
    while episode_count < n_eval_episodes:
        action_masks = get_action_masks(env)
        actions, state, log_dict = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_start,
            deterministic=deterministic,
            action_masks=action_masks,
        )
        observations, reward, done, infos = env.step(actions)
        per_episode_log_dicts.append(log_dict)
        current_reward += reward
        current_length += 1
        if done:
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            episodes_terminated.append(not infos[0]["TimeLimit.truncated"])
            if episode_count == 0:
                first_step_log = per_episode_log_dicts[0]
                first_step_log["final_result"] = infos[0][
                    "terminal_observation"
                ]["result"]
                first_step_log["episode_reward"] = infos[0]["episode"]["r"]
                first_step_log["program"] = infos[0]["terminal_observation"][
                    "program"
                ]
            episode_count += 1
            current_reward = 0
            current_length = 0
            log_dicts.append(per_episode_log_dicts)
            per_episode_log_dicts = []

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    actions = np.bincount(
        [
            log["actions"].item()
            for episode_logs in log_dicts
            for log in episode_logs
        ],
        minlength=model.action_space.n,
    )
    trace = [log["actions"].item() for log in log_dicts[0]]
    log_dict = {
        "first_step_log": first_step_log,
        "actions": actions,
        "trace": trace,
    }
    if return_episode_terminated:
        return episode_rewards, episodes_terminated, log_dict
    return mean_reward, std_reward, log_dict
