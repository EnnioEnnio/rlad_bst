"""
This script sets up and runs a reinforcement learning training pipeline using
the Stable-Baselines3 (SB3) PPO algorithm and integrates with the
Weights & Biases (wandb) library for experiment tracking.

It leverages a custom environment registered under `rlad/bst-v0` and uses a
configuration system that reads parameters from
a YAML file and optionally overrides them with command-line arguments.

Usage:
1. **Train a New Model**:
    `poetry run python3 train.py --config path/to/config.yaml`
2. **Debug Mode**:
    `poetry run python3 train.py --config path/to/config.yaml --debug true`
3. **Run with Pre-Trained Model**:
    `poetry run python3 train.py --config path/to/config.yaml --model-checkpoint path/to/model.zip`
"""  # noqa: E501

import gymnasium as gym
import wandb
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback

import rlad_bst.bst_sort_machine  # noqa: F401 # for env registration only
from rlad_bst.helpers import GrowDataLenCallback
from rlad_bst.model import get_model, load_from_checkpoint
from rlad_bst.parser import load_config_from_yaml

WANDB_PROJECT = "rlad_bst"
WANDB_ENTITY = "rlad_bst"


def wait_for_debugger(port: int = 56785):
    """
    Pauses the program until a remote debugger is attached.
    Should only be called on rank0.
    """

    import debugpy

    debugpy.listen(("0.0.0.0", port))
    print(f"Waiting for client to attach on port {port}... ")
    debugpy.wait_for_client()


def main():
    config: dict = load_config_from_yaml()
    set_random_seed(42)

    if config.get("debug", False):
        wait_for_debugger()

    if config.get("offline", False):
        import os

        os.environ["WANDB_MODE"] = "dryrun"

    start_data_len = config.get("start_data_len", 3)
    print(
        f"Load enviroment with array of size {start_data_len}, max program length {config.get('max_program_len_factor', 10) * start_data_len} and max exec cost {config.get('max_exec_cost_factor', 20) * start_data_len}"  # noqa: E501
    )
    if config["grow_data"]:
        print(
            f"Growing activated: Checking to grow data length every {config['eval_interval']} steps with patience {config['patience']} and delta {config['delta']}"  # noqa: E501
        )
        print(f"Maximum data length is {config.get('max_data_len', 7)}")
    else:
        print("Growing deactivated")
        assert start_data_len == config.get(
            "max_data_len", 7
        ), "Without growing start and max data length must be the same"

    env_config = {
        "id": "rlad/bst-v0",
        "render_mode": "human",
        "max_data_len": config.get("max_data_len", 7),
        "start_data_len": start_data_len,
        "max_program_len_factor": config.get("max_program_len_factor", 10),
        "max_exec_cost_factor": config.get("max_exec_cost_factor", 20),
        "do_action_masking": config.get("do_action_masking", False),
        "verbosity": config.get("verbosity", 0),
    }

    env = gym.make(**env_config)

    eval_env = gym.make(**env_config)

    check_env(env.unwrapped)

    # If we do NOT have a model checkpoint, train a model
    if not config.get("model_checkpoint"):
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,
            save_code=True,
        )
        wandb.define_metric("event_step")
        wandb.define_metric("Event Distribution", step_metric="event_step")
        env = Monitor(env)
        eval_env = Monitor(eval_env)

        model = get_model(
            env,
            config["verbosity"],
            f"runs/{run.id}",
            config["batch_size"],
            config["entropy_coefficient"],
            config["pretrained_encoder"],
            config["temperature"],
        )

        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=CallbackList(
                [
                    WandbCallback(
                        gradient_save_freq=config["gradient_save_freq"],
                        model_save_path=f"models/{run.id}",
                        verbose=config["verbosity"],
                    ),
                    GrowDataLenCallback(
                        n_steps=config["eval_interval"],
                        eval_env=eval_env,
                        patience=config["patience"],
                        delta=config["delta"],
                        checkpoint_path=f"checkpoints/{run.id}",
                        grow_data=config["grow_data"],
                    ),
                ]
            ),
        )

        run.finish()

    else:
        # Otherwise, load a previously trained model
        model = load_from_checkpoint(
            config["model_checkpoint"],
            env,
            config["verbosity"],
            None,
            config["batch_size"],
            0.0,
            config["pretrained_encoder"],
        )
        obs, info = env.reset()
        terminated, truncated = False, False
        c = 0
        while not terminated and not truncated:
            action, _states = model.predict(
                obs,
                deterministic=True,
                action_masks=env.unwrapped.action_masks(),
            )
            obs, reward, terminated, truncated, info = env.step(action)
            print(env.unwrapped._action_nr_to_cmd_name[action.item()])
            print(f"Reward: {reward}")
            print(f"Result: {obs['result']}")
            c += 1
        program = [
            env.unwrapped._action_nr_to_cmd_name[cmd] if cmd != -1 else "PAD"
            for cmd in obs["program"]
        ]
        print("\n".join(program))
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Result: {env.unwrapped.result}")

        env.close()


if __name__ == "__main__":
    main()
