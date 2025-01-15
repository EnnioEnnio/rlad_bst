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
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback

import rlad_bst.bst_sort_machine  # noqa: F401 # for env registration only
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

    env = gym.make(
        "rlad/bst-v0",
        render_mode="human",
        data_len=config.get("data_len", 7),
        program_len=config.get("program_len", 64),
        maximum_exec_cost=config.get("maximum_exec_cost", 128),
        verbosity=config.get("verbosity", 0),
        early_stop_delta=config.get("early_stop_delta", 0.01),
        early_stop_patience=config.get("early_stop_patience", 5),
    )

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
        env = Monitor(env)

        model = get_model(
            env,
            config["verbosity"],
            f"runs/{run.id}",
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
                    CheckpointCallback(
                        save_freq=10_000,
                        save_path=f"/mnt/vast-nfs/rlad/checkpoints/{run.id}",
                        name_prefix="rlad_bst",
                    ),
                ]
            ),
        )

        run.finish()

    else:
        # Otherwise, load a previously trained model
        model = load_from_checkpoint(
            config["model_checkpoint"], env, config["verbosity"], None
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
