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

import time

import gymnasium as gym
import wandb
from gymnasium.utils.env_checker import check_env
from print_on_steroids import logger
from simple_parsing import parse
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback

import rlad_bst.bst_sort_machine  # noqa: F401 # for env registration only

# from rlad_bst.parser import load_config_from_yaml
from rlad_bst.args import TrainingArgs
from rlad_bst.helpers import GrowDataLenCallback
from rlad_bst.model import get_model, load_from_checkpoint

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
    # config: dict = load_config_from_yaml()
    args = parse(TrainingArgs, add_config_path_arg=True)
    set_random_seed(42)
    logger.info(f"Training with args: {args}")
    if args.debug:
        wait_for_debugger()

    if args.offline:
        import os

        os.environ["WANDB_MODE"] = "dryrun"

    start_data_len = args.start_data_len
    logger.info(
        f"Load enviroment with array of size {start_data_len}, max program length {args.max_program_len_factor * start_data_len} and max exec cost {args.max_exec_cost_factor * start_data_len}"  # noqa: E501
    )
    if args.grow_data:
        logger.info(
            f"Growing activated: Checking to grow data length every {args.eval_interval} steps with patience {args.patience} and delta {args.delta}"  # noqa: E501
        )
        logger.info(f"Maximum data length is {args.max_data_len}")
    else:
        logger.info("Growing deactivated")
        assert (
            start_data_len == args.max_data_len
        ), "Without growing start and max data length must be the same"

    env_config = {
        "id": "rlad/bst-v0",
        "render_mode": None,
        "max_data_len": args.max_data_len,
        "start_data_len": args.start_data_len,
        "start_program_len_factor": args.start_program_len_factor,
        "max_program_len_factor": args.max_program_len_factor,
        "max_exec_cost_factor": args.max_exec_cost_factor,
        "do_action_masking": args.do_action_masking,
        "verbosity": args.verbosity,
        "reward_function": args.reward_function,
        "naive": args.naive,
        "correct_reward_scale": args.correct_reward_scale,
    }

    env = gym.make(**env_config)

    eval_env = gym.make(**env_config)

    check_env(env.unwrapped)

    model_args = {
        "pretrained_encoder": args.pretrained_encoder,
        "custom_value_net": args.use_custom_value_net,
        "custom_action_net": args.use_custom_action_net,
    }

    # If we do NOT have a model checkpoint, train a model
    if not args.model_checkpoint:
        run_name = args.run_name
        run_name = run_name + str(time.time()) if run_name else None

        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=args,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,
            save_code=True,
            name=run_name,
        )
        wandb.define_metric("event_step")
        wandb.define_metric("Event Distribution", step_metric="event_step")
        env = Monitor(env)
        eval_env = Monitor(eval_env)

        ppo_model = get_model(
            env=env,
            verbose=args.verbosity,
            tensorboard_log=f"runs/{run.id}",
            batch_size=args.batch_size,
            ent_coef=args.entropy_coefficient,
            model_args=model_args,
            temperature=args.temperature,
            learning_rate=args.learning_rate,
        )

        ppo_model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList(
                [
                    WandbCallback(
                        gradient_save_freq=args.gradient_save_freq,
                        model_save_path=f"models/{run.name}",
                        verbose=args.verbosity,
                    ),
                    GrowDataLenCallback(
                        n_steps=args.eval_interval,
                        eval_env=eval_env,
                        patience=args.patience,
                        delta=args.delta,
                        checkpoint_path=f"checkpoints/{run.name}",
                        grow_data=args.grow_data,
                        grow_program_len=args.grow_program_len,
                    ),
                ]
            ),
        )

        run.finish()

    else:
        # Otherwise, load a previously trained model
        ppo_model = load_from_checkpoint(
            args.model_checkpoint,
            env,
            args.verbosity,
            None,
            args.batch_size,
            0.0,
            model_args,
        )
        obs, info = env.reset()
        terminated, truncated = False, False
        c = 0
        while not terminated and not truncated:
            action, _states = ppo_model.predict(
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
