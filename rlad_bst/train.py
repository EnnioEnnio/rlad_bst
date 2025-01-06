import gymnasium as gym
import wandb
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

# importing for env registration only
import rlad_bst.sort_machine_env  # noqa: F401
from rlad_bst.parser import load_config_from_yaml


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


def main():
    config: dict = load_config_from_yaml()

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
    )

    check_env(env.unwrapped)

    # If we do NOT have a model checkpoint, train a model
    if not config.get("model_checkpoint"):
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
            verbose=config.get("verbosity", 0),
            tensorboard_log=f"runs/{run.id}",
        )
        model.learn(
            total_timesteps=config.get("total_timesteps"),
            callback=WandbCallback(
                gradient_save_freq=config.get("gradient_save_freq"),
                model_save_path=f"models/{run.id}",
                verbose=config.get("verbosity"),
            ),
        )

        run.finish()

    else:
        # Otherwise, load a previously trained model
        model = PPO.load(config.get("model_checkpoint"))
        obs, info = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
        print(obs["program"])

        env.close()


if __name__ == "__main__":
    main()
