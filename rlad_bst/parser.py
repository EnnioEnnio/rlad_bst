"""
This module provides functionality for parsing configuration settings for a training run.

It supports:
1. Loading default configuration settings from a YAML file.
2. Overriding specific configuration parameters via command-line arguments.

Example:
    `poetry run python3 train.py --config path/to/config.yaml --data-len 100`

Command-Line Arguments:
- `--config` (str, required): Path to the YAML configuration file.
- `--data-len` (int, optional): Length of the dataset.
- `--program-len` (int, optional): Length of the program.
- `--maximum-exec-cost` (int, optional): Maximum execution cost allowed.
- `--verbosity` (int, optional): Verbosity level of the program.
- `--total-timesteps` (int, optional): Total number of timesteps for training.
- `--gradient-save-freq` (int, optional): Frequency at which gradients are saved.
- `--offline` (bool, optional): Whether the program runs offline.
- `--debug` (bool, optional): Whether debugging is enabled.
- `--model-checkpoint` (str, optional): Path to the model checkpoint file.
"""  # noqa: E501

import argparse

import yaml


def parse_arguments() -> dict:
    """
    Parse command-line arguments.

    Example usage:
    `poetry run python3 train.py --config path/to/config.yaml --data-len 100`
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="""
        Path to the YAML configuration file to use.\n
        This YAML is required, you may overwrite specific parameters using command line.
        """,  # noqa: E501
    )

    parser.add_argument(
        "--start-data-len",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--max-data-len",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--max-program-len-factor",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--max-exec-cost-factor",
        type=int,
        required=False,
    )

    parser.add_argument("--verbosity", type=int, required=False)

    parser.add_argument(
        "--total-timesteps",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--gradient-save-freq",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--offline",
        type=bool,
        required=False,
    )

    parser.add_argument(
        "--debug",
        type=bool,
        required=False,
    )

    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=False,
        help="Absolute path (str) to model checkpoint",
    )

    parser.add_argument(
        "--entropy-coefficient",
        type=float,
        required=False,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--patience",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--delta",
        type=float,
        required=False,
    )

    return vars(parser.parse_args())


def load_config_from_yaml() -> dict:
    """
    Loads a configuration stored in a config.yaml file and
    merges it with optional command-line arguments.

    Command-line arguments overwrite YAML values.
    """
    args = parse_arguments()

    with open(args["config"], "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(f"Error while parsing Config YAML file: {e}")

    for key, value in args.items():
        if key == "config":
            continue
        if value is not None:
            config_key = key.replace("-", "_")
            config[config_key] = value

    return config
