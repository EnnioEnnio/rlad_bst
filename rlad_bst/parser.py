import argparse

import yaml


def parse_arguments() -> dict:
    """
    Parse command-line arguments.

    Example usage:
    `python3 train.py --config path/to/config.yaml -dl 100`
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
        "--data-len",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--program-len",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--maximum-exec-cost",
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
