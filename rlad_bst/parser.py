import argparse

import yaml


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Example usage:
    python3 train.py --config path/to/config.yaml
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file to use.",
    )

    return parser.parse_args()


def load_config_from_yaml() -> dict:
    """
    Loads a configuration stored in a config.yaml file and returns it
    """
    args = parse_arguments()
    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(f"Error while parsing Config YAML file: {e}")
        return config
