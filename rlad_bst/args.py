from dataclasses import dataclass

from simple_parsing import field


@dataclass(kw_only=True)
class TrainingArgs:
    """
    Argument class for training configuration using simple_parsing.

    Example usage:
        from simple_parsing import parse
        args = parse(TrainingArgs, add_config_path_arg=True)
    """

    run_name: str = field(default="standard_run_")

    # Env params
    start_data_len: int = field(default=7)
    max_data_len: int = field(default=7)
    start_program_len_factor: int = field(default=9)
    max_program_len_factor: int = field(default=9)
    max_exec_cost_factor: int = field(default=18)
    verbosity: int = field(default=0)
    do_action_masking: bool = field(default=True)
    reward_function: str = field(
        default="new", help='Choose between "old" or "new"'
    )
    naive: bool = field(default=False)

    # Train params
    total_timesteps: int = field(default=1_000_000)
    gradient_save_freq: int = field(default=100)
    batch_size: int = field(default=64)
    model_checkpoint: str | None = field(
        default=None, help="Absolute path (str) to model checkpoint"
    )
    learning_rate: float = field(default=0.00003)
    temperature: float = field(default=2.0)
    entropy_coefficient: float = field(default=0.1)

    # Callback params
    grow_data: bool = field(default=False)
    grow_program_len: bool = field(default=False)
    delta: float = field(default=0.5)
    patience: int = field(default=2)
    eval_interval: int = field(default=10_000)

    # Model params
    pretrained_encoder: str = field(
        default="jina-pretrained",
        help='Choose from "jina-pretrained", "jina-not-pretrained", "default"',
    )
    use_custom_value_net: bool = field(default=True)
    use_custom_action_net: bool = field(default=True)

    # Debug params
    offline: bool = field(default=False)
    debug: bool = field(default=False)
