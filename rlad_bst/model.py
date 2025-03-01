from functools import partial
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import (
    MaskableCategoricalDistribution,
)
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.type_aliases import Schedule, TensorDict
from torch.distributions import Categorical
from torch.distributions.utils import lazy_property
from transformers import AutoConfig, AutoModel


class CustomMaskablePPO(MaskablePPO):
    """
    Custom maskable PPO model to change the policy networks
    """

    def __init__(self, *args, **kwargs):
        self.model_args = kwargs.pop("model_args", False)
        self.temperature = kwargs.pop("temperature")
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self.policy = CustomMaskableActorCriticPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
            model_args=self.model_args,
            temperature=self.temperature,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

    def predict(  # type: ignore[override]
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations.

        :param observation: the input observation
        :param state: The last hidden states (None, for recurrent policies)
        :param episode_start: The last masks (None, for recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(
            observation,
            state,
            episode_start,
            deterministic,
            action_masks=action_masks,
            should_log=True,
        )


class CustomMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """
    Custom maskable policy with custom mlp extractor, action net and value net
    """

    def __init__(self, *args, **kwargs):
        self.model_args = kwargs.pop("model_args")
        self.temperature = kwargs.pop("temperature")
        super().__init__(**kwargs)
        self.action_dist = CustomMaskableCategoricalDistribution(
            int(kwargs.get("action_space").n), temperature=self.temperature
        )
        self.custom_build(lr_schedule=kwargs.get("lr_schedule"))

    def custom_build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        if self.model_args["pretrained_encoder"] != "default":
            self.mlp_extractor = CustomExtractor(
                self.features_dim,
                pretrained_encoder=self.model_args["pretrained_encoder"],
                offset_size=self.features_extractor.offset_size,
            )

        if self.model_args["custom_action_net"]:
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=self.mlp_extractor.latent_dim_pi
            )
        if self.model_args["custom_value_net"]:
            self.value_net = CustomHead(self.mlp_extractor.latent_dim_vf, 1)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
        should_log=False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations.

        :param observation: the input observation
        :param state: The last states (None, for recurrent policies)
        :param episode_start: The last masks (None, for recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :param action_masks: Action masks to apply to the action distribution
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if (
            isinstance(observation, tuple)
            and len(observation) == 2
            and isinstance(observation[1], dict)
        ):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "  # noqa: E501
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "  # noqa: E501
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "  # noqa: E501
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"  # noqa: E501
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with torch.no_grad():
            if should_log:
                dist, log_dict = self.get_distribution(
                    obs_tensor, action_masks, should_log
                )
                actions = dist.get_actions(deterministic=deterministic)
            else:
                actions = self.get_distribution(
                    obs_tensor, action_masks, should_log
                ).get_actions(deterministic=deterministic)
            # Convert to numpy
            actions = actions.cpu().numpy()  # type: ignore[assignment]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, clip the actions
                # to avoid out of bound error
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        if should_log:
            log_dict["actions"] = actions
            return actions, state, log_dict
        return actions, state  # type: ignore[return-value]

    def get_distribution(
        self, obs, action_masks: Optional[np.ndarray] = None, should_log=False
    ):
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :param action_masks: Actions' mask
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution, log_dict = self._get_action_dist_from_latent(
            latent_pi, should_log=should_log
        )
        if action_masks is not None:
            additional_logs = distribution.apply_masking(action_masks)
            if should_log:
                log_dict.update(additional_logs)
        if should_log:
            log_dict["latent_pi"] = latent_pi
            return distribution, log_dict
        return distribution

    def _get_action_dist_from_latent(
        self, latent_pi: torch.Tensor, should_log=False
    ):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_logits = self.action_net(latent_pi)
        dist = self.action_dist.proba_distribution(
            action_logits=action_logits, should_log=should_log
        )
        if should_log:
            return dist, {"action_logits": action_logits}
        return dist


class CustomCombinedExtractor(CombinedExtractor):
    """
    Custom combined extractor, which flattens the observation space into one
    tensor including seperator tokens in between observations and adds a bias
    to differing observations
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space)
        self._features_dim += len(observation_space) - 1
        # We order the features in a semantically meaningful way
        self.custom_feature_order = [
            "data",
            "result",
            "pointers",
            "pointersresult",
            "program",
            "last_action",
            "skipflag",
            "stack",
            "commandpointer",
            "execcost",
        ]

        # Hard coded whether the offset should be increased after the features
        # based on the meaning of the features
        offset_increase = [
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            False,
        ]
        max_lengths = [
            observation_space[feature_name].high[0].item() + 1
            for feature_name in self.custom_feature_order
        ]
        # The offset is the possible values of all previous features
        # Beginning with the two special tokens
        self.offsets = []
        start_offset = 2
        for i, length in enumerate(max_lengths):
            self.offsets.append(start_offset)
            if offset_increase[i]:
                start_offset += length

        self.offset_size = self.offsets[-1] + max_lengths[-1]

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for i, feature_name in enumerate(self.custom_feature_order):
            feature = self.extractors[feature_name](observations[feature_name])
            feature = torch.where(
                feature == -1, torch.tensor(0), feature + self.offsets[i]
            )
            encoded_tensor_list.append(feature)
            encoded_tensor_list.append(
                torch.ones((feature.shape[0], 1), device=feature.device)
            )
        return torch.cat(encoded_tensor_list, dim=1)


class CustomExtractor(nn.Module):
    """
    Custom mlp extractor which uses a BERT-style encoder-only Transformer
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        pretrained_encoder: str = kwargs.get("pretrained_encoder")
        if pretrained_encoder.split("-")[0] == "jina":
            encoder_name = "jinaai/jina-embeddings-v2-small-en"
        else:
            raise ValueError(
                f"Pretrained encoder {pretrained_encoder} not supported"
            )

        if pretrained_encoder.split("-")[1] == "pretrained":
            self.encoder = AutoModel.from_pretrained(
                encoder_name, trust_remote_code=True
            )
        else:
            config = AutoConfig.from_pretrained(
                encoder_name, trust_remote_code=True
            )
            self.encoder = AutoModel.from_config(
                config, trust_remote_code=True
            )
        # The embedding matrix of the model does not make sense for our model
        # we will replace it by a embedding matrix of only 112 + 2 elements
        self.encoder.embeddings.word_embeddings = nn.Embedding(
            kwargs.get("offset_size"), self.encoder.config.hidden_size
        )
        self.latent_dim_pi = self.encoder.config.hidden_size
        self.latent_dim_vf = self.encoder.config.hidden_size

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.step(features)
        return latent, latent

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:

        return self.step(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.step(features)

    def step(self, features: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.where(
            features == 0, torch.tensor(0), torch.tensor(1)
        )
        return self.encoder(features.int(), attention_mask).pooler_output


class CustomMaskableCategoricalDistribution(MaskableCategoricalDistribution):
    """
    Custom maskable Categorical distribution to replace the action net
    """

    def __init__(self, *args, **kwargs):
        self.temperature = kwargs.pop("temperature")
        super().__init__(*args, **kwargs)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = CustomHead(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(
        self, action_logits: torch.Tensor, should_log=False
    ):
        # Restructure shape to align with logits
        reshaped_logits = action_logits.view(-1, self.action_dim)
        self.distribution = CustomMaskableCategorical(
            logits=reshaped_logits,
            temperature=self.temperature,
            should_log=should_log,
        )
        return self

    def apply_masking(self, masks) -> Optional[Dict[str, torch.Tensor]]:
        assert (
            self.distribution is not None
        ), "Must set distribution parameters"
        return self.distribution.apply_masking(masks)


def logits_to_probs(
    logits: torch.Tensor, temperature: float, should_log=False
) -> torch.Tensor:
    # Log logits and how they are scaled
    temp_logs = logits / temperature
    if should_log:
        temp_probs = torch.nn.functional.softmax(temp_logs, dim=-1)
        no_temp_probs = torch.nn.functional.softmax(logits, dim=-1)
        log_dict = {
            "temp_probs": temp_probs,
            "no_temp_probs": no_temp_probs,
        }
        return temp_probs, log_dict
    return torch.nn.functional.softmax(temp_logs, dim=-1), None


class CustomMaskableCategorical(Categorical):
    """
    Modified PyTorch Categorical distribution with support
    for invalid action masking.

    To instantiate, must provide either probs or logits, but not both.

    :param probs: Tensor containing finite non-negative values,
        which will be renormalized to sum to 1 along the last dimension.
    :param logits: Tensor of unnormalized log probabilities.
    :param validate_args: Whether or not to validate that arguments to methods
        like lob_prob() and icdf() match the distribution's shape, support ...
    :param masks: An optional boolean ndarray of compatible shape with the
        distribution. If True, the corresponding choice's logit value is
        preserved. If False, it is set to a large negative value,
        resulting in near 0 probability.
    """

    def __init__(
        self,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
        masks=None,
        temperature: float = 1.0,
        should_log=False,
    ):
        self.masks: Optional[torch.Tensor] = None
        super().__init__(probs, logits, validate_args)
        self._original_logits = self.logits
        self.temperature = temperature
        self.should_log = should_log
        self.apply_masking(masks)

    def apply_masking(self, masks) -> None:
        """
        Eliminate ("mask out") chosen categorical outcomes by setting their
        probability to 0.

        :param masks: An optional boolean ndarray of compatible shape with the
            distribution. If True, the corresponding choice's logit value is
            preserved. If False, it is set to a large negative value,
            resulting in near 0 probability. If masks is None, any previously
            applied masking is removed, and the original logits are restored.
        """

        if masks is not None:
            device = self.logits.device
            self.masks = torch.as_tensor(
                masks, dtype=torch.bool, device=device
            ).reshape(self.logits.shape)
            HUGE_NEG = torch.tensor(
                -1e8, dtype=self.logits.dtype, device=device
            )

            logits = torch.where(self.masks, self._original_logits, HUGE_NEG)
        else:
            self.masks = None
            logits = self._original_logits

        # Reinitialize with updated logits
        super().__init__(logits=logits)

        # self.probs may already be cached, so we must force an update
        self.probs, log_dict = logits_to_probs(
            self.logits,
            temperature=self.temperature,
            should_log=self.should_log,
        )
        return log_dict

    def entropy(self) -> torch.Tensor:
        if self.masks is None:
            return super().entropy()

        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy, since
        # masked actions possess no uncertainty.
        device = self.logits.device
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(
            self.masks, p_log_p, torch.tensor(0.0, device=device)
        )
        return -p_log_p.sum(-1)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, self.temperature)


class CustomHead(nn.Module):
    """
    Custom head for the policy network
    """

    def __init__(self, latent_dim: int, result_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, result_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


def get_model(
    env,
    verbose: bool,
    tensorboard_log,
    batch_size: int,
    ent_coef: float,
    model_args: dict,
    temperatur: float,
    learning_rate: float,
):
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
        activation_fn=nn.ReLU,
    )

    if model_args["pretrained_encoder"] != "default":
        policy_kwargs["features_extractor_class"] = CustomCombinedExtractor
    else:
        policy_kwargs["features_extractor_class"] = CombinedExtractor

    # Create the model
    model = CustomMaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        batch_size=batch_size,
        ent_coef=ent_coef,
        model_args=model_args,
        temperature=temperatur,
        learning_rate=learning_rate,
    )
    return model


def load_from_checkpoint(
    path,
    env,
    verbose: bool,
    tensorboard_log,
    batch_size: int,
    ent_coef: float,
    pretrained_encoder: str,
    learning_rate: float,
):
    model = get_model(
        env,
        verbose,
        tensorboard_log,
        batch_size,
        ent_coef,
        pretrained_encoder,
        temperatur=1.0,
        learning_rate=learning_rate,
    )
    model.set_parameters(path)
    return model
