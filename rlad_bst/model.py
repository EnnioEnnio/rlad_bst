from functools import partial

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.distributions import (
    MaskableCategoricalDistribution,
)
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.type_aliases import Schedule
from transformers import AutoModel


class CustomMaskablePPO(MaskablePPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self.policy = CustomMaskableActorCriticPolicy(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)


class CustomMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dist = CustomMaskableCategoricalDistribution(
            int(args[1].n)
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.mlp_extractor = CustomExtractor(self.features_dim)

        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=self.mlp_extractor.latent_dim_pi
        )
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


class CustomExtractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v2-small-en", trust_remote_code=True
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
        return self.encoder(features.int()).pooler_output


class CustomMaskableCategoricalDistribution(MaskableCategoricalDistribution):
    def __init__(self, *args, **kwargs):
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
        action_logits = CustomHead(latent_dim, self.action_dim.n)
        return action_logits


class CustomHead(nn.Module):
    def __init__(self, latent_dim: int, result_dim: int):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, result_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.value_net(latent)


def get_model(env, verbose, tensorboard_log):
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
        activation_fn=nn.ReLU,
        features_extractor_class=CombinedExtractor,
    )

    # Create the model
    model = CustomMaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        batch_size=256,
    )
    return model
