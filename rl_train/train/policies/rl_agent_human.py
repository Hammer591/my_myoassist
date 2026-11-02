from rl_train.train.policies.rl_agent_base import BasePPOCustomNetwork, BaseCustomActorCriticPolicy
import torch as th
from torch import nn
from gymnasium import spaces
import rl_train.train.train_configs.config as myoassist_config
import rl_train.train.train_configs.config_imitation as myoassist_config_imitation


class HumanPPOCustomNetwork(BasePPOCustomNetwork):
    def __init__(self, observation_space, action_space, custom_policy_params):
        super().__init__(observation_space, action_space, custom_policy_params)
        self.structural_symmetry = custom_policy_params.structural_symmetry
        if self.structural_symmetry:
            for name, permutation, size in (
                ("symmetry_obs_perm", custom_policy_params.symmetry_observation_permutation, observation_space.shape[0]),
                ("symmetry_act_perm", custom_policy_params.symmetry_action_permutation, action_space.shape[0]),
            ):
                perm = th.as_tensor(permutation, dtype=th.long)
                if perm.shape != (size,) or not th.equal(perm.sort().values, th.arange(size)):
                    raise ValueError(f"{name} must be a full permutation of length {size}")
                if not th.equal(perm[perm], th.arange(size)):
                    raise ValueError(f"{name} must be an involution")
                self.register_buffer(name, perm)
            if th.equal(self.symmetry_act_perm, th.arange(action_space.shape[0])):
                raise ValueError("Structural symmetry must exchange left/right actions")

    def forward_actor(self, obs: th.Tensor) -> th.Tensor:
        if not self.structural_symmetry:
            return self.policy_net(obs)
        # Group average over identity and left/right reflection, with shared weights.
        # Mirrored observations include all reference horizons, not only current pose.
        action = self.policy_net(obs)
        mirrored = self.policy_net(obs[:, self.symmetry_obs_perm])[:, self.symmetry_act_perm]
        return 0.5 * (action + mirrored)

    def forward_critic(self, obs: th.Tensor) -> th.Tensor:
        return self.value_net(obs)

    def reset_policy_networks(self):
        """Reset both networks to their initial state"""
        layers = []
        last_dim = self.observation_space.shape[0]

        for dim in self.net_arch["human_actor"]:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.Tanh())
            last_dim = dim

        layers.append(nn.Linear(last_dim, self.action_space.shape[0]))
        layers.append(nn.Tanh())
        self.policy_net = nn.Sequential(*layers)

    def reset_value_network(self):
        value_layers = []
        value_last_dim = self.observation_space.shape[0]

        for dim in self.net_arch["common_critic"]:
            value_layers.append(nn.Linear(value_last_dim, dim))
            value_layers.append(nn.Tanh())
            value_last_dim = dim

        value_layers.append(nn.Linear(value_last_dim, 1))

        self.value_net = nn.Sequential(*value_layers)


class HumanActorCriticPolicy(BaseCustomActorCriticPolicy):
    def _distribution_log_std(self) -> th.Tensor:
        network = self.policy_network
        if network.structural_symmetry:
            # Both raw pair parameters receive the same gradient. The effective noise
            # is exactly shared even if their raw values were initialized differently.
            return 0.5 * (self.log_std + self.log_std[network.symmetry_act_perm])
        return self.log_std

    def _get_custom_policy_type(self):
        return myoassist_config_imitation.ImitationTrainSessionConfig.PolicyParams.CustomPolicyParams

    def _build_policy_network(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        custom_policy_params: myoassist_config.TrainSessionConfigBase.PolicyParams.CustomPolicyParams,
    ) -> BasePPOCustomNetwork:
        return HumanPPOCustomNetwork(observation_space, action_space, custom_policy_params)

    # Custom reset network
    def reset_network(self, reset_shared_net: bool = False, reset_policy_net: bool = False, reset_value_net: bool = False):
        """Reset the networks if specified"""
        if reset_policy_net:
            self.policy_network.reset_policy_networks()
        if reset_value_net:
            self.policy_network.reset_value_network()
