"""Compatible assisted actors with independent human/exo critics and action noise."""

import torch as th
from torch import nn

from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy


class CoAdaptiveActorCriticPolicy(HumanExoActorCriticPolicy):
    N_HUMAN = 22

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.action_space.shape != (24,):
            raise ValueError("Co-adaptation currently requires 22 muscle + 2 motor actions")
        self.log_std.requires_grad_(False)  # Compatibility snapshot; effective noise below.
        self.human_log_std = nn.Parameter(self.log_std[:22].detach().clone())
        self.exo_log_std = nn.Parameter(self.log_std[22:].detach().clone())
        n_obs = self.policy_network.network_index_handler.get_observation_num("exo_actor_r")
        self.exo_value_net = nn.Sequential(nn.Linear(n_obs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.exo_value_net.apply(self.init_weights)
        self.to(self.device)
        schedule = args[2] if len(args) > 2 else kwargs["lr_schedule"]
        self.rebuild_optimizer(schedule(1))

    def agent_parameters(self):
        net = self.policy_network
        human = list(net.human_policy_net.parameters()) + [self.human_log_std] + list(net.value_net.parameters())
        exo_nets = {id(v): v for k, v in net.actor_nets.items() if k.startswith("exo_actor")}
        exo = (
            [p for v in exo_nets.values() for p in v.parameters()] + [self.exo_log_std] + list(self.exo_value_net.parameters())
        )
        if set(map(id, human)) & set(map(id, exo)):
            raise ValueError("Human and exo optimization parameters must be disjoint")
        return human, exo

    def rebuild_optimizer(self, learning_rate):
        groups = self.agent_parameters()
        self.optimizer = self.optimizer_class(
            [{"params": groups[0], "name": "human"}, {"params": groups[1], "name": "exo"}],
            lr=learning_rate,
            **self.optimizer_kwargs,
        )

    def _distribution_log_std(self):
        human = self.human_log_std
        if self.policy_network.frozen_human_structural_symmetry:
            perm = self.policy_network.human_symmetry_act_perm
            human = 0.5 * (human + human[perm])
        return th.cat((human, self.exo_log_std))

    def predict_values(self, obs):
        human = self.policy_network.forward_critic(obs)
        exo_obs = self.policy_network.network_index_handler.map_observation_to_network(obs, "exo_actor_r")
        return th.cat((human, self.exo_value_net(exo_obs)), dim=-1)

    def evaluate_actions(self, obs, actions):
        distribution = self.get_distribution(obs).distribution
        per_dim = distribution.log_prob(actions)
        ent = distribution.entropy()
        log_prob = th.stack((per_dim[:, :22].sum(-1), per_dim[:, 22:].sum(-1)), dim=-1)
        entropy = th.stack((ent[:, :22].sum(-1), ent[:, 22:].sum(-1)), dim=-1)
        return self.predict_values(obs), log_prob, entropy

    def forward(self, obs, deterministic=False):
        actions = self.get_distribution(obs).get_actions(deterministic=deterministic)
        values, log_probs, _ = self.evaluate_actions(obs, actions)
        return actions, values, log_probs
