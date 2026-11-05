"""Two cooperative PPO agents on one synchronized musculoskeletal rollout.

Independent rewards, GAE, critics, action likelihood ratios and KL stopping.
One Adam owns disjoint per-agent parameter groups; each agent's gradients are
clipped separately. This is not PPO on a summed 24-dimensional likelihood.
"""

from typing import NamedTuple

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance, obs_as_tensor


class AgentSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class CoAdaptiveRolloutBuffer(RolloutBuffer):
    def reset(self):
        super().reset()
        for name in ("rewards", "returns", "values", "log_probs", "advantages"):
            setattr(self, name, np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32))

    def add(self, obs, action, reward, episode_start, value, log_prob):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.detach().cpu().numpy()
        self.log_probs[self.pos] = log_prob.detach().cpu().numpy()
        self.pos += 1
        self.full = self.pos == self.buffer_size

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.detach().cpu().numpy()
        gae = np.zeros((self.n_envs, 2), dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            next_values = last_values if step == self.buffer_size - 1 else self.values[step + 1]
            next_start = dones if step == self.buffer_size - 1 else self.episode_starts[step + 1]
            mask = (1 - next_start.astype(np.float32))[:, None]
            delta = self.rewards[step] + self.gamma * next_values * mask - self.values[step]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            self.advantages[step] = gae
        self.returns = self.advantages + self.values

    def _get_samples(self, batch_inds, env=None):
        return AgentSamples(
            *[
                self.to_torch(getattr(self, name)[batch_inds])
                for name in ("observations", "actions", "values", "log_probs", "advantages", "returns")
            ]
        )


class CoAdaptivePPO(PPO):
    def _snapshot_agent_metrics(self):
        metrics = getattr(self, "latest_agent_metrics", {}).copy()
        metrics.update(
            {
                key: float(value)
                for key, value in self.logger.name_to_value.items()
                if key.startswith(("train/human_", "train/exo_", "rollout/human_", "rollout/exo_", "rollout/coadaptation_"))
            }
        )
        self.latest_agent_metrics = metrics

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("rollout_buffer_class", CoAdaptiveRolloutBuffer)
        super().__init__(*args, **kwargs)
        if self.use_sde:
            raise ValueError("Co-adaptation does not support SDE")

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if self._last_obs is None:
            raise ValueError("No previous observation")
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        agent_reward_log = []
        component_log = []
        for n_steps in range(n_rollout_steps):
            with th.no_grad():
                actions, values, log_probs = self.policy(obs_as_tensor(self._last_obs, self.device))
            actions = actions.cpu().numpy()
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            agent_rewards = np.asarray([info["agent_rewards"] for info in infos], dtype=np.float32)
            if agent_rewards.shape != (env.num_envs, 2) or not np.isfinite(agent_rewards).all():
                raise ValueError("Expected two finite agent rewards from every environment")
            agent_reward_log.append(agent_rewards.copy())
            component_log.extend(info["coadaptation"] for info in infos)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                    and "reference_exhausted" not in infos[idx].get("termination_reasons", [])
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0].cpu().numpy()
                    agent_rewards[idx] += self.gamma * terminal_value
            rollout_buffer.add(self._last_obs, actions, agent_rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones
        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        means = np.mean(agent_reward_log, axis=(0, 1))
        for i, agent in enumerate(("human", "exo")):
            self.logger.record(f"rollout/{agent}_reward_per_step", float(means[i]))
        for key in component_log[0]:
            self.logger.record(f"rollout/coadaptation_{key}", float(np.mean([row[key] for row in component_log])))
        self._snapshot_agent_metrics()
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip = self.clip_range(self._current_progress_remaining)
        clip_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None
        groups = self.policy.agent_parameters()
        active = [True, True]
        logs = {name: [] for name in ("pg", "value", "entropy", "kl", "clip", "total")}
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(batch.observations, batch.actions)
                advantages = batch.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean(0)) / (advantages.std(0) + 1e-8)
                ratio = th.exp(log_prob - batch.old_log_prob)
                policy_losses = -th.minimum(advantages * ratio, advantages * th.clamp(ratio, 1 - clip, 1 + clip)).mean(0)
                predicted = (
                    values if clip_vf is None else batch.old_values + th.clamp(values - batch.old_values, -clip_vf, clip_vf)
                )
                value_losses = ((predicted - batch.returns) ** 2).mean(0)
                entropy_losses = -entropy.mean(0)
                with th.no_grad():
                    log_ratio = log_prob - batch.old_log_prob
                    kls = ((th.exp(log_ratio) - 1) - log_ratio).mean(0)
                    if self.target_kl is not None:
                        active = [was_active and kl <= 1.5 * self.target_kl for was_active, kl in zip(active, kls.tolist())]
                if not any(active):
                    break
                losses = policy_losses + self.vf_coef * value_losses + self.ent_coef * entropy_losses
                loss = sum(losses[i] for i in range(2) if active[i])
                if not th.isfinite(loss):
                    raise FloatingPointError("Nonfinite cooperative PPO objective")
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for i in range(2):
                    if active[i]:
                        th.nn.utils.clip_grad_norm_(groups[i], self.max_grad_norm)
                    else:
                        # Adam momentum would still move a stopped agent with zero
                        # gradients; None skips its parameter updates entirely.
                        for parameter in groups[i]:
                            parameter.grad = None
                self.policy.optimizer.step()
                for key, value in (
                    ("pg", policy_losses),
                    ("value", value_losses),
                    ("entropy", entropy_losses),
                    ("kl", kls),
                    ("clip", (abs(ratio - 1) > clip).float().mean(0)),
                    ("total", losses),
                ):
                    logs[key].append(value.detach().cpu().numpy())
            self._n_updates += 1
            if not any(active):
                break
        names = {
            "pg": "policy_gradient_loss",
            "value": "value_loss",
            "entropy": "entropy_loss",
            "kl": "approx_kl",
            "clip": "clip_fraction",
            "total": "loss",
        }
        for key, name in names.items():
            if logs[key]:
                means = np.mean(logs[key], axis=0)
                self.logger.record(f"train/{name}", float(means.mean()))
                for i, agent in enumerate(("human", "exo")):
                    self.logger.record(f"train/{agent}_{name}", float(means[i]))
        explained = []
        for i, agent in enumerate(("human", "exo")):
            value = explained_variance(
                self.rollout_buffer.values[..., i].flatten(), self.rollout_buffer.returns[..., i].flatten()
            )
            explained.append(value)
            self.logger.record(f"train/{agent}_explained_variance", value)
            self.logger.record(f"train/{agent}_kl_stopped", float(not active[i]))
        self.logger.record("train/explained_variance", float(np.mean(explained)))
        with th.no_grad():
            self.policy.log_std.copy_(self.policy._distribution_log_std())
        self.logger.record("train/std", self.policy._distribution_log_std().exp().mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip)
        if clip_vf is not None:
            self.logger.record("train/clip_range_vf", clip_vf)
        self._snapshot_agent_metrics()
