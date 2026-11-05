"""Exercise real two-agent updates, transfer, independent GAE and checkpoint resume."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch as th
import gymnasium as gym
from gymnasium.spaces import Box

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.coadaptive_ppo import CoAdaptivePPO, CoAdaptiveRolloutBuffer
from rl_train.train.frozen_human import freeze_human_actor
from rl_train.train.policies.rl_agent_coadaptive import CoAdaptiveActorCriticPolicy
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from tools.tests.conftest import REPO_ROOT

CONFIG = REPO_ROOT / "rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_coadaptive_power_10M.json"


def test_agent_gae_is_independent_and_stops_at_episode_boundary():
    buf = CoAdaptiveRolloutBuffer(2, Box(-1, 1, (1,)), Box(-1, 1, (24,)), gamma=0.9, gae_lambda=1, n_envs=2)
    zero = th.zeros(2, 2)
    buf.add(np.zeros((2, 1)), np.zeros((2, 24)), [[1, 10], [3, 30]], [True, True], zero, zero)
    buf.add(np.zeros((2, 1)), np.zeros((2, 24)), [[2, 20], [4, 40]], [False, True], zero, zero)
    buf.compute_returns_and_advantage(th.ones(2, 2) * 1000, np.array([True, True]))
    np.testing.assert_allclose(buf.returns, [[[2.8, 28], [3, 30]], [[2, 20], [4, 40]]], rtol=1e-6)
    sample = next(buf.get(4))
    assert sample.advantages.shape == sample.old_log_prob.shape == (4, 2)


@pytest.fixture
def assisted_source(tmp_path):
    config = DictionableDataclass.create(ExoImitationTrainSessionConfig, json.loads(CONFIG.read_text()))
    config.env_params.num_envs = 1
    config.env_params.robustness.enabled = False
    config.env_params.coadaptation.enabled = False
    config.env_params.coadaptation.init_path = ""
    config.ppo_params.n_steps = config.ppo_params.batch_size = 64
    config.ppo_params.n_epochs = 2
    env = EnvironmentHandler.create_environment(config, False)
    try:
        source = EnvironmentHandler.get_stable_baselines3_model(config, env)
        source.num_timesteps = 123456
        freeze_human_actor(source.policy)
        checkpoint = tmp_path / "source" / "trained_models" / "model_123456.zip"
        checkpoint.parent.mkdir(parents=True)
        source.save(checkpoint)
        (checkpoint.parent.parent / "session_config.json").write_text(json.dumps(DictionableDataclass.to_dict(config)))
        yield config, env, source, checkpoint
    finally:
        env.close()


def snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def changed(before, module):
    return any(not th.equal(value, module.state_dict()[key]) for key, value in before.items())


def test_transfer_two_agent_updates_and_resume(assisted_source, tmp_path):
    config, _, source, checkpoint = assisted_source
    config.env_params.coadaptation.enabled = True
    config.env_params.coadaptation.init_path = str(checkpoint)
    env = EnvironmentHandler.create_environment(config, False)
    try:
        model = EnvironmentHandler.get_stable_baselines3_model(config, env)
        assert isinstance(model, CoAdaptivePPO)
        assert model.num_timesteps == 0
        assert model.coadaptation_provenance["source_timesteps"] == 123456
        assert all(p.requires_grad for p in model.policy.policy_network.human_policy_net.parameters())
        assert getattr(model.policy, "_frozen_human_log_std_hook", None) is None
        obs = th.randn(13, 76)
        with th.no_grad():
            th.testing.assert_close(
                model.policy.policy_network.forward_actor(obs), source.policy.policy_network.forward_actor(obs), rtol=0, atol=0
            )
            th.testing.assert_close(model.policy._distribution_log_std(), source.policy.log_std, rtol=0, atol=0)
        actions, values, log_prob = model.policy(obs)
        assert values.shape == log_prob.shape == (13, 2)
        modified = actions.detach().clone()
        modified[:, 22:] += 0.3
        _, modified_log_prob, _ = model.policy.evaluate_actions(obs, modified)
        th.testing.assert_close(log_prob[:, 0], modified_log_prob[:, 0], rtol=0, atol=0)
        assert not th.equal(log_prob[:, 1], modified_log_prob[:, 1])
        model.policy.optimizer.zero_grad(set_to_none=True)
        modified_log_prob[:, 0].sum().backward()
        assert all(p.grad is None or th.count_nonzero(p.grad) == 0 for p in model.policy.agent_parameters()[1])
        human = snapshot(model.policy.policy_network.human_policy_net)
        exo = snapshot(model.policy.policy_network.exo_policy_net_right)
        hv, ev = snapshot(model.policy.policy_network.value_net), snapshot(model.policy.exo_value_net)
        noise = model.policy.human_log_std.detach().clone()
        model.learn(128, progress_bar=False)
        assert changed(human, model.policy.policy_network.human_policy_net)
        assert changed(exo, model.policy.policy_network.exo_policy_net_right)
        assert changed(hv, model.policy.policy_network.value_net) and changed(ev, model.policy.exo_value_net)
        assert not th.equal(noise, model.policy.human_log_std)
        effective = model.policy._distribution_log_std()[:22]
        th.testing.assert_close(effective, effective[model.policy.policy_network.human_symmetry_act_perm], rtol=0, atol=0)
        # KL-stopped agents must also stay fixed under existing Adam momentum.
        before_stop = [p.detach().clone() for p in model.policy.agent_parameters()[0]]
        evaluate = model.policy.evaluate_actions

        def high_human_kl(observations, actions):
            value, probs, entropy = evaluate(observations, actions)
            return value, probs + probs.new_tensor([1.0, 0.0]), entropy

        model.policy.evaluate_actions = high_human_kl
        model.train()
        for before, parameter in zip(before_stop, model.policy.agent_parameters()[0]):
            th.testing.assert_close(before, parameter, rtol=0, atol=0)
        model.policy.evaluate_actions = evaluate
        output = tmp_path / "coadapt" / "trained_models" / "model_128.zip"
        output.parent.mkdir(parents=True)
        model.save(output)
        (output.parent.parent / "session_config.json").write_text(json.dumps(DictionableDataclass.to_dict(config)))
        observed, _ = env.reset(seed=1234)
        expected, _ = model.predict(observed, deterministic=True)
        restored = EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path=str(output))
        actual, _ = restored.predict(observed, deterministic=True)
        np.testing.assert_array_equal(actual, expected)
        assert restored.num_timesteps == 128
        assert len(restored.policy.optimizer.state) == len(model.policy.optimizer.state)
        # Resume uses its own saved actors; the old warm-start archive is not needed.
        checkpoint.unlink()
        resumed_config = deepcopy(config)
        resumed_config.env_params.coadaptation.init_path = ""
        resumed_config.env_params.prev_trained_policy_path = str(output)
        resumed = EnvironmentHandler.get_stable_baselines3_model(resumed_config, env)
        before = snapshot(resumed.policy.policy_network.human_policy_net)
        resumed.learn(64, reset_num_timesteps=False, progress_bar=False)
        assert resumed.num_timesteps == 192 and changed(before, resumed.policy.policy_network.human_policy_net)
    finally:
        env.close()


def test_agent_reward_split_and_history_reset(assisted_source):
    config, _, _, _ = assisted_source
    config.env_params.coadaptation.enabled = True
    env = EnvironmentHandler.create_environment(config, False)
    try:
        env.reset(seed=1234)
        action = np.zeros(24)
        action[22:] = [0.2, -0.3]
        _, human, _, _, info = env.step(action)
        terms = info["coadaptation"]
        assert terms["action_smoothness"] == 0
        assert terms["action_magnitude"] == pytest.approx(-env.dt * 0.01 * (0.2**2 + 0.3**2))
        np.testing.assert_allclose(
            info["agent_rewards"],
            [
                human,
                sum(
                    terms[k]
                    for k in [
                        "effort_proxy",
                        "action_smoothness",
                        "action_magnitude",
                        "power",
                        "human_coupling",
                        "torque_excess",
                    ]
                ),
            ],
        )
        action[22:] = [-0.4, 0.5]
        info = env.step(action)[-1]
        assert info["coadaptation"]["action_smoothness"] < 0
        env.reset(seed=1234)
        assert env.step(action)[-1]["coadaptation"]["action_smoothness"] == 0
    finally:
        env.close()


def test_reject_incompatible_source_and_fake_metabolic_model(assisted_source):
    config, env, _, checkpoint = assisted_source
    config.env_params.coadaptation.enabled = True
    config.env_params.coadaptation.init_path = str(checkpoint)
    config.env_params.reference_motion_mode = "legacy_scaled"
    with pytest.raises(ValueError, match="reference_motion_mode"):
        EnvironmentHandler.get_stable_baselines3_model(config, env)
    config.env_params.coadaptation.effort_model = "uchida"
    with pytest.raises(ValueError, match="not Uchida"):
        config.env_params.coadaptation.validate()


@pytest.mark.parametrize("reason,bootstrap", [("time_limit", True), ("reference_exhausted", False), ("height", False)])
def test_both_critics_bootstrap_only_timeouts(assisted_source, reason, bootstrap):
    _, _, source, _ = assisted_source

    class ShortEnv(gym.Env):
        observation_space = source.observation_space
        action_space = source.action_space

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(76, dtype=np.float32), {}

        def step(self, action):
            info = dict(
                agent_rewards=np.array([1.0, 10.0], dtype=np.float32),
                coadaptation={"effort_proxy": 0.0},
                termination_reasons=[reason],
            )
            return np.zeros(76, dtype=np.float32), 1.0, reason == "height", reason != "height", info

    env = ShortEnv()
    model = CoAdaptivePPO(
        CoAdaptiveActorCriticPolicy, env, policy_kwargs=source.policy_kwargs, n_steps=2, batch_size=2, n_epochs=1, gamma=0.9
    )
    with th.no_grad():
        for parameter in model.policy.policy_network.value_net.parameters():
            parameter.zero_()
        model.policy.policy_network.value_net[-1].bias.fill_(2.0)
        for parameter in model.policy.exo_value_net.parameters():
            parameter.zero_()
        model.policy.exo_value_net[-1].bias.fill_(3.0)
    model.learn(2, progress_bar=False)
    expected = np.array([1.0, 10.0]) + (0.9 * np.array([2.0, 3.0]) if bootstrap else 0)
    np.testing.assert_allclose(model.rollout_buffer.rewards[:, 0, :], np.tile(expected, (2, 1)), rtol=1e-6)
    assert "rollout/human_reward_per_step" in model.latest_agent_metrics
    assert model.latest_agent_metrics["rollout/exo_reward_per_step"] == 10.0
    model.env.close()
