"""Exact equivariance must hold for inference, PPO likelihoods, and saved policies."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.policies.rl_agent_human import HumanPPOCustomNetwork
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from tools.tests.conftest import REPO_ROOT


CONFIG = REPO_ROOT / "rl_train/train/train_configs/imitation_22_NoDevice_hip35_knee3_smooth_symmetric_30M.json"


def make_config():
    config = DictionableDataclass.create(ImitationTrainSessionConfig, json.loads(CONFIG.read_text()))
    config.env_params.num_envs = 1
    config.env_params.robustness.enabled = False
    config.ppo_params.n_steps = 32
    config.ppo_params.batch_size = 32
    config.ppo_params.n_epochs = 1
    return config


@pytest.fixture
def model_env():
    config = make_config()
    env = EnvironmentHandler.create_environment(config, False)
    try:
        yield config, env, EnvironmentHandler.get_stable_baselines3_model(config, env)
    finally:
        env.close()


def assert_equivariant(policy):
    network = policy.policy_network
    obs = torch.randn(41, policy.observation_space.shape[0], device=policy.device)
    mirrored = obs[:, network.symmetry_obs_perm]
    action = network.forward_actor(obs)
    mirror_action = network.forward_actor(mirrored)[:, network.symmetry_act_perm]
    torch.testing.assert_close(action, mirror_action, rtol=0, atol=1e-7)
    # Avoid a vacuous test: the underlying unconstrained MLP is not itself symmetric.
    assert torch.max(abs(network.policy_net(obs) - network.policy_net(mirrored)[:, network.symmetry_act_perm])) > 1e-4
    return obs, mirrored, action


def test_exact_actor_and_deterministic_prediction(model_env):
    _, _, model = model_env
    obs, mirrored, expected = assert_equivariant(model.policy)
    action, _ = model.predict(obs.cpu().numpy(), deterministic=True)
    counterpart, _ = model.predict(mirrored.cpu().numpy(), deterministic=True)
    np.testing.assert_allclose(action, expected.detach().cpu().numpy(), atol=1e-7)
    np.testing.assert_allclose(action, counterpart[:, model.policy.policy_network.symmetry_act_perm.cpu()], atol=1e-7)


def test_exploration_likelihood_and_entropy_share_mirror_pairs(model_env):
    _, _, model = model_env
    policy = model.policy
    with torch.no_grad():
        policy.log_std.copy_(torch.linspace(-2, 0, len(policy.log_std), device=policy.device))
    obs, mirrored, _ = assert_equivariant(policy)
    perm = policy.policy_network.symmetry_act_perm
    effective = policy._distribution_log_std()
    torch.testing.assert_close(effective, effective[perm], rtol=0, atol=0)
    actions = torch.randn(len(obs), len(perm), device=policy.device)
    _, logp, entropy = policy.evaluate_actions(obs, actions)
    _, mirror_logp, mirror_entropy = policy.evaluate_actions(mirrored, actions[:, perm])
    torch.testing.assert_close(logp, mirror_logp, atol=2e-5, rtol=1e-6)
    torch.testing.assert_close(entropy, mirror_entropy, atol=2e-5, rtol=1e-6)
    (-logp.mean()).backward()
    assert torch.isfinite(policy.log_std.grad).all()
    torch.testing.assert_close(policy.log_std.grad, policy.log_std.grad[perm], atol=1e-6, rtol=1e-6)
    # The action sampling path uses the same effective standard deviations.
    policy(obs)
    std = policy.action_dist.distribution.stddev
    torch.testing.assert_close(std, std[:, perm], rtol=0, atol=0)


def test_ppo_update_save_load_and_resume_preserve_symmetry(model_env, tmp_path):
    config, env, model = model_env
    before = [p.detach().clone() for p in model.policy.policy_network.policy_net.parameters()]
    model.learn(total_timesteps=64)
    assert model._n_updates > 0
    assert any(not torch.equal(p, q) for p, q in zip(before, model.policy.policy_network.policy_net.parameters()))
    assert_equivariant(model.policy)
    path = str(tmp_path / "symmetric.zip")
    model.save(path)
    # Plain SB3 loading must work without rebuilding maps from external files.
    loaded = PPO.load(path, env=env)
    for key, value in model.policy.state_dict().items():
        torch.testing.assert_close(value, loaded.policy.state_dict()[key], rtol=0, atol=0)
    assert_equivariant(loaded.policy)
    config.env_params.prev_trained_policy_path = path
    resumed = EnvironmentHandler.get_stable_baselines3_model(config, env)
    assert_equivariant(resumed.policy)
    resumed.learn(total_timesteps=32, reset_num_timesteps=False)
    assert resumed.num_timesteps == 96
    assert_equivariant(resumed.policy)


def test_old_architecture_requires_explicit_new_training(model_env, tmp_path):
    config, env, symmetric = model_env
    config.policy_params.custom_policy_params.structural_symmetry = False
    old = EnvironmentHandler.get_stable_baselines3_model(config, env)
    old_path = str(tmp_path / "old.zip")
    old.save(old_path)
    loaded_old = EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path=old_path)
    assert not loaded_old.policy.policy_network.structural_symmetry
    config.policy_params.custom_policy_params.structural_symmetry = True
    with pytest.raises(ValueError, match="structural_symmetry differs"):
        EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path=old_path)
    symmetric_path = str(tmp_path / "symmetric.zip")
    symmetric.save(symmetric_path)
    config.policy_params.custom_policy_params.structural_symmetry = False
    with pytest.raises(ValueError, match="structural_symmetry differs"):
        EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path=symmetric_path)


def test_frozen_transfer_requires_assisted_policy(model_env, tmp_path):
    from rl_train.train.frozen_human import initialize_frozen_human_actor

    config, _, model = model_env
    path = str(tmp_path / "symmetric.zip")
    model.save(path)
    with pytest.raises(TypeError, match="HumanExoActorCriticPolicy"):
        initialize_frozen_human_actor(SimpleNamespace(device="cpu", policy=None), path, config.env_params)


@pytest.mark.parametrize("bad", [[0], [0, 0], [1, 2, 0]])
def test_invalid_permutations_rejected(bad):
    from gymnasium import spaces

    params = ImitationTrainSessionConfig.PolicyParams.CustomPolicyParams()
    params.net_arch = {"human_actor": [8], "common_critic": [8]}
    params.structural_symmetry = True
    params.symmetry_observation_permutation = bad
    params.symmetry_action_permutation = [1, 0]
    with pytest.raises(ValueError, match="permutation|involution"):
        HumanPPOCustomNetwork(spaces.Box(-1, 1, shape=(3,)), spaces.Box(-1, 1, shape=(2,)), params)
