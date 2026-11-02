"""A human-only checkpoint must transfer intact and remain fixed during assisted PPO."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch as th

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass

CONFIG_DIR = Path(__file__).resolve().parents[2] / "rl_train/train/train_configs"
HUMAN = CONFIG_DIR / "imitation_22_NoDevice_hip35_reference_30M.json"
EXO = CONFIG_DIR / "imitation_22_PhotoHipExo_L1_frozen_human_hip35_30M.json"


def configured(path, cls):
    config = DictionableDataclass.create(cls, json.loads(path.read_text()))
    config.env_params.num_envs = 1
    config.env_params.robustness.enabled = False
    config.ppo_params.n_steps = config.ppo_params.batch_size = 64
    config.ppo_params.n_epochs = 1
    return config


def checkpoint(tmp_path):
    config = configured(HUMAN, ImitationTrainSessionConfig)
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
    path = tmp_path / "trained_models" / "human.zip"
    path.parent.mkdir()
    try:
        model = EnvironmentHandler.get_stable_baselines3_model(config, env)
        model.save(path)
        (tmp_path / "session_config.json").write_text(HUMAN.read_text())
        return path, model
    finally:
        env.close()


def test_frozen_human_transfer_training_and_resume(tmp_path):
    path, human = checkpoint(tmp_path)
    config = configured(EXO, ExoImitationTrainSessionConfig)
    config.env_params.human_actor_init_path = str(path)
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
    try:
        model = EnvironmentHandler.get_stable_baselines3_model(config, env)
        assert env.observation_space.shape == (76,)
        assert env.action_space.shape == (24,)
        policy = model.policy
        actor = policy.policy_network.human_policy_net
        obs = th.as_tensor(env.reset()[0], dtype=th.float32).unsqueeze(0)
        with th.no_grad():
            th.testing.assert_close(actor(obs), human.policy.policy_network.policy_net(obs), rtol=0, atol=0)
        assert all(not parameter.requires_grad for parameter in actor.parameters())
        before = {name: value.clone() for name, value in actor.state_dict().items()}
        human_noise = policy.log_std[:22].detach().clone()
        exo_before = {name: value.clone() for name, value in policy.policy_network.exo_policy_net_right.state_dict().items()}
        assert th.allclose(policy.log_std[22:], th.full((2,), -2.0))
        assert th.allclose(policy.policy_network.exo_policy_net_right[-2].weight, th.zeros_like(policy.policy_network.exo_policy_net_right[-2].weight))
        model.learn(total_timesteps=128, progress_bar=False)
        for name, value in before.items():
            th.testing.assert_close(actor.state_dict()[name], value, rtol=0, atol=0)
        th.testing.assert_close(policy.log_std[:22], human_noise, rtol=0, atol=0)
        assert any(not th.equal(exo_before[name], value) for name, value in policy.policy_network.exo_policy_net_right.state_dict().items())
        assisted = tmp_path / "assisted.zip"
        model.save(assisted)
        restored = EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path=str(assisted))
        assert all(not parameter.requires_grad for parameter in restored.policy.policy_network.human_policy_net.parameters())
        restored.learn(total_timesteps=64, progress_bar=False)
        th.testing.assert_close(restored.policy.log_std[:22], human_noise, rtol=0, atol=0)
        assert np.isfinite(restored.predict(env.reset()[0], deterministic=True)[0]).all()
    finally:
        env.close()


def test_missing_checkpoint_and_wrong_source_fail_loudly(tmp_path):
    config = configured(EXO, ExoImitationTrainSessionConfig)
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
    try:
        with pytest.raises(ValueError, match="human_actor_init_path"):
            EnvironmentHandler.get_stable_baselines3_model(config, env)
        path, _ = checkpoint(tmp_path)
        config.env_params.human_actor_init_path = str(path)
        session = tmp_path / "session_config.json"
        source = json.loads(session.read_text())
        source["env_params"]["reference_observation_steps"] = [2, 4]
        session.write_text(json.dumps(source))
        with pytest.raises(ValueError, match="incompatible reference_observation_steps"):
            EnvironmentHandler.get_stable_baselines3_model(config, env)
    finally:
        env.close()
