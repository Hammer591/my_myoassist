"""Mirror optimisation must survive loading PPO and MirrorPPO checkpoints."""

import json

import numpy as np
import pytest
import torch

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.mirror_ppo import MirrorPPO
from rl_train.utils.train_log_handler import TrainLogHandler
from tools.tests.conftest import REPO_ROOT, build_env


CONFIG = REPO_ROOT / "rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_robustness_20M_20Nm.json"


@pytest.mark.parametrize("source_mirror", [0.0, 0.1])
@pytest.mark.parametrize("load_mode", ["resume", "explicit"])
def test_loaded_checkpoint_keeps_weights_and_runs_mirror_update(tmp_path, monkeypatch, source_mirror, load_mode):
    config, env = build_env(CONFIG)
    config.ppo_params.n_epochs = 1
    config.ppo_params.mirror_coef = source_mirror
    try:
        source = EnvironmentHandler.get_stable_baselines3_model(config, env)
        source.learn(total_timesteps=64)
        checkpoint = str(tmp_path / "checkpoint.zip")
        source.save(checkpoint)
        weights = {k: v.detach().clone() for k, v in source.policy.state_dict().items()}
        optimizer = source.policy.optimizer.state_dict()
        assert optimizer["state"]
        config.ppo_params.mirror_coef = 0.1
        if load_mode == "resume":
            config.env_params.prev_trained_policy_path = checkpoint
            resumed = EnvironmentHandler.get_stable_baselines3_model(config, env)
        else:
            resumed = EnvironmentHandler.get_stable_baselines3_model(config, env, trained_model_path=checkpoint)
        assert isinstance(resumed, MirrorPPO)
        assert resumed.mirror_coef == 0.1
        assert resumed.num_timesteps == source.num_timesteps
        for key, value in resumed.policy.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        restored_optimizer = resumed.policy.optimizer.state_dict()
        assert optimizer["param_groups"] == restored_optimizer["param_groups"]
        for key, state in optimizer["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(restored_optimizer["state"][key][name], value, rtol=0, atol=0)
        assert resumed._obs_perm.device == resumed.device
        np.testing.assert_array_equal(resumed._act_perm[resumed._act_perm].cpu(), np.arange(24))
        calls = []
        original = MirrorPPO._mirror_squared_error

        def capture(self, obs):
            calls.append(len(obs))
            return original(self, obs)

        monkeypatch.setattr(MirrorPPO, "_mirror_squared_error", capture)
        config.logger_params.logging_frequency = 1
        config.logger_params.evaluate_frequency = 1000
        handler = TrainLogHandler(str(tmp_path / "logs"))
        callback = EnvironmentHandler.get_callback(config, handler)
        resumed.learn(total_timesteps=128, reset_num_timesteps=False, callback=callback)
        assert calls
        record = json.loads((tmp_path / "logs/train_log.json").read_text())["log_datas"][-1]
        assert record["mirror_loss"] is not None and np.isfinite(record["mirror_loss"])
        assert record["mirror_loss_weighted"] == pytest.approx(0.1 * record["mirror_loss"])
        assert np.isfinite(record["mirror_loss_muscle"])
        assert np.isfinite(record["mirror_loss_device"])
        assert np.isfinite(record["mirror_device_grad_share"])
    finally:
        env.close()


def test_disabled_mirror_is_respected_when_loading_mirror_checkpoint(tmp_path):
    config, env = build_env(CONFIG)
    config.ppo_params.n_epochs = 1
    try:
        source = EnvironmentHandler.get_stable_baselines3_model(config, env)
        checkpoint = str(tmp_path / "mirror.zip")
        source.save(checkpoint)
        config.ppo_params.mirror_coef = 0
        config.env_params.prev_trained_policy_path = checkpoint
        resumed = EnvironmentHandler.get_stable_baselines3_model(config, env)
        assert not isinstance(resumed, MirrorPPO)
        assert resumed.mirror_coef == 0
    finally:
        env.close()


def test_zero_mirror_error_produces_finite_group_share():
    config, env = build_env(CONFIG)
    config.ppo_params.n_epochs = 1
    config.ppo_params.learning_rate = 0
    try:
        model = EnvironmentHandler.get_stable_baselines3_model(config, env)
        with torch.no_grad():
            for parameter in model.policy.parameters():
                parameter.zero_()
        model.learn(total_timesteps=64)
        assert model.logger.name_to_value["train/mirror_loss"] == 0
        assert model.logger.name_to_value["train/mirror_device_grad_share"] == 0
    finally:
        env.close()
