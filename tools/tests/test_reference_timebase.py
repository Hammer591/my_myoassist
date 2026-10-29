"""Reference targets must share one physical clock in rewards, resets and replay."""

from copy import deepcopy
import json
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.mirror_ppo import MirrorPPO
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils.reference_data import resample_reference_data
from rl_train.utils.train_log_handler import TrainLogHandler
from tools.tests.conftest import REPO_ROOT


CONFIG = REPO_ROOT / "rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_robustness_20M_20Nm.json"


def make_config(mode="fixed_rate"):
    config = DictionableDataclass.create(ExoImitationTrainSessionConfig, json.loads(CONFIG.read_text()))
    config.env_params.num_envs = 1
    config.env_params.robustness.enabled = False
    config.env_params.reference_motion_mode = mode
    config.env_params.trajectory_curriculum.enabled = False
    config.ppo_params.n_steps = config.ppo_params.batch_size = 64
    config.ppo_params.n_epochs = 1
    return config


def linear_reference():
    time = np.arange(1003) / 500
    return {
        "metadata": {"sample_rate": 500, "data_length": 999999},
        "series_data": {
            "q_knee_angle_l": 0.1 + 2 * time,
            "dq_knee_angle_l": np.full(len(time), 2.0),
            "dq_pelvis_tx": 1 + 0.2 * np.sin(time),
        },
    }


def test_fixed_grid_matches_analytic_motion_without_endpoint_stretch():
    raw = linear_reference()
    original = deepcopy(raw)
    fixed = resample_reference_data(raw, 30, "fixed_rate")
    n = fixed["metadata"]["resampled_data_length"]
    np.testing.assert_allclose(fixed["series_data"]["q_knee_angle_l"], 0.1 + 2 * np.arange(n) / 30, atol=1e-14)
    np.testing.assert_allclose(np.diff(fixed["series_data"]["q_knee_angle_l"]) * 30, 2.0, atol=1e-12)
    np.testing.assert_allclose(fixed["series_data"]["dq_knee_angle_l"], 2.0)
    assert n == 61
    assert raw["metadata"] == original["metadata"]
    for key in raw["series_data"]:
        np.testing.assert_array_equal(raw["series_data"][key], original["series_data"][key])
    again = resample_reference_data(fixed, 30, "fixed_rate")
    np.testing.assert_array_equal(again["series_data"]["q_knee_angle_l"], fixed["series_data"]["q_knee_angle_l"])
    legacy = resample_reference_data(raw, 30)
    expected = np.interp(np.linspace(0, 1002, 60), np.arange(1003), original["series_data"]["q_knee_angle_l"])
    np.testing.assert_array_equal(legacy["series_data"]["q_knee_angle_l"], expected)


@pytest.mark.parametrize("mode", ["fixed_rate", "legacy_scaled"])
def test_reset_reward_and_reference_replay_use_same_velocity(mode):
    config = make_config(mode)
    env = EnvironmentHandler.create_environment(config, False, is_evaluate_mode=True)
    try:
        for seed in [0, 1, 1234]:
            env.reset(seed=seed)
            idx = env._imitation_index
            series = env._reference_data["series_data"]
            factor = 1 if mode == "fixed_rate" else env._target_velocity / series["dq_pelvis_tx"][idx]
            assert env.sim.data.joint("pelvis_tx").qvel[0] == 1.25
            for joint in config.env_params.reference_data_keys:
                if joint != "pelvis_tx":
                    assert env.sim.data.joint(joint).qvel[0] == pytest.approx(series["dq_" + joint][idx] * factor)
            np.testing.assert_allclose(list(env._get_qvel_diff().values()), 0, atol=1e-12)
        env.imitation_step(True, specific_index=42)
        assert env.sim.data.joint("knee_angle_l").qvel[0] == pytest.approx(env._get_reference_qvel("knee_angle_l", 42))
        if mode == "fixed_rate":
            # Changing the speed task must not secretly change a fixed-clock
            # internal-joint target. Zero reference root speed is safe as well.
            velocity = env._get_reference_qvel("knee_angle_l", 42)
            env._target_velocity = 0.8
            env._reference_data["series_data"]["dq_pelvis_tx"][42] = 0
            with np.errstate(divide="raise", invalid="raise"):
                assert env._get_reference_qvel("knee_angle_l", 42) == velocity
                np.testing.assert_allclose(list(env._get_qvel_diff().values()), 0, atol=1e-12)
    finally:
        env.close()


def test_evaluator_and_training_share_reference_loader_and_cache_modes(tmp_path):
    config = make_config()
    training = EnvironmentHandler.load_reference_data(config)
    evaluator = SimpleNamespace(session_config=config)
    ImitationGaitEvaluator.load_reference_data(evaluator)
    for key in training["series_data"]:
        np.testing.assert_array_equal(training["series_data"][key], evaluator.ref_data_dict["series_data"][key])
    path = tmp_path / "cache.json"
    metadata = {"evaluation": {"mode": "relaxed", "terminate_when_done": True}, "episodes": [{}]}
    path.write_text(json.dumps({"metadata": metadata}))
    assert ImitationGaitEvaluator.is_single_episode_cache(str(path), "relaxed", "legacy_scaled")
    assert not ImitationGaitEvaluator.is_single_episode_cache(str(path), "relaxed", "fixed_rate")
    metadata["evaluation"]["reference_motion_mode"] = "fixed_rate"
    path.write_text(json.dumps({"metadata": metadata}))
    assert ImitationGaitEvaluator.is_single_episode_cache(str(path), "relaxed", "fixed_rate")
    path.write_text('{"metadata":')
    assert not ImitationGaitEvaluator.is_single_episode_cache(str(path), "relaxed", "fixed_rate")


def test_legacy_checkpoint_resumes_with_fixed_clock_and_same_layout(tmp_path):
    old_config = make_config("legacy_scaled")
    old_env = EnvironmentHandler.create_environment(old_config, False)
    try:
        old_model = EnvironmentHandler.get_stable_baselines3_model(old_config, old_env)
        old_model.learn(total_timesteps=64)
        checkpoint = str(tmp_path / "old.zip")
        old_model.save(checkpoint)
        weights = {k: v.detach().clone() for k, v in old_model.policy.state_dict().items()}
    finally:
        old_env.close()
    config = make_config("fixed_rate")
    config.env_params.prev_trained_policy_path = checkpoint
    config.logger_params.logging_frequency = 1
    config.logger_params.evaluate_frequency = 1000
    env = EnvironmentHandler.create_environment(config, False)
    try:
        model = EnvironmentHandler.get_stable_baselines3_model(config, env)
        assert isinstance(model, MirrorPPO)
        assert model.num_timesteps == 64
        assert env.observation_space.shape == (44,) and env.action_space.shape == (24,)
        for key, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        handler = TrainLogHandler(str(tmp_path / "logs"))
        model.learn(total_timesteps=128, reset_num_timesteps=False, callback=EnvironmentHandler.get_callback(config, handler))
        assert model.num_timesteps == 192
        record = json.loads((tmp_path / "logs/train_log.json").read_text())["log_datas"][-1]
        assert record["reference_motion_mode"] == "fixed_rate"
        assert record["mirror_loss"] is not None and np.isfinite(record["mirror_loss"])
    finally:
        env.close()


def test_training_cli_exposes_imitation_fields_and_disables_curriculum(monkeypatch):
    class Parsed(Exception):
        pass

    seen = []
    original = DictionableDataclass.set_from_args

    def capture(instance, args, prefix=""):
        original(instance, args, prefix)
        if prefix == "config.":
            seen.append(instance)
            raise Parsed

    monkeypatch.setattr(DictionableDataclass, "set_from_args", staticmethod(capture))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_train",
            "--config_file_path",
            str(CONFIG),
            "--config.total_timesteps",
            "10000000",
            "--no-config.env_params.trajectory_curriculum.enabled",
            "--config.env_params.reference_motion_mode",
            "fixed_rate",
            "--config.env_params.prev_trained_policy_path",
            "latest.zip",
        ],
    )
    with pytest.raises(Parsed):
        runpy.run_module("rl_train.run_train", run_name="__main__")
    assert len(seen) == 1
    assert seen[0].total_timesteps == 10000000
    assert seen[0].env_params.reference_motion_mode == "fixed_rate"
    assert seen[0].env_params.trajectory_curriculum.enabled is False
    assert seen[0].env_params.prev_trained_policy_path == "latest.zip"
