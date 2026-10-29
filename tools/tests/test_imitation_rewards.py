"""Check normalized rewards, historical compatibility and recorded contributions."""

import json
from types import SimpleNamespace

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pytest

from rl_train.analyzer.train_log_analyzer import TrainLogAnalyzer
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
from rl_train.utils.train_log_handler import TrainLogHandler
from tools.tests.conftest import REPO_ROOT, build_env


CONFIG = REPO_ROOT / "rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_robustness_20M_20Nm.json"


def test_normalization_bounds_relative_weight_invariance_and_units(monkeypatch):
    config, env = build_env(CONFIG)
    try:
        env.reset(seed=1234)
        settings = config.env_params.imitation_reward
        errors = {key: scale * 0.5 for key, scale in settings.qpos_error_scales.items()}
        monkeypatch.setattr(env, "_get_qpos_diff", lambda: errors)
        monkeypatch.setattr(env, "_get_qvel_diff", lambda: {key: 0.0 for key in settings.qvel_error_scales})
        r = env.get_reward_dict(env.obs_dict)
        assert r["qpos_imitation_rewards"] == pytest.approx(env.dt * np.exp(-0.25))
        assert r["qvel_imitation_rewards"] == pytest.approx(env.dt)
        assert env.rwd_keys_wt["qpos_imitation_rewards"] == 8.0
        assert env.rwd_keys_wt["qvel_imitation_rewards"] == 1.0
        assert env.rwd_keys_wt["foot_force_penalty"] == 2.0
        for joint in env._reward_keys_and_weights.qpos_imitation_rewards:
            env._reward_keys_and_weights.qpos_imitation_rewards[joint] *= 7
        scaled = env.get_reward_dict(env.obs_dict)
        assert scaled["qpos_imitation_rewards"] == pytest.approx(r["qpos_imitation_rewards"])
        assert scaled["dense"] == pytest.approx(sum(scaled[key] * weight for key, weight in env.rwd_keys_wt.items()))
        assert r["dense"] == pytest.approx(sum(r[key] * weight for key, weight in env.rwd_keys_wt.items()))
    finally:
        env.close()


def test_boundary_error_is_penalized_and_legacy_rewards_are_preserved(monkeypatch):
    for path, normalized in [(CONFIG, True), (CONFIG.with_name(CONFIG.name.replace("20M", "50M")), False)]:
        config, env = build_env(path)
        try:
            env.reset(seed=1234)
            monkeypatch.setattr(
                env, "_get_qpos_diff", lambda: {key: 0.2 for key in env._reward_keys_and_weights.qpos_imitation_rewards}
            )
            monkeypatch.setattr(
                env, "_get_qvel_diff", lambda: {key: 0.0 for key in env._reward_keys_and_weights.qvel_imitation_rewards}
            )
            r = env.get_reward_dict(env.obs_dict)
            weights = config.env_params.reward_keys_and_weights.qpos_imitation_rewards
            scales = config.env_params.imitation_reward.qpos_error_scales
            expected = (
                env.dt * sum(w * np.exp(-((0.2 / scales[key]) ** 2)) for key, w in weights.items()) / sum(weights.values())
                if normalized
                else env.dt * sum(weights.values()) * np.exp(-8 * 0.2**2)
            )
            assert r["qpos_imitation_rewards"] == pytest.approx(expected)
            if not normalized:
                assert env.rwd_keys_wt["qpos_imitation_rewards"] == pytest.approx(3.9)
            else:
                assert np.exp(-((0.2 / scales["knee_angle_l"]) ** 2)) == pytest.approx(0.5272924240)
        finally:
            env.close()


@pytest.mark.parametrize(
    "bad", ["missing_scale", "zero_scale", "negative_joint", "nan_weight", "empty_weights", "unknown_mode"]
)
def test_invalid_reward_configuration_fails_early(bad):
    weights = ImitationTrainSessionConfig.EnvParams.RewardWeights(qpos_imitation_rewards={"knee_angle_l": 1.0})
    settings = ImitationTrainSessionConfig.EnvParams.ImitationReward(
        mode="normalized", qvel_weight=0, qpos_error_scales={"knee_angle_l": 0.15}
    )
    if bad == "missing_scale":
        settings.qpos_error_scales.clear()
    elif bad == "zero_scale":
        settings.qpos_error_scales["knee_angle_l"] = 0
    elif bad == "negative_joint":
        weights.qpos_imitation_rewards["knee_angle_l"] = -1
    elif bad == "nan_weight":
        settings.qpos_weight = float("nan")
    elif bad == "empty_weights":
        weights.qpos_imitation_rewards.clear()
    else:
        settings.mode = "typo"
    with pytest.raises(ValueError):
        settings.validate(weights)


def test_real_ppo_logs_reconstruct_normalized_return(tmp_path):
    config, env = build_env(CONFIG)
    config.ppo_params.n_epochs = 1
    config.logger_params.logging_frequency = 1
    config.logger_params.evaluate_frequency = 1000
    handler = TrainLogHandler(str(tmp_path))
    try:
        model = EnvironmentHandler.get_stable_baselines3_model(config, env)
        model.learn(total_timesteps=128, callback=EnvironmentHandler.get_callback(config, handler))
        records = json.loads((tmp_path / "train_log.json").read_text())["log_datas"]
        assert len(records) == 2
        for record in records:
            assert record["completed_episodes"] > 0
            assert record["imitation_reward_mode"] == "normalized"
            components = record["average_reward_dict_per_episode"]
            weights = record["effective_reward_weights"]
            assert weights["qpos_imitation_rewards"] == 8
            assert weights["qvel_imitation_rewards"] == 1
            assert weights["exo_activation_penalty"] == 0.1
            assert weights["exo_action_diff_penalty"] == 0.05
            assert weights["exo_torque_excess_penalty"] == 2.0
            assert components["exo_activation_penalty"] < 0
            assert components["exo_action_diff_penalty"] < 0
            assert components["exo_torque_excess_penalty"] < 0
            assert sum(components[k] * w for k, w in weights.items()) == pytest.approx(record["average_reward_per_episode"])
            assert components["dense"] == pytest.approx(record["average_reward_per_episode"])
    finally:
        env.close()


def test_plot_uses_each_records_effective_weights_with_legacy_fallback(tmp_path, monkeypatch):
    legacy = ImitationTrainCheckpointData(
        num_timesteps=1,
        average_reward_dict_per_episode={"qpos_imitation_rewards": 2.0},
        reward_weights={"qpos_imitation_rewards": {"a": 1, "b": 0.5}},
        reward_log_schema_version=2,
    )
    normalized = ImitationTrainCheckpointData(
        num_timesteps=2,
        average_reward_dict_per_episode={"qpos_imitation_rewards": 2.0},
        reward_weights=legacy.reward_weights,
        effective_reward_weights={"qpos_imitation_rewards": 4},
        imitation_reward_mode="normalized",
        reward_log_schema_version=2,
    )
    seen = []
    original = matplotlib.axes.Axes.fill_between

    def capture(self, x, y1, y2, **kwargs):
        seen.append(y2)
        return original(self, x, y1, y2, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", capture)
    try:
        TrainLogAnalyzer(SimpleNamespace(log_datas=[legacy, normalized])).plot_reward_dict(
            result_dir=str(tmp_path), show_plot=False, mult_weights=True
        )
        assert seen == [[3.0, 8.0]]
    finally:
        plt.close("all")
