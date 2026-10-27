"""Check randomised physics, observation causality and motor timing on the actual RL environment."""

import json
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.envs.robustness import Robustness
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass


SOURCE = Path(__file__).resolve().parents[2] / "rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_trial_20Nm.json"


def make_env(params=None, evaluate=False):
    cfg = DictionableDataclass.create(ExoImitationTrainSessionConfig, json.loads(SOURCE.read_text()))
    cfg.env_params.flag_random_ref_index = False
    if params is not None:
        cfg.env_params.robustness = DictionableDataclass.create(
            type(cfg.env_params.robustness), {"enabled": True, **params}
        )
    return cfg, EnvironmentHandler.create_environment(cfg, False, is_evaluate_mode=evaluate)


def test_physics_sampling_reproducible_and_not_compounded():
    _, env = make_env({
        "human_body_names": ["pelvis", "femur_r", "femur_l"],
        "human_mass_scale_range": [0.9, 1.1],
        "human_inertia_scale_range": [0.9, 1.1],
        "muscle_strength_scale_range": [0.8, 1.2],
        "observation_noise_std": {"qpos": 0.01},
        "observation_bias_std": {"qpos": 0.02},
        "observation_delay_steps_range": [0, 2],
        "motor_delay_seconds_range": [0.002, 0.01],
        "motor_time_constant_seconds_range": [0.01, 0.03],
    })
    try:
        rb = env._robustness
        first, info = env.reset(seed=42)
        params = info["robustness"]
        mass = env.sim.model.body_mass.copy()
        inertia = env.sim.model.body_inertia.copy()
        muscle = env.sim.model.actuator_gainprm.copy()
        ids = rb.body_ids
        np.testing.assert_allclose(mass[ids], rb.nominal["body_mass"][ids] * params["human_mass_scale"])
        np.testing.assert_allclose(inertia[ids], rb.nominal["body_inertia"][ids]
                                   * params["human_mass_scale"] * params["human_inertia_scale"])
        device = env.sim.model.body("PhotoHipExo_L1_waist").id
        assert mass[device] == rb.nominal["body_mass"][device]
        np.testing.assert_allclose(muscle[rb.muscle_ids, 2],
                                   rb.nominal["actuator_gainprm"][rb.muscle_ids, 2]
                                   * params["muscle_strength_scale"])
        np.testing.assert_allclose(env.sim.model.actuator_biasprm[rb.muscle_ids, 2], muscle[rb.muscle_ids, 2])
        for seed in range(4):
            env.reset(seed=seed)
            env.step(np.zeros(24))
        second, info2 = env.reset(seed=42)
        assert info2["robustness"] == params
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(mass, env.sim.model.body_mass)
        np.testing.assert_array_equal(inertia, env.sim.model.body_inertia)
        np.testing.assert_array_equal(muscle, env.sim.model.actuator_gainprm)
        assert env.sim.model.body_subtreemass[0] == pytest.approx(env.sim.model.body_mass.sum())
    finally:
        env.close()


def test_noisy_delayed_observation_does_not_corrupt_physics_or_rewards():
    _, clean = make_env()
    _, noisy = make_env({
        "observation_noise_std": {"qpos": 0.1},
        "observation_bias_std": {"qpos": 0.1},
        "observation_delay_steps_range": [1, 1],
    })
    try:
        clean.reset(seed=7)
        reset_obs, _ = noisy.reset(seed=7)
        action = np.ones(24) * 0.1
        expected, reward, term, trunc, _ = clean.step(action)
        observed, noisy_reward, noisy_term, noisy_trunc, _ = noisy.step(action)
        rb = noisy._robustness
        np.testing.assert_allclose(clean.sim.data.qpos, noisy.sim.data.qpos, atol=1e-12)
        assert noisy_reward == pytest.approx(reward, abs=1e-12)
        assert (term, trunc) == (noisy_term, noisy_trunc)
        np.testing.assert_array_equal(observed[rb.indices], reset_obs[rb.indices])
        assert not np.allclose(expected[rb.indices], observed[rb.indices])
        # The command target and muscle activation observations remain current and uncorrupted.
        other = np.setdiff1d(np.arange(44), rb.indices)
        np.testing.assert_allclose(expected[other], observed[other], atol=1e-12)
    finally:
        clean.close()
        noisy.close()


def test_motor_delay_bandwidth_gain_and_saturation_at_physics_rate():
    rb = Robustness.__new__(Robustness)
    rb.commands = deque([np.zeros(2), np.zeros(2)], maxlen=3)
    rb.output = np.zeros(2)
    rb.sample = {"motor_time_constant_seconds": 0.02, "motor_gain_scale": 2.0}
    limits = np.array([[-1.0, 1.0]] * 2)
    np.testing.assert_array_equal(rb.motor_control([1, -1], 0.001, limits), [0, 0])
    np.testing.assert_array_equal(rb.motor_control([1, -1], 0.001, limits), [0, 0])
    for i in range(1, 21):
        actual = rb.motor_control([1, -1], 0.001, limits)
        np.testing.assert_allclose(actual, np.array([1, -1]) * (1 - np.exp(-i * 0.001 / 0.02)))


def test_action_smoothness_prices_requested_commands_and_clears_on_reset():
    cfg, env = make_env({"motor_time_constant_seconds_range": [0.02, 0.02]})
    try:
        env.rwd_keys_wt["exo_action_diff_penalty"] = 0.1
        env.reset(seed=0)
        action = np.zeros(24)
        action[-2:] = 1
        _, _, _, _, info = env.step(action)
        assert env.sim.data.time == pytest.approx(env.dt)
        assert info["rwd_dict"]["exo_action_diff_penalty"] == 0
        assert np.all(env.sim.data.ctrl[-2:] < 1)
        action[-2:] = -1
        _, _, _, _, info = env.step(action)
        assert info["rwd_dict"]["exo_action_diff_penalty"] == pytest.approx(-env.dt * 4)
        env.reset(seed=0)
        assert np.all(env._robustness.output == 0)
        _, _, _, _, info = env.step(action)
        assert info["rwd_dict"]["exo_action_diff_penalty"] == 0
    finally:
        env.close()


@pytest.mark.parametrize("params", [
    {"human_mass_scale_range": [0.9, 1.1]},
    {"motor_time_constant_seconds_range": [-0.01, 0.01]},
    {"motor_gain_scale_range": [2, 1]},
    {"observation_delay_steps_range": [0, 0.5]},
    {"observation_noise_std": {"target_velocity": 0.1}},
    {"observation_noise_std": {"qvel": -0.1}},
])
def test_invalid_randomization_fails_clearly(params):
    with pytest.raises(ValueError):
        make_env(params)


def test_evaluation_uses_nominal_parameters_unless_opted_in():
    _, env = make_env({"motor_gain_scale_range": [0.9, 1.1]}, evaluate=True)
    try:
        assert not env._robustness.active
        _, info = env.reset(seed=0)
        assert "robustness" not in info
    finally:
        env.close()
