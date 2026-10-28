"""The soft torque limit penalizes physical output without changing the hard limit."""

import json

import numpy as np
import pytest

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from tools.tests.conftest import REPO_ROOT


CONFIG = REPO_ROOT / "rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_robustness_20M_20Nm.json"


def make_config():
    config = DictionableDataclass.create(ExoImitationTrainSessionConfig, json.loads(CONFIG.read_text()))
    config.env_params.num_envs = 1
    return config


@pytest.fixture(scope="module")
def nominal_env():
    env = EnvironmentHandler.create_environment(make_config(), False, is_evaluate_mode=True)
    env.reset(seed=1235)
    yield env
    env.close()


@pytest.mark.parametrize(
    "torque, cost_per_second",
    [(0, 0), (10, 0), (15, 0), (16, 0.0832), (17, 0.3712), (18, 0.9792), (19, 2.0992), (20, 4.0)],
)
def test_signed_threshold_and_nonlinear_growth(nominal_env, torque, cost_per_second):
    env = nominal_env
    ids = env._device_actuator_ids
    for sign in (-1, 1):
        env.sim.data.ctrl[ids] = [sign * torque / 20, -sign * torque / 20]
        env.sim.forward()
        np.testing.assert_allclose(env.sim.data.actuator_force[ids], [sign * torque, -sign * torque])
        r = env.get_reward_dict(env.obs_dict)
        weighted = r["exo_torque_excess_penalty"] * env.rwd_keys_wt["exo_torque_excess_penalty"]
        assert weighted / env.dt == pytest.approx(-cost_per_second)


def test_motor_averaging_gear_conversion_and_hard_limit(nominal_env):
    env = nominal_env
    ids = env._device_actuator_ids
    env.sim.data.ctrl[ids] = [1, 0]
    env.sim.forward()
    assert env._get_device_torque_excess_cost() == pytest.approx(1.0)
    original_gear = env.sim.model.actuator_gear[ids].copy()
    try:
        env.sim.model.actuator_gear[ids, 0] = 2
        env.sim.data.ctrl[ids] = [0.5, -0.5]
        env.sim.forward()
        np.testing.assert_allclose(env.sim.data.actuator_force[ids], [10, -10])
        assert env._get_device_torque_excess_cost() == pytest.approx(2.0)
    finally:
        env.sim.model.actuator_gear[ids] = original_gear
    env.sim.data.ctrl[ids] = [2, -2]
    env.sim.forward()
    np.testing.assert_allclose(env.sim.data.actuator_force[ids], [20, -20])


def test_penalty_uses_filtered_output_instead_of_requested_action():
    config = make_config()
    params = type(config.env_params.robustness)
    config.env_params.robustness = params(enabled=True, motor_time_constant_seconds_range=[0.2, 0.2])
    env = EnvironmentHandler.create_environment(config, False)
    try:
        env.reset(seed=1235)
        action = np.zeros(env.action_space.shape)
        action[-2:] = 1
        _, _, _, _, info = env.step(action)
        assert np.all(np.abs(env.sim.data.actuator_force[env._device_actuator_ids]) < 15)
        assert info["rwd_dict"]["exo_torque_excess_penalty"] == 0
    finally:
        env.close()


@pytest.mark.parametrize("field,value", [("exo_torque_soft_limit_nm", -1), ("exo_torque_excess_scale_nm", 0)])
def test_invalid_torque_penalty_settings_fail_early(field, value):
    config = make_config()
    setattr(config.env_params, field, value)
    with pytest.raises(ValueError, match="Torque penalty requires"):
        EnvironmentHandler.create_environment(config, False)
