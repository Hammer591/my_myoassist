"""Check physical reward scales, episode boundaries, and a real PPO update."""

import json

import numpy as np
import pytest

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from tools.tests.conftest import REPO_ROOT


CONFIG = REPO_ROOT / "rl_train/train/train_configs/imitation_22_NoDevice_hip35_knee3_smooth_30M.json"


def make_config():
    config = DictionableDataclass.create(ImitationTrainSessionConfig, json.loads(CONFIG.read_text()))
    config.env_params.num_envs = 1
    config.env_params.robustness.enabled = False
    return config


@pytest.fixture
def env():
    instance = EnvironmentHandler.create_environment(make_config(), False, is_evaluate_mode=True)
    instance.reset(seed=1234)
    yield instance
    instance.close()


def test_knee_release_and_proximity_begin_before_limit(env):
    ids = [env.sim.model.joint(f"knee_angle_{side}").id for side in "rl"]
    np.testing.assert_allclose(np.rad2deg(env.sim.model.jnt_range[ids, 1]), 3)
    assert set(ids) <= set(env._proximity_joint_ids)
    # The +3 deg physical boundary has a 5 deg approach region: onset at -2 deg.
    assert env.sim.model.jnt_range[ids[0], 1] - env._proximity_margin == pytest.approx(np.deg2rad(-2))


@pytest.mark.parametrize("dt", [1 / 1200, 1 / 600])
def test_physics_rate_cost_and_control_boundary_continuity(env, dt):
    env._hip_antagonist_enabled = False
    ids = env._human_torque_dofs
    env.sim.data.qfrc_actuator[ids] = 0
    env._accumulate_physics_costs(dt, 780)
    assert env._human_torque_rate_integral == 0
    # A 600 Nm/s ramp has unit normalized squared rate, at any physics timestep.
    env.sim.data.qfrc_actuator[ids] = 600 * dt
    env._accumulate_physics_costs(dt, 780)
    assert env._human_torque_rate_integral == pytest.approx(dt)
    env._human_torque_rate_integral = 0  # start next control interval, retain previous torque
    env.sim.data.qfrc_actuator[ids] += 600 * dt
    env._accumulate_physics_costs(dt, 780)
    assert env._human_torque_rate_integral == pytest.approx(dt)
    env._accumulate_physics_costs(dt, 780)
    assert env._human_torque_rate_integral == pytest.approx(dt)  # constant moment costs zero


def test_antagonist_cost_has_free_allowance_and_is_sign_symmetric(env, monkeypatch):
    env._human_rate_enabled = False
    parts = np.array([[100.0, -80.0], [-60.0, 40.0]])
    monkeypatch.setattr(env, "_muscle_moment_contributions", lambda _: parts)
    env._accumulate_physics_costs(0.01, 780)
    assert env._hip_antagonist_integral == pytest.approx(0.001)
    parts *= -1
    env._accumulate_physics_costs(0.01, 780)
    assert env._hip_antagonist_integral == pytest.approx(0.002)
    parts[:] = [[15, -10], [-10, 5]]
    env._accumulate_physics_costs(0.01, 780)
    assert env._hip_antagonist_integral == pytest.approx(0.002)


def test_muscle_projection_matches_active_torque(env):
    env.sim.forward()
    parts = env._muscle_moment_contributions(env._human_torque_dofs)
    np.testing.assert_allclose(parts.sum(0), env.sim.data.qfrc_actuator[env._human_torque_dofs], atol=1e-8)


def test_projection_excludes_exoskeleton_motors():
    from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig

    path = REPO_ROOT / "rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_robustness_20M_20Nm.json"
    config = DictionableDataclass.create(ExoImitationTrainSessionConfig, json.loads(path.read_text()))
    config.env_params.num_envs = 1
    instance = EnvironmentHandler.create_environment(config, False, is_evaluate_mode=True)
    try:
        instance.reset(seed=1234)
        instance.sim.data.ctrl[instance._device_actuator_ids] = [0.5, -0.5]
        instance.sim.forward()
        model = instance.sim.model
        dofs = np.array([int(model.joint(f"hip_flexion_{s}").dofadr[0]) for s in "rl"])
        human = instance._muscle_moment_contributions(dofs).sum(0)
        expected_device = np.zeros(model.nv)
        for actuator in instance._device_actuator_ids:
            joint = model.actuator_trnid[actuator, 0]
            expected_device[model.jnt_dofadr[joint]] += (
                model.actuator_gear[actuator, 0] * instance.sim.data.actuator_force[actuator]
            )
        assert np.any(abs(expected_device[dofs]) > 0)
        np.testing.assert_allclose(instance.sim.data.qfrc_actuator[dofs] - human, expected_device[dofs], atol=1e-8)
    finally:
        instance.close()


def test_excitation_cost_and_reset_do_not_bridge_episodes(env):
    action = np.zeros(env.action_space.shape)
    _, _, _, _, info = env.step(action)
    assert info["rwd_dict"]["muscle_excitation_diff_penalty"] == 0
    before = env.sim.data.ctrl[env._muscle_actuator_ids].copy()
    _, _, _, _, info = env.step(np.ones_like(action) * 0.2)
    after = env.sim.data.ctrl[env._muscle_actuator_ids].copy()
    expected = -env.dt * np.mean(((after - before) / 0.1) ** 2)
    assert info["rwd_dict"]["muscle_excitation_diff_penalty"] == pytest.approx(expected)
    assert expected < 0
    assert info["rwd_dict"]["human_joint_torque_rate_penalty"] < 0
    assert info["rwd_dict"]["hip_antagonist_moment_penalty"] <= 0
    env.reset(seed=1234)
    assert env._prev_human_torque is None
    assert env._prev_muscle_excitation is None
    assert env._human_torque_rate_integral == env._hip_antagonist_integral == 0
    _, _, _, _, info = env.step(action)
    assert info["rwd_dict"]["muscle_excitation_diff_penalty"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("torque_rate_scale_nm_s", 0),
        ("excitation_diff_scale", -1),
        ("antagonist_scale_nm", float("nan")),
        ("antagonist_free_nm", -1),
    ],
)
def test_invalid_normalization_rejected(field, value):
    config = make_config()
    setattr(config.env_params.human_moment_regularization, field, value)
    with pytest.raises(ValueError, match="finite"):
        EnvironmentHandler.create_environment(config, False)


def test_legacy_config_keeps_new_rewards_disabled():
    config = make_config()
    for key in ("human_joint_torque_rate_penalty", "muscle_excitation_diff_penalty", "hip_antagonist_moment_penalty"):
        setattr(config.env_params.reward_keys_and_weights, key, 0)
    config.env_params.human_moment_regularization.torque_joints = []
    config.env_params.human_moment_regularization.antagonist_joints = []
    instance = EnvironmentHandler.create_environment(config, False)
    try:
        instance.reset(seed=1234)
        _, _, _, _, info = instance.step(np.zeros(instance.action_space.shape))
        for key in ("human_joint_torque_rate_penalty", "muscle_excitation_diff_penalty", "hip_antagonist_moment_penalty"):
            assert info["rwd_dict"][key] == 0
    finally:
        instance.close()


def test_real_ppo_update_with_new_rewards():
    config = make_config()
    config.ppo_params.n_steps = 32
    config.ppo_params.batch_size = 32
    instance = EnvironmentHandler.create_environment(config, False)
    try:
        model = EnvironmentHandler.get_stable_baselines3_model(config, instance)
        model.learn(total_timesteps=64, progress_bar=False)
        assert model.num_timesteps == 64
        assert model._n_updates > 0
        assert np.isfinite(model.logger.name_to_value["train/loss"])
    finally:
        instance.close()
