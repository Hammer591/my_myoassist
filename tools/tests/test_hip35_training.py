"""Fresh bare-human training: reachable reference, phase observations and limit costs."""

import json
from pathlib import Path

import numpy as np
import pytest

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from rl_train.train.policies.mirror import observation_permutation

CONFIG = Path("rl_train/train/train_configs/imitation_22_NoDevice_hip35_reference_30M.json")


def config():
    c = DictionableDataclass.create(ImitationTrainSessionConfig, json.loads(CONFIG.read_text()))
    c.env_params.num_envs = 1
    c.env_params.robustness.enabled = False
    c.ppo_params.n_steps = c.ppo_params.batch_size = 64
    c.ppo_params.n_epochs = 1
    return c


def test_hip35_policy_train_reload_and_reference(tmp_path):
    c = config()
    env = EnvironmentHandler.create_environment(c, is_rendering_on=False)
    try:
        base = env.unwrapped
        assert env.observation_space.shape == (76,)
        assert env.action_space.shape == (22,)
        assert len(base._device_actuator_ids) == 0
        for side in ("l", "r"):
            assert base.sim.model.joint("hip_flexion_" + side).range[0] == pytest.approx(np.deg2rad(-35))
        env.reset()
        base._imitation_index = 10
        target = base.get_obs_dict(base.sim)["reference"]
        assert target[2] == base._reference_data["series_data"]["q_hip_flexion_l"][11]
        assert target[18] == base._reference_data["series_data"]["q_hip_flexion_l"][14]
        names = [base.sim.model.actuator(i).name for i in range(22)]
        perm = observation_permutation(c.env_params, 22, names)
        assert perm[46] == 47 and perm[62] == 63
        np.testing.assert_array_equal(perm[perm], np.arange(76))
        model = EnvironmentHandler.get_stable_baselines3_model(c, env)
        model.learn(total_timesteps=128, progress_bar=False)
        model.save(tmp_path / "hip35")
        restored = type(model).load(tmp_path / "hip35", env=env)
        action, _ = restored.predict(env.reset()[0], deterministic=True)
        assert action.shape == (22,)
        assert np.isfinite(env.step(action)[0]).all()
    finally:
        env.close()


def test_substep_cost_catches_pulse_and_preserves_dynamics():
    c = config()
    env = EnvironmentHandler.create_environment(c, is_rendering_on=False)
    old = config()
    old.env_params.joint_limit_penalty_mode = "legacy"
    old.env_params.joint_limit_proximity_joints = []
    legacy = EnvironmentHandler.create_environment(old, is_rendering_on=False)
    try:
        a, b = env.unwrapped, legacy.unwrapped
        a.reset(seed=19)
        b.reset(seed=19)
        for _ in range(3):
            action = np.linspace(-1, 1, 22)
            a.step(action)
            b.step(action)
            np.testing.assert_allclose(a.sim.data.qpos, b.sim.data.qpos, atol=1e-10)
            np.testing.assert_allclose(a.sim.data.act, b.sim.data.act, atol=1e-10)
        a._joint_limit_cost_integral = 0.0
        a.sim.data.sensordata[:] = 0.0
        a.sim.data.sensordata[a._limit_sensor_adrs[:2]] = [50.0, -100.0]
        a._accumulate_physics_costs(0.001, 700.0)
        a.sim.data.sensordata[:] = 0.0
        a._accumulate_physics_costs(0.001, 700.0)
        assert a._joint_limit_cost_integral == pytest.approx(0.005)
        a.reset(seed=19)
        assert a._joint_limit_cost_integral == 0.0
    finally:
        env.close()
        legacy.close()


def test_unreachable_reference_rejected():
    c = config()
    c.env_params.joint_range_overrides["hip_flexion_l"] = [-0.349, 2.356]
    with pytest.raises(ValueError, match="Reference for hip_flexion_l exceeds"):
        EnvironmentHandler.create_environment(c, is_rendering_on=False)


def test_legacy_observations_and_lumbar_direction_preserved():
    old_path = CONFIG.with_name("imitation_22_NoDevice_robustness_30M.json")
    old = DictionableDataclass.create(ImitationTrainSessionConfig, json.loads(old_path.read_text()))
    old.env_params.num_envs = 1
    old.env_params.robustness.enabled = False
    new_env = EnvironmentHandler.create_environment(config(), is_rendering_on=False)
    old_env = EnvironmentHandler.create_environment(old, is_rendering_on=False)
    try:
        assert old_env.observation_space.shape == (44,)
        assert old_env.sim.model.joint("hip_flexion_l").range[0] == pytest.approx(-0.349)
        q_old = old_env.sim.model.body("torso").quat.copy()
        q_new = new_env.sim.model.body("torso").quat.copy()
        np.testing.assert_allclose(q_new, q_old / np.linalg.norm(q_old), atol=1e-14)
        base = new_env.unwrapped
        base._imitation_index = base._reference_data_length - 1
        base._loop_reference_data = False
        target = base.get_obs_dict(base.sim)["reference"]
        assert target[2] == base._reference_data["series_data"]["q_hip_flexion_l"][-1]
        base._loop_reference_data = True
        target = base.get_obs_dict(base.sim)["reference"]
        assert target[2] == base._reference_data["series_data"]["q_hip_flexion_l"][0]
    finally:
        new_env.close()
        old_env.close()
