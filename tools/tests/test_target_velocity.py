"""Configured speed bounds and seeded resets must describe the actual task."""

import numpy as np
import pytest

from tools.tests.conftest import build_env, shipped_configs


PHOTO = next(p for p in shipped_configs() if "PhotoHipExo_L1" in p.name)


def test_fixed_speed_survives_resets_and_every_velocity_mode():
    _, env = build_env(PHOTO)
    try:
        for seed in range(12):
            obs, _ = env.reset(seed=seed)
            assert env._target_velocity == 1.25
            assert env._min_target_velocity == env._max_target_velocity == 1.25
            assert env.sim.data.joint("pelvis_tx").qvel[0] == 1.25
            for mode in env.VelocityMode:
                env.set_target_velocity_mode_manually(mode, 0, 1.25, 1.25, 1.25, 0.001)
                _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
                assert info["target_velocity"] == 1.25
                assert "termination_reasons" in info
    finally:
        env.close()


@pytest.mark.parametrize("reset_seed", [0, 1, 42])
def test_variable_speed_reset_is_seeded_and_reference_uses_new_target(reset_seed):
    _, env = build_env(PHOTO)
    try:
        env._min_target_velocity, env._max_target_velocity = 0.8, 1.4
        first, _ = env.reset(seed=reset_seed)
        target = env._target_velocity
        phase = env._imitation_index
        initial_velocity = env.sim.data.qvel.copy()
        assert 0.8 <= target <= 1.4
        assert env.sim.data.joint("pelvis_tx").qvel[0] == target
        for _ in range(3):
            env.step(np.zeros(env.action_space.shape))
        second, _ = env.reset(seed=reset_seed)
        np.testing.assert_allclose(first, second, atol=1e-12, rtol=0)
        np.testing.assert_allclose(env.sim.data.qvel, initial_velocity, atol=1e-12, rtol=0)
        assert env._target_velocity == target
        assert env._imitation_index == phase
        assert env._prev_step_changed_time == env.sim.data.time
        for seed in range(20):
            env.reset(seed=seed)
            assert 0.8 <= env._target_velocity <= 1.4
            assert (env._min_target_velocity, env._max_target_velocity) == (0.8, 1.4)
    finally:
        env.close()


def test_worker_factories_have_distinct_reproducible_seeds(monkeypatch):
    from rl_train.envs import environment_handler

    factories = []

    def capture(env_fns):
        factories.extend(env_fns)
        return "captured"

    monkeypatch.setattr(environment_handler, "SubprocVecEnv", capture)
    config, env = build_env(PHOTO, num_envs=3)
    assert env == "captured"
    assert [f.keywords["seed"] for f in factories] == [config.env_params.seed + i for i in range(3)]
