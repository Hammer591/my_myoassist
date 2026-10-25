"""An imitation reset must not inherit joints omitted from the reference motion."""

import numpy as np

from tools.tests.conftest import build_env, shipped_configs


HIPPO = next(p for p in shipped_configs() if "Hippo_L1" in p.name)


def test_fixed_start_is_independent_of_previous_rollout():
    _, env = build_env(HIPPO)
    try:
        env.is_evaluate_mode = True
        env._flag_random_ref_index = False
        env.set_target_velocity_mode_manually(env.VelocityMode.UNIFORM, 0, 1.25, 1.25, 1.25, 2)
        initial_obs, _ = env.reset()
        initial_qpos = env.sim.data.qpos.copy()
        initial_qvel = env.sim.data.qvel.copy()
        # Exercise the real dynamics so wrap joints, contacts and activation state
        # differ from construction, as they do after a failed startup.
        action = np.zeros(env.action_space.shape)
        for _ in range(20):
            env.step(action)
        repeated_obs, _ = env.reset()
        np.testing.assert_allclose(env.sim.data.qpos, initial_qpos, atol=1e-12, rtol=0)
        np.testing.assert_allclose(env.sim.data.qvel, initial_qvel, atol=1e-12, rtol=0)
        np.testing.assert_allclose(repeated_obs, initial_obs, atol=1e-12, rtol=0)
    finally:
        env.close()


def test_random_start_respects_reset_seed():
    _, env = build_env(HIPPO)
    try:
        env.is_evaluate_mode = True
        env._flag_random_ref_index = True
        first_obs, _ = env.reset(seed=1234)
        first_phase = env._imitation_index
        for _ in range(5):
            env.step(np.zeros(env.action_space.shape))
        repeated_obs, _ = env.reset(seed=1234)
        assert env._imitation_index == first_phase
        np.testing.assert_allclose(repeated_obs, first_obs, atol=1e-12, rtol=0)
    finally:
        env.close()
