"""Bare-human baseline must train with muscles only and match assisted settings."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from myoassist_utils.env_spec import EnvSpec
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from tools.tests.conftest import take_a_few_ppo_steps

CONFIG_DIR = Path(__file__).resolve().parents[2] / 'rl_train/train/train_configs'


def test_bare_human_trains():
    raw = json.loads((CONFIG_DIR / 'imitation_22_NoDevice_robustness_50M.json').read_text())
    assisted = json.loads((CONFIG_DIR / 'imitation_22_PhotoHipExo_L1_fixed_rate_50M_20Nm.json').read_text())
    for key in ('ppo_params', 'total_timesteps', 'evaluate_param_list'):
        assert raw[key] == assisted[key]
    for key, value in raw['env_params'].items():
        if key not in ('env_id', 'device_key', 'reward_keys_and_weights'):
            assert value == assisted['env_params'][key], key
    for key, value in raw['env_params']['reward_keys_and_weights'].items():
        assert value == assisted['env_params']['reward_keys_and_weights'][key]
    assert raw['env_params']['reference_motion_mode'] == 'fixed_rate'
    assert raw['env_params']['imitation_reward']['mode'] == 'normalized'
    assert raw['env_params']['imitation_reward']['qpos_weight'] == 8.0
    assert raw['env_params']['reward_keys_and_weights']['foot_force_penalty'] == 2.0
    assert raw['env_params']['prev_trained_policy_path'] is None
    bare = mj.MjModel.from_xml_string(EnvSpec('myolegs22', 'NoDevice').validate().compose())
    aided = mj.MjModel.from_xml_string(EnvSpec('myolegs22', 'PhotoHipExo_L1').validate().compose())
    assert bare.nu == 22
    assert np.all(bare.actuator_dyntype == mj.mjtDyn.mjDYN_MUSCLE)
    assert not any('PhotoHipExo' in bare.body(i).name for i in range(bare.nbody))
    np.testing.assert_allclose(aided.body_mass.sum() - bare.body_mass.sum(), 3.15, atol=5e-6)
    config = DictionableDataclass.create(ImitationTrainSessionConfig, raw)
    config.env_params.num_envs = 1
    config.ppo_params.n_steps = config.ppo_params.batch_size = 64
    config.ppo_params.n_epochs = 1
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
    try:
        assert env.observation_space.shape == (44,)
        assert env.action_space.shape == (22,)
        assert take_a_few_ppo_steps(config, env) == 'MirrorPPO'
    finally:
        env.close()
