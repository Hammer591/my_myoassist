"""Exercise the curriculum through real vector environments and PPO callbacks."""

import json

import numpy as np
import pytest

from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.utils.train_log_handler import TrainLogHandler
from tools.tests.conftest import REPO_ROOT, build_env


CONFIG = REPO_ROOT / "rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_robustness_20M_20Nm.json"


def test_curriculum_reaches_workers_and_is_logged_after_resumed_ppo(tmp_path):
    config, env = build_env(CONFIG, num_envs=2)
    try:
        assert env.get_attr("_out_of_trajectory_threshold") == [0.3, 0.3]
        config.env_params.trajectory_curriculum.anneal_timesteps = 128
        config.ppo_params.n_epochs = 1
        handler = TrainLogHandler(str(tmp_path))
        callback = EnvironmentHandler.get_callback(config, handler)
        callback.evaluate_freq = 1000
        callback.log_rollout_freq = 1
        model = EnvironmentHandler.get_stable_baselines3_model(config, env)
        model.num_timesteps = 50_003_968
        model.learn(total_timesteps=384, reset_num_timesteps=False, callback=callback)
        assert env.get_attr("_out_of_trajectory_threshold") == pytest.approx([0.2, 0.2])
        records = json.loads((tmp_path / "train_log.json").read_text())["log_datas"]
        assert [r["training_trajectory_threshold"] for r in records] == pytest.approx([0.3, 0.2, 0.2])
        assert records[-1]["num_timesteps"] == 50_003_968 + 384
        assert config.env_params.out_of_trajectory_threshold == 0.2
    finally:
        env.close()


@pytest.mark.parametrize("mode, threshold, terminated", [("strict", 0.2, True), ("relaxed", 1e6, False)])
def test_evaluation_never_uses_training_curriculum(mode, threshold, terminated, tmp_path, monkeypatch):
    config, training_env = build_env(CONFIG)
    training_env.close()
    evaluator = ImitationGaitEvaluator(TrainLogHandler(str(tmp_path)), config)
    evaluator.initialize_env(evaluation_mode=mode, render=False)
    env = evaluator.env
    try:
        assert env._out_of_trajectory_threshold == threshold
        env.set_training_trajectory_threshold(0.3)
        assert env._out_of_trajectory_threshold == threshold
        env.reset(seed=1234)
        monkeypatch.setattr(env, "_get_done", lambda: False)
        monkeypatch.setattr(env, "_get_qpos_diff", lambda: {"knee_angle_l": 0.25})
        _, _, done, _, info = env.step(np.zeros(env.action_space.shape))
        assert bool(done) == terminated
        assert ("trajectory:knee_angle_l" in info["termination_reasons"]) == terminated
    finally:
        env.close()


def test_curriculum_config_boundaries_and_unchanged_baseline():
    from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig

    schedule = ImitationTrainSessionConfig.EnvParams.TrajectoryCurriculum(enabled=True)
    assert schedule.threshold_at(0, 0.2) == 0.3
    assert schedule.threshold_at(2_500_000, 0.2) == pytest.approx(0.25)
    assert schedule.threshold_at(20_000_000, 0.2) == 0.2
    schedule.anneal_timesteps = 0
    with pytest.raises(ValueError):
        schedule.threshold_at(0, 0.2)
    schedule.enabled = False
    assert schedule.threshold_at(0, 0.2) == 0.2
    current = json.loads(CONFIG.read_text())
    baseline = json.loads(CONFIG.with_name(CONFIG.name.replace("20M", "50M")).read_text())
    assert current["total_timesteps"] == 20_000_000
    assert current["ppo_params"] == baseline["ppo_params"]
    assert current["env_params"]["reward_keys_and_weights"] == baseline["env_params"]["reward_keys_and_weights"]
    assert current["env_params"]["robustness"] == baseline["env_params"]["robustness"]
