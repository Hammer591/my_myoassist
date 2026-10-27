"""Evaluation limits and episode boundaries must remain explicit in saved gait data."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from rl_train.analyzer.gait_analyze import GaitAnalyzer
from rl_train.analyzer.gait_data import GaitData
from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.utils.train_log_handler import TrainLogHandler
from tools.tests.conftest import build_env, shipped_configs


PHOTO = next(p for p in shipped_configs() if "PhotoHipExo_L1" in p.name)


@pytest.mark.parametrize("mode, threshold, cap", [("strict", 0.2, 1000), ("relaxed", 1e6, 1e9)])
def test_evaluation_mode_does_not_mutate_training_config(mode, threshold, cap, tmp_path, monkeypatch):
    config, initial_env = build_env(PHOTO)
    initial_env.close()
    original = deepcopy(config)
    handler = TrainLogHandler(str(tmp_path))
    handler.log_datas = [SimpleNamespace(num_timesteps=0)]
    evaluator = ImitationGaitEvaluator(handler, config)
    evaluator.load_reference_data()
    evaluator.initialize_env(evaluation_mode=mode, render=False)
    try:
        assert config == original
        assert evaluator.env._out_of_trajectory_threshold == threshold
        assert evaluator.env.CUSTOM_MAX_EPISODE_STEPS == cap
        # Force short episodes to exercise saving boundaries in both modes.
        evaluator.env.CUSTOM_MAX_EPISODE_STEPS = 2
        policy = SimpleNamespace(predict=lambda obs, deterministic: (np.zeros(evaluator.env.action_space.shape), None))
        monkeypatch.setattr(EnvironmentHandler, "get_stable_baselines3_model", lambda *a, **kw: policy)
        path = evaluator.evaluate(
            str(tmp_path),
            f"{mode}.json",
            velocity_mode=evaluator.env.VelocityMode.UNIFORM,
            target_velocity_period=2,
            min_target_velocity=1.25,
            max_target_velocity=1.25,
            terminate_when_done=False,
            max_timestep=7,
            seed=123,
        )
        gait = GaitData()
        gait.read_json_data(path)
        assert gait.metadata["data_length"] == 7
        assert len(gait.metadata["episode_ids"]) == 7
        assert len(gait.metadata["episodes"]) >= 4
        assert sum(e["length"] for e in gait.metadata["episodes"]) == 7
        assert gait.metadata["evaluation"]["mode"] == mode
        assert gait.metadata["evaluation"]["seed"] == 123
        assert all(e["terminated"] or e["truncated"] or e["evaluation_budget_reached"] for e in gait.metadata["episodes"])
        assert all(v == [1.25] for v in gait.series_data["target_data"]["target_velocity"])
    finally:
        evaluator.env.close()


def test_gait_cycles_never_cross_reset_even_with_hip_fallback():
    length = 200
    gait = GaitData()
    # Each half has several cycles; a global peak finder also finds a spurious
    # stride straddling the reset at frame 100.
    hip = np.cos(np.arange(length) * 2 * np.pi / 30)
    gait.series_data["joint_data"] = {"hip_flexion_r": {"qpos": [[float(v)] for v in hip]}}
    gait.series_data["sensor_data"] = {key: {"data": [[0.0]] * length} for key in ["r_foot", "r_toes"]}
    gait.metadata = {"data_length": length, "episode_ids": [0] * 100 + [1] * 100}
    analyzer = GaitAnalyzer(gait, None, False)
    segments = analyzer.get_gait_segment_index(is_right_foot_based=True)
    assert len(segments) >= 2
    assert all(start <= toe_off < end for start, toe_off, end in segments)
    assert all(gait.metadata["episode_ids"][start] == gait.metadata["episode_ids"][end] for start, _, end in segments)
