"""Evaluation limits and episode boundaries must remain explicit in saved gait data."""

from copy import deepcopy
import json
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


@pytest.mark.parametrize("mode, expected_length", [(None, 3), ("strict", 1)])
def test_default_replay_allows_tracking_error_and_never_restarts(mode, expected_length, tmp_path, monkeypatch):
    config, initial_env = build_env(PHOTO)
    initial_env.close()
    handler = TrainLogHandler(str(tmp_path))
    handler.log_datas = [SimpleNamespace(num_timesteps=0)]
    evaluator = ImitationGaitEvaluator(handler, config)
    evaluator.initialize_env(render=False, **({"evaluation_mode": mode} if mode else {}))
    try:
        env = evaluator.env
        env.CUSTOM_MAX_EPISODE_STEPS = 3
        monkeypatch.setattr(env, "_get_done", lambda: False)
        monkeypatch.setattr(
            env, "_get_qpos_diff", lambda: {k: 0.3 for k in env._reward_keys_and_weights.qpos_imitation_rewards}
        )
        policy = SimpleNamespace(predict=lambda obs, deterministic: (np.zeros(env.action_space.shape), None))
        monkeypatch.setattr(EnvironmentHandler, "get_stable_baselines3_model", lambda *a, **kw: policy)
        path = evaluator.evaluate(
            str(tmp_path),
            "continuous.json",
            velocity_mode=env.VelocityMode.UNIFORM,
            target_velocity_period=2,
            min_target_velocity=1.25,
            max_target_velocity=1.25,
            max_timestep=10,
        )
        gait = GaitData()
        gait.read_json_data(path)
        assert gait.metadata["data_length"] == expected_length
        assert gait.metadata["episode_ids"] == [0] * expected_length
        assert len(gait.metadata["episodes"]) == 1
        assert gait.metadata["evaluation"]["terminate_when_done"] is True
        assert gait.metadata["evaluation"]["checkpoint"].endswith("model_0.zip")
        if mode is None:
            assert gait.metadata["evaluation"]["mode"] == "relaxed"
            assert gait.metadata["episodes"][0]["termination_reasons"] == ["time_limit"]
    finally:
        evaluator.env.close()


def test_stitched_or_legacy_cache_is_not_reused_for_continuous_replay(tmp_path):
    path = tmp_path / "cache.json"
    metadata = {"evaluation": {"mode": "relaxed", "terminate_when_done": False}, "episodes": [{}, {}]}
    path.write_text(json.dumps({"metadata": metadata}))
    assert not ImitationGaitEvaluator.is_single_episode_cache(str(path), "relaxed")
    metadata["evaluation"]["terminate_when_done"] = True
    metadata["episodes"] = [{}]
    path.write_text(json.dumps({"metadata": metadata}))
    assert ImitationGaitEvaluator.is_single_episode_cache(str(path), "relaxed")
    assert not ImitationGaitEvaluator.is_single_episode_cache(str(path), "strict")


def test_periodic_analyzer_requests_relaxed_single_episode_by_default(tmp_path, monkeypatch):
    from rl_train.analyzer.train_analyzer import TrainAnalyzer
    from rl_train.analyzer.train_log_analyzer import TrainLogAnalyzer
    from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData

    config = json.loads(PHOTO.read_text())
    config["evaluate_param_list"] = config["evaluate_param_list"][:1]
    config["evaluate_param_list"][0].pop("evaluation_mode", None)
    (tmp_path / "session_config.json").write_text(json.dumps(config))
    handler = TrainLogHandler(str(tmp_path))
    handler.log_datas = [ImitationTrainCheckpointData(num_timesteps=933888)]
    handler.write_json_file()
    called = {}

    def initialize(self, **kwargs):
        called.update(kwargs)
        self.env = SimpleNamespace(close=lambda: None)

    def evaluate(self, result_dir, file_name, **kwargs):
        called.update(kwargs)
        data = GaitData()
        path = str(tmp_path / "gait.json")
        data.save_json_data(path)
        return path

    monkeypatch.setattr(ImitationGaitEvaluator, "initialize_env", initialize)
    monkeypatch.setattr(ImitationGaitEvaluator, "load_reference_data", lambda self: None)
    monkeypatch.setattr(ImitationGaitEvaluator, "evaluate", evaluate)
    monkeypatch.setattr(ImitationGaitEvaluator, "replay", lambda *a, **kw: None)
    monkeypatch.setattr(TrainLogAnalyzer, "plot_reward", lambda *a, **kw: None)
    monkeypatch.setattr(TrainLogAnalyzer, "plot_reward_dict", lambda *a, **kw: None)
    monkeypatch.setattr(TrainAnalyzer, "analyze", lambda *a, **kw: [])
    TrainAnalyzer().analyze_in_sequence(str(tmp_path), show_plot=False)
    assert called["evaluation_mode"] == "relaxed"
    assert called["terminate_when_done"] is True
    assert (tmp_path / "analyze_results_933888_00_relaxed" / "train_analyzer_report.json").exists()


def test_cli_defaults_to_relaxed_replay(monkeypatch):
    import sys
    from rl_train.run_policy_eval import _parse_args

    monkeypatch.setattr(sys, "argv", ["run_policy_eval"])
    assert _parse_args().evaluation_mode == "relaxed"
