"""Episode accounting must survive asynchronous endings and logging boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest

from rl_train.utils.learning_callback import BaseCustomLearningCallback
from rl_train.utils.train_log_handler import TrainLogHandler


def callback(tmp_path, num_envs=1):
    cb = BaseCustomLearningCallback(log_rollout_freq=1, evaluate_freq=1000, log_handler=TrainLogHandler(str(tmp_path)))
    cb.model = SimpleNamespace(
        get_env=lambda: SimpleNamespace(num_envs=num_envs),
        num_timesteps=0,
        save=lambda path: None,
        logger=SimpleNamespace(name_to_value={}),
    )
    cb._init_callback()
    return cb


def step(cb, rewards, dones, infos=None):
    cb.locals = {
        "rewards": np.array(rewards),
        "dones": dones,
        "infos": infos
        if infos is not None
        else [
            {
                "rwd_dict": {"dense": np.array(r)},
                "target_velocity": 1.25,
                "termination_reasons": ["trajectory:ankle_angle_l"] if done else [],
            }
            for r, done in zip(rewards, dones)
        ],
    }
    cb._on_step()
    cb.num_timesteps += len(rewards)
    cb.model.num_timesteps = cb.num_timesteps


def test_each_completed_episode_is_counted_once(tmp_path):
    cb = callback(tmp_path)
    step(cb, [2], [True])
    step(cb, [4], [True])
    result = cb._on_rollout_end()
    assert result.average_reward_per_episode == 3
    assert result.average_reward_dict_per_episode["dense"] == 3
    assert result.completed_episodes == 2
    assert result.termination_counts == {"trajectory:ankle_angle_l": 2}
    assert result.target_velocity_stats == {"min": 1.25, "mean": 1.25, "max": 1.25, "samples": 2}


def test_unfinished_episode_spans_logging_windows(tmp_path):
    cb = callback(tmp_path, num_envs=2)
    step(cb, [2, 10], [True, False])
    first = cb._on_rollout_end()
    assert first.average_reward_dict_per_episode["dense"] == 2
    step(cb, [3, 20], [False, True])
    second = cb._on_rollout_end()
    assert second.completed_episodes == 1
    assert second.average_num_timestep == 2
    assert second.average_reward_per_episode == 30
    assert second.average_reward_dict_per_episode["dense"] == 30
    step(cb, [4, 1], [True, False])
    third = cb._on_rollout_end()
    assert third.average_reward_dict_per_episode["dense"] == 7


def test_no_finished_episode_and_sparse_info_are_safe(tmp_path):
    cb = callback(tmp_path, num_envs=2)
    step(cb, [1, 2], [False, False], [None, {}])
    empty = cb._on_rollout_end()
    assert empty.completed_episodes == 0
    assert empty.average_reward_dict_per_episode == {}
    step(cb, [2, 3], [True, True], [{"rwd_dict": {"left_only": 2}}, {"rwd_dict": {"dense": 3}}])
    result = cb._on_rollout_end()
    assert result.average_reward_dict_per_episode == {"left_only": 1, "dense": 1.5}
    assert result.average_reward_per_episode == pytest.approx(4)


def test_plot_weights_match_current_environment_reward_formula(tmp_path, monkeypatch):
    from rl_train.analyzer.train_log_analyzer import TrainLogAnalyzer
    from rl_train.utils.train_checkpoint_data_imitation import ImitationTrainCheckpointData
    import matplotlib.axes
    import matplotlib.pyplot as plt

    record = ImitationTrainCheckpointData(
        num_timesteps=1,
        reward_log_schema_version=2,
        average_reward_dict_per_episode={"qpos_imitation_rewards": 2.0},
        reward_weights={"qpos_imitation_rewards": {"a": 1.0, "b": 0.5}},
    )
    seen = []
    original = matplotlib.axes.Axes.fill_between

    def capture(self, x, y1, y2, **kwargs):
        seen.append(y2)
        return original(self, x, y1, y2, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", capture)
    analyzer = TrainLogAnalyzer(SimpleNamespace(log_datas=[record]))
    analyzer.plot_reward_dict(result_dir=str(tmp_path), show_plot=False, mult_weights=True)
    assert seen == [[3.0]]
    plt.close("all")


def test_real_ppo_rollout_writes_consistent_imitation_log(tmp_path):
    import json
    from rl_train.envs.environment_handler import EnvironmentHandler
    from tools.tests.conftest import build_env, shipped_configs

    photo = next(p for p in shipped_configs() if "PhotoHipExo_L1" in p.name)
    config, env = build_env(photo)
    config.logger_params.logging_frequency = 1
    config.logger_params.evaluate_frequency = 1000
    try:
        model = EnvironmentHandler.get_stable_baselines3_model(config, env)
        handler = TrainLogHandler(str(tmp_path))
        cb = EnvironmentHandler.get_callback(config, handler)
        model.learn(total_timesteps=128, callback=cb, progress_bar=False)
        records = json.loads((tmp_path / "train_log.json").read_text())["log_datas"]
        assert len(records) == 2
        for record in records:
            assert record["completed_episodes"] > 0
            assert record["reward_log_schema_version"] == 2
            assert record["average_reward_dict_per_episode"]["dense"] == pytest.approx(
                record["average_reward_per_episode"], abs=1e-6
            )
            assert record["target_velocity_stats"]["min"] == record["target_velocity_stats"]["max"] == 1.25
    finally:
        env.close()


def test_sparse_checkpoints_evaluate_current_imitation_record_and_save_final(tmp_path, monkeypatch):
    import json
    from rl_train.envs.environment_handler import EnvironmentHandler
    from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
    from rl_train.utils import learning_callback

    config = ExoImitationTrainSessionConfig()
    config.logger_params.logging_frequency = 4
    config.logger_params.evaluate_frequency = 8
    handler = TrainLogHandler(str(tmp_path))
    cb = EnvironmentHandler.get_callback(config, handler)
    cb.model = callback(tmp_path).model
    saved, evaluated = [], []
    cb.model.save = lambda path: saved.append(cb.model.num_timesteps)
    cb._init_callback()

    class Pool:
        def apply(self, fn, args):
            records = json.loads((tmp_path / "train_log.json").read_text())["log_datas"]
            latest = records[-1]
            # The evaluation worker must read this rollout, including imitation metadata.
            assert latest["num_timesteps"] == cb.model.num_timesteps
            assert "reward_weights" in latest and "reward_accumulate" in latest
            evaluated.append(latest["num_timesteps"])

        def close(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(learning_callback, "get_context", lambda method: SimpleNamespace(Pool=lambda **kw: Pool()))
    for _ in range(17):
        rewards = {key: 0.0 for key in cb.reward_accumulate}
        rewards["dense"] = 1.0
        step(cb, [1], [True], [{"rwd_dict": rewards}])
        cb._on_rollout_end()
    assert saved == [1, 4, 8, 12, 16]
    assert evaluated == [8, 16]
    cb._on_training_end()
    assert saved == [1, 4, 8, 12, 16, 17]
    assert evaluated == [8, 16]
    assert handler.log_datas[-1].num_timesteps == 17
    assert handler.log_datas[-1].completed_episodes == 1
    assert handler.log_datas[-1].average_reward_per_episode == 1


def test_final_checkpoint_refreshes_post_update_weights_without_duplicate_log(tmp_path):
    cb = callback(tmp_path)
    saved = []
    cb.model.save = lambda path: saved.append(path)
    step(cb, [1], [True])
    cb._on_rollout_end()
    cb._on_training_end()
    assert len(saved) == 2 and saved[0] == saved[1]
    assert len(cb.train_log_handler.log_datas) == 1
