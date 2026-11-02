"""Evaluate the latest NoDevice checkpoint and export bilateral joint moments.

Run from the repository root with .my_venv/bin/python tools/plot_human_joint_torques.py.
Moments are MuJoCo qfrc_actuator, in model coordinate signs, without passive or
constraint forces. Each leg is normalized to its own detected gait cycles.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from rl_train.analyzer.gait_analyze import GaitAnalyzer
from rl_train.analyzer.gait_data import GaitData
from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils.train_log_handler import TrainLogHandler


JOINTS = (("Hip", "hip_flexion"), ("Knee", "knee_angle"), ("Ankle", "ankle_angle"))
SIDES = (("r", "Right", "#2166ac"), ("l", "Left", "#d6604d"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", nargs="?", type=Path)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    session = args.session or sorted(Path("rl_train/results").glob("train_session_*"))[-1]
    cfg_dict = json.loads((session / "session_config.json").read_text())
    if cfg_dict["env_params"]["device_key"] != "NoDevice":
        raise ValueError(f"Expected NoDevice model: {session}")
    checkpoint = max((session / "trained_models").glob("model_*.zip"), key=lambda p: int(p.stem.split("_")[-1]))
    step = int(checkpoint.stem.split("_")[-1])
    out = session / f"human_joint_torques_{step}"
    out.mkdir(exist_ok=True)
    config = DictionableDataclass.create(ImitationTrainSessionConfig, cfg_dict)
    handler = TrainLogHandler(str(session))
    handler.log_datas = [SimpleNamespace(num_timesteps=step)]
    evaluator = ImitationGaitEvaluator(handler, config)
    evaluator.initialize_env(evaluation_mode="relaxed", render=False)
    try:
        path = evaluator.evaluate(
            str(out), "gait_evaluated_data.json",
            velocity_mode=MyoAssistLegBase.VelocityMode.UNIFORM,
            target_velocity_period=2,
            min_target_velocity=config.env_params.min_target_velocity,
            max_target_velocity=config.env_params.max_target_velocity,
            max_timestep=args.steps, seed=args.seed, terminate_when_done=True,
        )
        mass = float(np.sum(evaluator.env.sim.model.body_mass))
    finally:
        evaluator.env.close()

    gait = GaitData()
    gait.read_json_data(path)
    analyzer = GaitAnalyzer(gait, None, show_plot=False)
    fps = config.env_params.control_framerate
    length = gait.metadata["data_length"]
    time = (np.arange(length) + 1) / fps
    columns, names = [time], ["time_s"]
    cycle_columns, cycle_names = [np.linspace(0, 100, 101)], ["gait_cycle_pct"]
    phase = cycle_columns[0]
    segments = {s: analyzer.get_gait_segment_index(is_right_foot_based=(s == "r")) for s, _, _ in SIDES}
    fig_ts, axes_ts = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig_gc, axes_gc = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    stats = {}
    for row, (label, stem) in enumerate(JOINTS):
        for side, side_label, color in SIDES:
            key = f"{stem}_{side}"
            torque = np.asarray(gait.series_data["joint_data"][key]["qfrc_actuator"], dtype=float).reshape(length)
            if not np.isfinite(torque).all():
                raise ValueError(f"Non-finite torque: {key}")
            columns.append(torque)
            names.append(f"{key}_Nm")
            axes_ts[row].plot(time, torque, color=color, lw=1.1, label=side_label)
            cycles = np.asarray([
                np.interp(phase, np.linspace(0, 100, end - start + 1), torque[start:end + 1])
                for start, _, end in segments[side] if end > start
            ])
            stats[key] = {"min_Nm": float(torque.min()), "max_Nm": float(torque.max()),
                          "rms_Nm": float(np.sqrt(np.mean(torque ** 2))), "cycles": len(cycles)}
            if len(cycles):
                mean, sd = cycles.mean(axis=0), cycles.std(axis=0)
                axes_gc[row].plot(phase, mean, color=color, label=f"{side_label} (n={len(cycles)})")
                axes_gc[row].fill_between(phase, mean - sd, mean + sd, color=color, alpha=0.18)
                cycle_columns.extend([mean, sd])
                cycle_names.extend([f"{key}_mean_Nm", f"{key}_sd_Nm"])
            axes_ts[row].set_ylabel(f"{label} moment (N m)")
        axes_gc[row].set_title(label)
        axes_gc[row].set_xlabel("Gait cycle (%), each leg separately")
        axes_gc[row].set_ylabel("Active joint moment (N m)")
        axes_gc[row].set_xlim(0, 100)
        for ax in (axes_ts[row], axes_gc[row]):
            ax.axhline(0, color="0.4", lw=0.6)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="best")
    axes_ts[-1].set_xlabel("Time (s)")
    title = f"No exoskeleton | checkpoint {step:,} | target {config.env_params.min_target_velocity:g} m/s"
    fig_ts.suptitle(title + "\nMuscle-driven joint moments (model coordinate signs)")
    fig_gc.suptitle(title + "\nGait-cycle mean and +/-1 SD (model coordinate signs)")
    for fig, name in ((fig_ts, "joint_torques_timeseries"), (fig_gc, "joint_torques_gait_cycle")):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"{name}.{suffix}", dpi=180)
        plt.close(fig)
    for values, headers, name in ((columns, names, "joint_torques_timeseries.csv"),
                                  (cycle_columns, cycle_names, "joint_torques_gait_cycle.csv")):
        np.savetxt(out / name, np.column_stack(values), delimiter=",", header=",".join(headers), comments="")
    summary = {"session": str(session), "checkpoint": str(checkpoint), "device": "NoDevice",
               "definition": "qfrc_actuator: muscle-driven generalized joint moments; excludes passive and constraint forces",
               "sign_convention": "positive along each model joint coordinate; no clinical sign conversion",
               "mass_kg": mass, "sample_rate_hz": fps, "duration_s": length / fps,
               "evaluation": gait.metadata, "segments": segments, "statistics": stats}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "README.md").write_text(
        f"# 无外骨骼人体关节力矩\n\n检查点：`{checkpoint.name}`；采样率 {fps} Hz；记录 {length / fps:.2f} s。\n\n"
        "双侧髋、膝、踝力矩取 MuJoCo `qfrc_actuator`，单位 N·m；为肌肉驱动的合关节力矩，"
        "不含被动力、约束力或接触力。正负号沿模型关节坐标，未转换为临床屈伸符号。\n\n"
        "步态周期图按各腿自身周期分别归一化，实线为均值，阴影为 ±1 标准差；"
        "周期使用仓库 GaitAnalyzer 的足部传感器分段及髋角峰值回退算法。"
        "时域图保留起步过程，周期图采用分段器返回的完整周期。\n\n"
        "评估使用 relaxed 模式、确定性策略及固定随机种子，放宽轨迹偏差终止条件；"
        "这是该次回放的力矩输出。详情见 summary.json。\n"
    )
    print(json.dumps({"output": str(out), "statistics": stats, "episodes": gait.metadata["episodes"]}, indent=2))


if __name__ == "__main__":
    main()
