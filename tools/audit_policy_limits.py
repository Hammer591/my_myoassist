"""Audit unmodified NoDevice policies using physics-rate joint-limit sensors.

Read JOINTLIMITFRC sensors rather than unsynchronized efc_force after mj_step:
the integrator may have rebuilt constraint rows without solving their force cache.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/myoassist-mpl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils.train_log_handler import TrainLogHandler


def audit(session, step, out, seed, steps):
    raw = json.loads((session / "session_config.json").read_text())
    if raw["env_params"]["device_key"] != "NoDevice":
        raise ValueError("This audit expects NoDevice")
    config = DictionableDataclass.create(ImitationTrainSessionConfig, raw)
    handler = TrainLogHandler(str(session))
    handler.log_datas = [SimpleNamespace(num_timesteps=step)]
    evaluator = ImitationGaitEvaluator(handler, config)
    evaluator.initialize_env(evaluation_mode="strict", render=False)
    env = evaluator.env.unwrapped
    model = env.sim.model.ptr
    names = [f"{joint}_{side}" for joint in ("hip_flexion", "knee_angle", "ankle_angle") for side in "rl"]
    joint_ids = np.array([model.joint(name).id for name in names])
    qids, dofs = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    ranges = np.rad2deg(model.jnt_range[joint_ids]).copy()
    addresses = []
    for joint in joint_ids:
        ids = np.flatnonzero((model.sensor_type == mujoco.mjtSensor.mjSENS_JOINTLIMITFRC) & (model.sensor_objid == joint))
        if len(ids) != 1:
            raise ValueError(f"Expected one limit-force sensor for {model.joint(joint).name}")
        addresses.append(model.sensor_adr[ids[0]])
    addresses = np.array(addresses)
    rows = []
    original = MyoAssistLegBase._accumulate_physics_costs

    def capture(self, dt, weight):
        original(self, dt, weight)
        data = self.sim.data.ptr
        rows.append(
            np.r_[
                data.time,
                data.qpos[qids],
                data.qfrc_actuator[dofs],
                data.sensordata[addresses],
                data.joint("pelvis_tx").qvel[0],
            ]
        )

    if config.env_params.joint_limit_penalty_mode != "substep_squared_torque":
        evaluator.env.close()
        raise ValueError("Physics audit requires substep_squared_torque integration")
    MyoAssistLegBase._accumulate_physics_costs = capture
    try:
        path = evaluator.evaluate(
            str(out),
            f"gait_strict_seed{seed}.json",
            velocity_mode=MyoAssistLegBase.VelocityMode.UNIFORM,
            target_velocity_period=2,
            min_target_velocity=config.env_params.min_target_velocity,
            max_target_velocity=config.env_params.max_target_velocity,
            max_timestep=steps,
            seed=seed,
            terminate_when_done=True,
        )
    finally:
        MyoAssistLegBase._accumulate_physics_costs = original
        evaluator.env.close()
    data = np.array(rows)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("Missing or nonfinite physics samples")
    physics_rate = config.env_params.physics_sim_framerate
    gait = json.loads(Path(path).read_text())
    summary = {
        "seed": seed,
        "mode": "strict",
        "checkpoint": step,
        "duration_s": len(data) / physics_rate,
        "physics_samples": len(data),
        "sample_rate_hz": physics_rate,
        "mean_velocity_m_s": float(data[:, 19].mean()),
        "episodes": gait["metadata"]["episodes"],
        "joints": {},
        "limit_measurement": "MuJoCo JOINTLIMITFRC sensor, absolute threshold 1e-6 N m",
    }
    for i, name in enumerate(names):
        angle = np.rad2deg(data[:, i + 1])
        torque = data[:, i + 7]
        limit = abs(data[:, i + 13])
        active = limit > 1e-6
        summary["joints"][name] = {
            "range_deg": ranges[i].tolist(),
            "angle_min_deg": float(angle.min()),
            "angle_max_deg": float(angle.max()),
            "distance_to_limit_min_deg": float(np.minimum(angle - ranges[i, 0], ranges[i, 1] - angle).min()),
            "limit_force_peak_Nm": float(limit.max()),
            "limit_nonzero_fraction": float(active.mean()),
            "limit_nonzero_duration_s": float(active.sum() / physics_rate),
            "first_limit_time_s": float(data[active, 0][0]) if active.any() else None,
            "active_torque_min_Nm": float(torque.min()),
            "active_torque_max_Nm": float(torque.max()),
            "active_torque_rms_Nm": float(np.sqrt(np.mean(torque**2))),
            "torque_rate_rms_Nm_s": float(np.sqrt(np.mean((np.diff(torque) * physics_rate) ** 2))),
        }
    np.savez_compressed(
        out / f"physics_strict_seed{seed}.npz",
        joint_names=names,
        time_s=data[:, 0],
        qpos_rad=data[:, 1:7],
        active_torque_Nm=data[:, 7:13],
        limit_sensor_Nm=data[:, 13:19],
    )
    (out / f"limit_audit_seed{seed}.json").write_text(json.dumps(summary, indent=2))
    if seed == 1234:
        time = np.arange(1, len(data) + 1) / physics_rate
        for kind, values, label, filename in (
            ("angles", np.rad2deg(data[:, 1:7]), "Angle (deg)", "joint_angles_with_limits"),
            ("limit", data[:, 13:19], "Limit sensor (N m)", "joint_limit_forces"),
            ("torques", data[:, 7:13], "Active moment (N m)", "joint_torques_physics_rate"),
        ):
            fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
            for i, name in enumerate(names):
                ax = axes.flat[i]
                ax.plot(time, values[:, i], lw=0.6)
                if kind == "angles":
                    ax.axhline(ranges[i, 0], color="red", ls="--", lw=0.7)
                    ax.axhline(ranges[i, 1], color="red", ls="--", lw=0.7)
                ax.set_title(name)
                ax.set_ylabel(label)
                ax.grid(alpha=0.2)
            for ax in axes[-1]:
                ax.set_xlabel("Elapsed time (s)")
            fig.suptitle(f"NoDevice | checkpoint {step:,} | strict seed {seed} | {physics_rate} Hz")
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                fig.savefig(out / f"{filename}.{suffix}", dpi=180)
            plt.close(fig)
        np.savetxt(
            out / "joint_limits_and_torques_physics.csv",
            data,
            delimiter=",",
            header=",".join(
                ["sim_time_s"]
                + [f"{n}_rad" for n in names]
                + [f"{n}_active_Nm" for n in names]
                + [f"{n}_limit_Nm" for n in names]
                + ["velocity_m_s"]
            ),
            comments="",
        )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--checkpoint", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235])
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    step = args.checkpoint or max(int(p.stem.split("_")[-1]) for p in (args.session / "trained_models").glob("model_*.zip"))
    out = args.session / f"human_joint_torques_{step}"
    out.mkdir(exist_ok=True)
    summaries = [audit(args.session, step, out, seed, args.steps) for seed in args.seeds]
    (out / "limit_audit_all_seeds.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
