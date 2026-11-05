"""Export signed motor and muscle-only hip moments from an assisted checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('MUJOCO_GL', 'disable')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/myoassist_mpl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from rl_train.analyzer.gait_analyze import GaitAnalyzer
from rl_train.analyzer.gait_data import GaitData
from rl_train.analyzer.gait_evaluate import ImitationGaitEvaluator
from rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig
from rl_train.utils.data_types import DictionableDataclass
from rl_train.utils.train_log_handler import TrainLogHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--checkpoint', type=int)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1234, 1235])
    parser.add_argument('--steps', type=int, default=600)
    args = parser.parse_args()
    torch.set_num_threads(2)
    step = args.checkpoint or max(int(f.stem.split('_')[-1]) for f in (args.session/'trained_models').glob('model_*.zip'))
    out = args.session / f'hip_assistance_moments_{step}'
    out.mkdir(exist_ok=True)
    raw = json.loads((args.session/'session_config.json').read_text())
    assert raw['env_params']['device_key'] == 'PhotoHipExo_L1'
    config = DictionableDataclass.create(ExoImitationTrainSessionConfig, raw)
    handler = TrainLogHandler(str(args.session))
    handler.log_datas = [SimpleNamespace(num_timesteps=step)]
    evaluator = ImitationGaitEvaluator(handler, config)
    evaluator.initialize_env(evaluation_mode='relaxed', render=False)
    env = evaluator.env.unwrapped
    model = env.sim.model.ptr
    dofs = np.array([model.jnt_dofadr[model.joint(f'hip_flexion_{s}').id] for s in 'rl'])
    assert len(env._muscle_actuator_ids) == 22 and len(env._device_actuator_ids) == 2
    original = MyoAssistLegBase._accumulate_physics_costs
    physics = []
    scratch = mujoco.MjData(model)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec))

    def capture(self, dt, weight):
        original(self, dt, weight)
        # Refresh force and moment-arm caches together on an independent state.
        # The live post-mj_step caches can correspond to different integrator stages.
        mujoco.mj_getState(model, self.sim.data.ptr, state, state_spec)
        mujoco.mj_setState(model, scratch, state, state_spec)
        mujoco.mj_forward(model, scratch)
        d = scratch
        moment = np.zeros((model.nu, model.nv))
        if d.actuator_moment.ndim == 2:
            moment[:] = d.actuator_moment
        else:
            for actuator in range(model.nu):
                start = d.moment_rowadr[actuator]
                end = start + d.moment_rownnz[actuator]
                moment[actuator, d.moment_colind[start:end]] = d.actuator_moment[start:end]
        muscle_ids = self._muscle_actuator_ids
        biological = (moment[muscle_ids][:, dofs] * d.actuator_force[muscle_ids, None]).sum(axis=0)
        total = d.qfrc_actuator[dofs].copy()
        # Only 22 muscles and 2 motors exist: subtract their generalized moments.
        exo = total - biological
        if not np.allclose(exo, d.actuator_force[self._device_actuator_ids], atol=1e-8, rtol=0):
            raise ValueError('Motor force and decomposed hip moment disagree at the same physics sample')
        velocity = d.qvel[dofs].copy()
        physics.append(np.r_[d.time, exo, biological, total, velocity, exo * velocity])

    MyoAssistLegBase._accumulate_physics_costs = capture
    fps = config.env_params.control_framerate
    rate = config.env_params.physics_sim_framerate
    phase = np.linspace(0, 100, 201)
    cycles = {side: [] for side in 'rl'}
    summaries = []
    try:
        for seed in args.seeds:
            physics.clear()
            gait_path = evaluator.evaluate(str(out), f'gait_seed{seed}.json',
                velocity_mode=MyoAssistLegBase.VelocityMode.UNIFORM, target_velocity_period=2,
                min_target_velocity=1.25, max_target_velocity=1.25,
                max_timestep=args.steps, seed=seed, terminate_when_done=True)
            arr = np.asarray(physics)
            arr[:, 0] = np.arange(1, len(arr)+1)/rate
            assert np.isfinite(arr).all()
            np.testing.assert_allclose(arr[:, 1:3]+arr[:, 3:5], arr[:, 5:7], atol=1e-10)
            gait = GaitData()
            gait.read_json_data(gait_path)
            assert len(arr) == gait.metadata['data_length'] * env.frame_skip
            # The PhotoHip motors have direct unit-gear hip transmissions.
            assert np.all(model.actuator_gear[env._device_actuator_ids, 0] == 1)
            np.savez_compressed(out/f'hip_moments_seed{seed}.npz', data=arr)
            np.savetxt(out/f'hip_moments_seed{seed}.csv', arr, delimiter=',',
                header='time_s,exo_r_Nm,exo_l_Nm,biological_r_Nm,biological_l_Nm,total_active_r_Nm,total_active_l_Nm,hip_r_velocity_rad_s,hip_l_velocity_rad_s,exo_r_power_W,exo_l_power_W', comments='')
            analyzer = GaitAnalyzer(gait, None, show_plot=False)
            counts = {}
            for i, side in enumerate('rl'):
                segments = analyzer.get_gait_segment_index(is_right_foot_based=(side == 'r'))
                accepted = []
                for start, toe_off, end in segments:
                    a, b = (start+1)*env.frame_skip-1, (end+1)*env.frame_skip-1
                    if a/rate < 1 or b <= a:
                        continue
                    values = arr[a:b+1, :]
                    profile = np.column_stack([np.interp(phase, np.linspace(0,100,len(values)), values[:, k]) for k in [1+i,3+i,5+i]])
                    cycles[side].append(profile)
                    accepted.append([int(start), int(toe_off), int(end)])
                counts[side] = accepted
            summaries.append(dict(seed=seed, metadata=gait.metadata, segments=counts,
                min_Nm=arr[:,1:5].min(axis=0).tolist(), max_Nm=arr[:,1:5].max(axis=0).tolist()))
    finally:
        MyoAssistLegBase._accumulate_physics_costs = original
        evaluator.env.close()

    colors = ['#176eaa', '#ce6230']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey='row')
    export = [phase]
    headers = ['gait_cycle_pct']
    for i, side in enumerate('rl'):
        values = np.asarray(cycles[side])
        assert len(values) > 0, f'No complete cycles for {side}'
        np.save(out/f'gait_cycles_{side}.npy', values)
        mean, sd = values.mean(axis=0), values.std(axis=0)
        for row, (name, label) in enumerate([('exo', 'Exoskeleton assistance'), ('biological', 'Biological hip moment')]):
            ax = axes[row, i]
            ax.plot(phase, mean[:,row], color=colors[i], lw=2)
            ax.fill_between(phase, mean[:,row]-sd[:,row], mean[:,row]+sd[:,row], color=colors[i], alpha=.2)
            ax.set_title(f'{"Right" if side=="r" else "Left"} hip: {label}\n{len(values)} complete strides')
            ax.set_ylabel('Moment (Nm)  |  + flexion / - extension')
            ax.axhline(0, color='0.4', lw=.7)
            ax.grid(alpha=.2)
            ax.set_xlim(0,100)
            export.extend([mean[:,row],sd[:,row]])
            headers.extend([f'{name}_{side}_mean_Nm', f'{name}_{side}_sd_Nm'])
        axes[1,i].set_xlabel('Own-leg gait cycle (%)')
    fig.suptitle(f'PhotoHipExo | checkpoint {step:,} | assistance ON | target 1.25 m/s\nMean +/- 1 SD across strides; seeds {args.seeds}; first second excluded')
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(out/f'hip_moments_gait_cycle.{ext}', dpi=180)
    plt.close(fig)
    np.savetxt(out/'hip_moments_gait_cycle.csv', np.column_stack(export), delimiter=',', header=','.join(headers),comments='')
    first = np.load(out/f'hip_moments_seed{args.seeds[0]}.npz')['data']
    fig, axes = plt.subplots(2,1,figsize=(12,7),sharex=True)
    for i, side in enumerate(['Right','Left']):
        axes[i].plot(first[:,0],first[:,3+i],label='Biological (muscles only)', color='#176eaa',lw=.8)
        axes[i].plot(first[:,0],first[:,1+i],label='Exoskeleton assistance', color='#ce6230',lw=.9)
        axes[i].set(ylabel=f'{side} hip moment (Nm)',xlim=(2,7))
        axes[i].axhline(0,color='0.4',lw=.6)
        axes[i].grid(alpha=.2)
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel('Elapsed time (s); positive = flexion, negative = extension')
    fig.suptitle(f'Assisted walking | checkpoint {step:,} | seed {args.seeds[0]} | 1200 Hz\nMuscle moment excludes motor, passive, contact and constraint contributions')
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(out/f'hip_moments_timeseries.{ext}',dpi=180)
    plt.close(fig)
    (out/'summary.json').write_text(json.dumps(dict(checkpoint=step,seeds=args.seeds,sample_rate_hz=rate,
        biological_definition='Sum of muscle actuator force times hip moment arm; excludes motor, passive and constraint forces',
        exo_definition='Total actuator generalized hip moment minus muscle-only generalized moment; verified against motor force for unit gear',
        sign='Positive hip flexion, negative hip extension, same coordinate signs for both legs',
        force_sampling='Forward dynamics caches refreshed on an independent MjData at each recorded physics state; original rollout state is unmodified',
        phase_definition='Each leg independently segmented by repository GaitAnalyzer (foot contact with hip-flexion fallback); cycles starting before 1 s excluded',
        band='Across-stride population SD, not a confidence interval',rollouts=summaries),indent=2))
    (out/'README.md').write_text(
        f'# 双侧髋关节助力与生物力矩\n\n最终模型：`model_{step}.zip`；开启助力；目标速度1.25 m/s；'
        f'种子{args.seeds}，使用前次评估中正常行走的两个名义起点。\n\n'
        '步态周期图上排为电机助力力矩，下排为肌肉产生的生物髋关节合力矩；左列右腿、右列左腿。'
        '各腿独立分段，排除起始不足1秒的周期；实线为跨周期均值，阴影为±1标准差，并非置信区间。\n\n'
        '所有力矩以N·m计，正值屈髋、负值伸髋。生物力矩仅为肌肉力乘髋关节力臂之和，'
        '不包含电机、被动力、接触和约束贡献。采样率1200 Hz，使用独立MuJoCo状态刷新力和力臂缓存，'
        '避免积分步骤之间缓存不同步；同时验证肌肉与电机力矩之和等于总主动关节力矩。\n\n'
        '`hip_moments_gait_cycle.png/.pdf/.csv`为归一化周期曲线；'
        '`hip_moments_timeseries.png/.pdf`为seed1234的2–7秒时域对照；'
        '`hip_moments_seed*.csv/.npz`保留完整时域数据。\n'
    )
    print(f'Curves saved to {out}')


if __name__ == '__main__':
    main()
