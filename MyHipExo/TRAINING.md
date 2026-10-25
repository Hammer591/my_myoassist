# PhotoHipExo 人体绑定与训练

设备注册键：`PhotoHipExo_L1`。支持 `myolegs22`、`myolegs26`，通过 MyoAssist 的 RL/CO 共用组合入口使用。
注册实现在 `myoassist_utils/devices.py`，无需修改已安装的 assist_sim；
`python -m assist_sim list` 仅列出该依赖自带的设备。查看完整组合：

```bash
.my_venv/bin/python -c 'from myoassist_utils.devices import available_combinations; print(available_combinations())'
```

## 绑定约定

沿用 Hippo 的理想刚性绑定：腰架、主舱、电机壳体固定到 pelvis；两侧输出座、连杆和固定器分别固定到 femur_r/l。驱动器 `Exo_R`、`Exo_L` 直接作用于 `hip_flexion_r/l`，正输出为屈髋。

设备和骨骼局部坐标均为 X 向前、Y 向上、Z 向右。人体骨盆到髋轴的局部位置为 `(-0.0707, -0.0661, ±0.0835)` m；腰架原点平移到 `(-0.0707, -0.0661, 0)` m，无额外旋转。因此中立站姿主舱竖直，运动中随骨盆转动。

保留原设备 373.657 mm 电机轴心间距和原网格尺寸。人体髋关节中心间距为 167 mm，电机壳体中心位于每侧髋中心外侧约 103.329 mm；两电机转轴与人体髋屈伸轴共线，壳体中心无需进入骨盆。固定器沿原连杆位于大腿远端，随股骨刚性运动。三视角见 `binding_preview.png`。

`L1model.xml` 保留原始两个被动关节；训练配置使用 `L1bound.xml`，在原几何零位去掉这两个自由度，保证固定器不会自由摆动。`build_model.py` 重建时同步生成该训练版本，并保留已校准的 YAML。当前与 Hippo 一样属于理想绑缚模型，不计算柔性绑带、人机接触或关节错位载荷。若要模拟被动铰链的实际运动，需要另建完整约束链。

## 开始训练

从项目根目录运行：

```bash
.my_venv/bin/python -m rl_train.run_train --config_file_path rl_train/train/train_configs/device_sweep/imitation_22_PhotoHipExo_L1_h128_e32_sidenet_mirror0p1_actpen10.json
```

该配置沿用 Hippo 的 22 肌肉人体、双侧外骨骼网络及镜像设置，只替换设备键。动作最后两维依次控制右、左电机。用户指定训练限值：每侧 `[-1, 1]` 输入对应 `[-20, 20]` N·m，实际额定/峰值扭矩仍待提供；这不是已校准的电机性能。修改 YAML 的 `gainprm` 可调整限值。质量和惯量保留 V13 数据及其原有估算假设。

短程试训使用以下配置：65,536 步、单环境、每次采样 512 步、batch size 512、10 个优化 epoch。使用单环境是因为当前沙箱不允许多进程通信。其余网络、参考轨迹、奖励和终止阈值沿用原训练配置。

```bash
MPLCONFIGDIR=/tmp/myoassist_mpl MYOASSIST_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .my_venv/bin/python -m rl_train.run_train --config_file_path rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_trial_20Nm.json
```

`validation/front_side_20Nm.png` 是实际训练环境模型在中立站姿下的正、侧视图；`validation/training_model_20Nm.xml` 是从该环境导出的模型，包含训练时设置的关节参数。图中仅关闭了肌腱和接触点辅助显示。

导出可独立打开的人体组合模型：

```bash
.my_venv/bin/python -c 'from myoassist_utils.env_spec import EnvSpec; EnvSpec("myolegs22", "PhotoHipExo_L1").validate().compose(export_path="/tmp/photo_hip_human.xml")'
.my_venv/bin/python -m mujoco.viewer --mjcf=/tmp/photo_hip_human.xml
```

## 验证范围

```bash
.my_venv/bin/python -m unittest tools.tests.test_photo_hip_exo -v
```

检查两个人体版本的加载、主舱中立姿态、髋轴同轴、不同髋角下固定器相对股骨不动、整机约 2.55 kg、驱动器顺序及 ±20 N·m 输出。另已运行实际训练环境的 128 步 MirrorPPO（44 维观测、24 维动作）；这只证明训练接入和梯度更新可运行，未验证步态收敛或真实助力效果。

最新 ±20 N·m 试训已完成 65,536 步。相同 20 个起始种子的确定性复测中，平均回合长度由 1.20 提高到 2.25 个控制步，但仍未学会稳定行走。完整指标与检查点位置见 `validation/REPORT.md`。
