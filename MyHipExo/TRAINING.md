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

人体参数随机化、传感器误差和电机延迟/带宽的可选配置见
[鲁棒训练说明](../docs/reinforcement-learning/08_robustness.md)。旧配置默认关闭这些功能。

5000 万步、温和随机化、关闭电机动作平滑奖励的配置：

```bash
MUJOCO_GL=egl MPLBACKEND=Agg MPLCONFIGDIR=/tmp/myoassist_mpl \
MYOASSIST_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  .my_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_robustness_50M_20Nm.json
```

该配置使用 32 环境，checkpoint 每 16 个 rollout 保存，relaxed 评估每 128 个 rollout 触发。
实际采样补齐完整 rollout 至 50,003,968 步；结束时保存最后一次更新后的权重。

从项目根目录运行：

```bash
.my_venv/bin/python -m rl_train.run_train --config_file_path rl_train/train/train_configs/device_sweep/imitation_22_PhotoHipExo_L1_h128_e32_sidenet_mirror0p1_actpen10.json
```

该配置沿用 Hippo 的 22 肌肉人体、双侧外骨骼网络及镜像设置，只替换设备键。动作最后两维依次控制右、左电机。用户指定训练限值：每侧 `[-1, 1]` 输入对应 `[-20, 20]` N·m。2026-10-27 核对官方手册后，确认产品表列有“峰值输出 20 Nm”，但未说明单侧/双侧口径，也未给出连续额定扭矩、峰值持续时间或扭矩—转速曲线；当前理想执行器不代表连续输出能力已校准。详见 [官方指标核对](OFFICIAL_OUTPUT_SPEC.md)。质量以用户实测为准；2026-10-30 按用户要求在后腰主舱增加 0.600 kg，当前整机 3.15 kg，不按官方含电池整机重量缩放。

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

检查两个人体版本的加载、主舱中立姿态、髋轴同轴、不同髋角下固定器相对股骨不动、整机约 3.15 kg、驱动器顺序及 ±20 N·m 输出。另已运行实际训练环境的 128 步 MirrorPPO（44 维观测、24 维动作）；这只证明训练接入和梯度更新可运行，未验证步态收敛或真实助力效果。

最新 ±20 N·m 试训已完成 65,536 步。相同 20 个起始种子的确定性复测中，平均回合长度由 1.20 提高到 2.25 个控制步，但仍未学会稳定行走。完整指标与检查点位置见 `validation/REPORT.md`。

## 2026-10-27 训练统计与评估修复

已修复目标速度参数错位、奖励分项重复累计及跨日志窗口丢失、多个环境使用相同种子等问题。惩罚、奖励公式、PPO 设置与设备物理参数保持不变；历史训练日志不会自动变成修复后的统计口径。完整修改与测试记录见 [修改日志](../docs/reinforcement-learning/07_training_fixes_20261027.md)。

默认评估采用 relaxed 单回合连续回放，允许偏离参考轨迹；触发高度等结束条件即停止，不再 reset 凑足帧数。strict 用于显式检查训练终止阈值：

```bash
.my_venv/bin/python -m rl_train.run_policy_eval rl_train/results/train_session_20261026-140410 --evaluation-mode strict --no-show
.my_venv/bin/python -m rl_train.run_policy_eval rl_train/results/train_session_20261026-140410 --evaluation-mode relaxed --steps 600 --no-show
```

输出目录带 `_strict` / `_relaxed` 后缀，避免复用旧评估缓存。定期训练评估默认 relaxed；需要严格诊断时，在 `evaluate_param_list` 对应条目加入 `"evaluation_mode": "strict"`。旧的多回合拼接缓存会重新生成。两种模式均保存种子、有效阈值、每回合长度、结束原因及 episode ID，步态分段不会跨 reset。


## 2026-10-30 裸人体基线与主舱增重

裸人体资产来自已安装的 `myo_sim`，通过 `assist_sim.loading.load_msk` 加载；
项目组合入口现在支持 `device_key: "NoDevice"`，仍使用相同地面和人体资产。
现有 `imitation_tutorial_22_separated_net_exo_off.json` 仍加载 Tutorial 设备，
且存在固定动作覆盖，不能当作完全不带设备的基线。

裸人体配置已对齐最新 40M 训练任务：`fixed_rate` 参考时序、normalized 模仿奖励
（qpos=8、qvel=1）、足力惩罚 2.0、关闭轨迹阈值课程，并包含 relaxed/strict 评估。
配套的从头训练助力配置为 `imitation_22_PhotoHipExo_L1_fixed_rate_50M_20Nm.json`；
原始 robustness_50M 配置保留历史设置，不作为这份裸人体配置的匹配对照。

启动裸人体训练：

```bash
MUJOCO_GL=egl MPLBACKEND=Agg MPLCONFIGDIR=/tmp/myoassist_mpl \
MYOASSIST_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  .my_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/imitation_22_NoDevice_robustness_50M.json
```

使用肌肉控制环境 `myoAssistLegImitation-v0`：44 维观测、22 维肌肉动作，
人体 actor 为 `[128, 128]`，critic 为 `[64, 64]`，没有外骨骼网络。
两组训练都从头开始，步数、速度 1.25 m/s、参考轨迹、人体奖励、终止阈值、
人体随机化与 PPO/镜像参数一致。最新历史 40M 策略包含续训历史，不能宣称其与
新训练的 50M 裸人体具有相同训练预算；严格对照应使用上述配套配置重新训练助力组。
裸人体配置保留电机随机化字段以保持配置一致；
无电机时这些字段不会施加任何力矩。

比较时，应在相同速度、参考轨迹、评估种子与结束条件下，比较稳定行走阶段的
全体及各肌肉平均激活、激活平方均值，并同时报告回合长度、速度与步态误差。
激活降低率可写为 `(裸人体均值 - 助力均值) / 裸人体均值 × 100%`。
不能把摔倒后的短片段或不同速度下的低激活解释为助力收益；肌肉激活降低也不等于
已经验证真实代谢消耗降低。建议多个训练种子复测。

质量变更由 `build_model.py` 中的 `total_mass=3.15`、`housing_added_mass=0.600`
生成，额外质量按主舱 `battery` 网格包络均匀分布，重算腰架刚体的重心与完整惯量；
其他部件质量分配保持原值。原始/训练/预览 MJCF 与物理属性报告已同步。
该分布是假设，后续有新增部件的实测安装位置与尺寸时可进一步校准。
已启动进程内的 MuJoCo 模型不会热更新；重启或恢复训练后重新加载才生效。
此前 2.55 kg 的训练/验证报告保留为历史记录。已通过裸人体 128 步 MirrorPPO 更新、两个人体版本绑定及增重惯量回归检查（3 项测试）。未启动完整 5000 万步训练。

本次沙箱使用 OSMesa 做无窗口验证，并在 Mesa 加载前导入 Torch 编译模块以避开原生库冲突：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=osmesa MPLBACKEND=Agg \
MPLCONFIGDIR=/tmp/myoassist_mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
.my_venv/bin/python -c 'import torch._dynamo; import pytest; raise SystemExit(pytest.main(["tools/tests/test_no_device.py", "tools/tests/test_photo_hip_exo.py", "-q"]))'
```
