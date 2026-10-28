# 人体随机化、传感器误差与电机动态

实现入口为 `rl_train/envs/robustness.py`，配置入口为
`env_params.robustness`。默认 `enabled=false`，旧配置保持理想观测和理想电机；
新增奖励 `exo_action_diff_penalty` 默认权重为 0。

PhotoHipExo 的完整示例配置：
`rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_robustness_example_20Nm.json`。
该配置沿用 65,536 步短程试训设置。数值仅用于验证接口，不是实测标定结果，也不表示训练收敛。

当前推荐的 2000 万步课程配置为
`rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_robustness_20M_20Nm.json`，
使用 32 个环境、每环境每次采样 512 步，沿用长程配置的 PPO 参数：

```bash
MUJOCO_GL=egl MPLBACKEND=Agg MPLCONFIGDIR=/tmp/myoassist_mpl \
MYOASSIST_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  .my_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_robustness_20M_20Nm.json
```

当前机器已在宿主环境验证 EGL 离屏渲染；训练使用 CPU，定期评估仍可生成 replay 视频。
完整 rollout 会将从零开始的实际总步数补齐至 20,004,864，步数为所有环境的合计，
不是 PPO 更新轮数。2000 万步是预算，不保证收敛；原 50M 配置保留用于复现实验。

新配置采用下述归一化模仿奖励和轨迹阈值课程，PPO、人体/传感器/电机扰动与 50M 配置相同。
前 500 万新增环境步将训练阈值由 0.3 线性收紧至 0.2，剩余训练保持 0.2。
阈值在每次 rollout 开始时同步到所有 worker，达到 500 万步后的首个 rollout 使用 0.2；
日志字段 `training_trajectory_threshold` 记录该批次采样实际使用的值。
旧配置默认关闭课程，行为保持不变。

自动评估同时生成 relaxed 和 strict 单回合结果，各最多 600 步（20 秒）。
严格评估始终使用 `out_of_trajectory_threshold=0.2`，不使用训练课程阈值；
宽松评估仍放开轨迹限制。高度条件保持 0.7，评估默认关闭扰动。
不能直接比较不同课程阈值下的训练回合长度来判断收敛，应比较固定严格标准下的结果。

配置默认从零训练。要从已有检查点续训，在上述命令后追加：

```bash
--config.env_params.prev_trained_policy_path rl_train/results/train_session_20261027-215530/trained_models/model_50003968.zip
```

续训保留检查点累计计数，但预算为额外 2000 万步，课程从本次 `learn()` 的零新增步开始。
不自动选择历史训练奖励最高的检查点；建议先以相同起始种子比较最终模型与 model_46399488。
新增课程参数位于 `env_params.trajectory_curriculum`：
`enabled=true`、`initial_threshold=0.3`、`anneal_timesteps=5000000`。

长程配置的 checkpoint/日志间隔从 8 改为 16 个 rollout（262,144 步），
评估间隔从默认的 64 翻倍至 128 个 rollout（2,097,152 步）。
它们都是固定 rollout 间隔，不是均分总步数后的固定评估次数。
首个 rollout 额外保存一份，训练结束时确保保存经过最后一次 PPO 更新的最终权重，
并为最终 checkpoint 写入可供 replay 读取的记录。
回调使用从 1 开始的 rollout 编号；完整模仿学习记录写入后才启动评估，避免读取上一次保存的记录。
短程示例保持原有评估间隔 1000，checkpoint 间隔从 1 增至 16。

独立 replay 默认使用 `relaxed`；20M 配置显式安排 relaxed 和 strict 两种评估。
该模式将参考轨迹偏差阈值放宽至 1e6、回合步数上限放宽至 1e9；
保留骨盆高度低于 0.7 m 等结束条件，结束时停止单回合回放，不拼接 reset 后的片段。

```bash
MPLCONFIGDIR=/tmp/myoassist_mpl MYOASSIST_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .my_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_robustness_example_20Nm.json
```

## 归一化模仿奖励

20M 配置设置 `env_params.imitation_reward.mode="normalized"`，每类模仿项按以下公式计算：

```text
component = dt * sum(w_j / sum(w) * exp(-(error_j / scale_j)^2))
contribution = overall_weight * component
```

`reward_keys_and_weights.qpos_imitation_rewards` 和 `qvel_imitation_rewards` 的字典只表示
关节相对重要性。将整组关节权重同比例放大，不会改变奖励；整体强度由
`imitation_reward.qpos_weight=4.0`、`qvel_weight=1.0` 单独控制。
两项在零误差下的每秒贡献上限分别为 4 和 1。原 50M 配置的上限为 15.21 和 1。
每个有正权重的关节都必须配置有限且大于零的误差尺度，缺失或非法值在创建环境时直接报错。

| 关节 | qpos_error_scales | qvel_error_scales |
|---|---:|---:|
| pelvis_ty | 0.03 m | 0.2 m/s |
| pelvis_tilt | 0.1 rad | 0.5 rad/s |
| 左右髋、膝、踝 | 0.15 rad | 1.0 rad/s |

尺度表示单关节得分下降到 exp(-1)（约 36.8%）时的误差，不是终止阈值。
膝踝角度误差为 0.2 rad 时，单关节得分约 16.9%，原公式约为 72.6%。
该设计仍是加权平均奖励，不能保证所有关节始终合格；严格终止条件继续独立检查任意关节越界。
以上参数用于首轮对照，并非已验证的最优值；要按固定严格评估、速度误差、回合完成率及肌肉用力判断效果。

本次仅改变奖励聚合与误差尺度。原有参考相位推进、qvel 的瞬时速度比缩放和策略观测接口保持现状，
因此参考位置/速度一致性和相位信息缺失仍需独立诊断；不能将本次调整视为已解决全部轨迹偏离问题。
人体用力、外骨骼用力、动作变化和速度跟踪等其他奖励公式与权重保持原值。

新日志保存 `imitation_reward_mode` 和 `effective_reward_weights`。加权贡献图优先使用
每条日志记录的实际外层系数，避免再次乘关节权重总和；没有新字段的历史日志继续使用旧口径。
`reward_log_schema_version=2` 仍表示回合累计统计口径，奖励公式版本另由 mode 区分。
旧配置省略 `imitation_reward` 时默认 `legacy`，保留历史奖励及图表语义。
新旧公式的总奖励数值不可直接作为性能提升对比。建议用20M新配置从零训练作为清晰基线，
也可加载旧检查点微调，但旧价值网络和优化器状态来自旧奖励尺度，初期需观察价值损失。

## 回合级参数采样

所有范围均为 `[下限, 上限]`，连续参数均匀采样，延迟步数为离散均匀采样。
每次 reset 从保存的标称参数重新计算，避免缩放累积；使用环境的种子生成随机数。
reset 和 step 的 `info["robustness"]` 返回本回合的参数、实际量化后的电机延迟和观测偏置。

| 参数 | 含义 | 示例范围 |
|---|---|---|
| `human_body_names` | 明确选择人体刚体，不包括附着的外骨骼刚体 | 示例列出 PhotoHipExo 组合中的人体刚体 |
| `human_mass_scale_range` | 所选人体刚体共同的质量倍率 | 0.98–1.02 |
| `human_inertia_scale_range` | 在质量同比例缩放惯量之后，再乘独立的惯量倍率 | 0.98–1.02 |
| `muscle_strength_scale_range` | 全部肌肉共同的最大力倍率 | 0.95–1.05 |
| `observation_delay_steps_range` | 运动学和足部力观测延迟，单位为控制步 | 0（暂时关闭） |
| `motor_delay_seconds_range` | 外骨骼指令传输延迟，单位为秒 | 0–0.003 |
| `motor_time_constant_seconds_range` | 电机一阶响应时间常数 τ，单位为秒，0 表示即时响应 | 0.00475–0.00525（5 ms ±5%） |
| `motor_gain_scale_range` | 电机输出增益倍率 | 0.98–1.02 |

惯量实际乘数为 `mass_scale * inertia_scale`。每个刚体的三个主惯量乘同一个正数，
保持惯量三角不等式。调用 MuJoCo `mj_setConst` 刷新派生量。
当前配置的惯量总倍率为 0.9604–1.0404，包含质量缩放后的总变化也在 ±5% 以内。
肌肉强度同时修改 MuJoCo 的肌肉 gain 和 bias 参数；对于自动计算最大力的肌肉，修改其 scale。
本实现不改变骨骼长度、肌肉路径、设备质量或绑缚几何，也不模拟柔性绑带。

## 传感器噪声、偏置与延迟

`observation_noise_std`、`observation_bias_std` 支持 `qpos`、`qvel`、`sensor` 三个组。
每组可以填单一标准差或按该组观测顺序排列的标准差数组。
单位与进入策略前的观测一致：角度为 rad，角速度为 rad/s，平移为 m、m/s。
当前足部力传感器已经除以系统重量，单位为体重倍数；示例噪声标准差 0.002 表示系统重量的 0.2%，不是 0.002 N。
示例为骨盆高度和关节角度指定了不同的标准差。
关节角度噪声标准差为 0.002 rad（约 0.11°）、偏置标准差为 0.003 rad（约 0.17°）；
骨盆高度噪声和偏置标准差各为 1 mm。角速度噪声和偏置标准差各为 0.01 rad/s，
骨盆平移速度为 0.005 m/s，足部力偏置标准差为系统重量的 0.3%。

- 噪声：每个观测时刻、每个通道独立采样零均值高斯噪声。
- 偏置：每个回合、每个通道采样一次零均值高斯偏置，回合内保持不变。
- 延迟：将加噪后的运动学和足部力观测放入 FIFO，返回过去的测量。
  初始历史用 reset 时的测量填充，避免跨回合泄漏。
- 肌肉激活和目标速度保持当前值，不加上述噪声或延迟。

这些处理发生在返回给策略的观测上，并且在 VecNormalize 之前。
仿真状态、奖励、终止条件和 `info` 中的真值诊断不受观测误差污染。
因此外骨骼和人体策略都会读取带误差的运动学/足部力输入；本版本没有为 critic 单设真值输入。

## 电机非理想模型

每个控制步先保留 MyoSuite 的动作映射和执行器限制，再在每个物理 tick 更新：

```text
策略指令 → 传输延迟 FIFO → 输出增益及饱和 → 一阶响应 → MuJoCo 执行器
```

延迟四舍五入到最近的物理 tick；以当前 1200 Hz 为例，分辨率约 0.833 ms。
一阶响应使用精确离散更新：

```text
alpha = 1 - exp(-physics_dt / tau)
output += alpha * (target - output)
```

`tau=0` 时 `alpha=1`。一阶系统的带宽为 `1/(2π tau)`。
初始电机输出和指令历史均为零，每回合重新初始化。
PhotoHipExo 的饱和仍沿用 ±20 N·m 限幅。
该模型没有实现扭矩—转速曲线、热限制、齿隙和柔性传动。

这与参数随机化不重复：一阶响应/延迟描述电机的动态形式，随机化描述不同回合的 τ、延迟、
增益如何变化。范围上下限相同，可以模拟参数固定的非理想设备。

## 外骨骼动作平滑奖励

```text
r_smooth = -dt * mean((a_exo[t] - a_exo[t-1])²)
```

使用归一化后的策略请求动作、在电机滤波之前计算，避免用滤波掩盖策略本身的高频变化。
首个动作不计变化惩罚，reset 清空历史。对 PhotoHipExo，动作从 -1 跳到 +1 对应
请求力矩从 -20 跳到 +20 N·m。当前短程示例和 5000 万步配置的权重均为 0，暂时关闭此奖励；旧配置默认也为 0。
该奖励随控制周期变化，切换控制频率后需重调权重；它不是硬性的力矩变化率限制。

## 评估、地面与速度边界

`enable_in_evaluation=false` 默认保留标称评估；要测随机条件下的鲁棒性，
在评估使用的配置中明确设为 true，再使用多个种子评估。
参数采样和延迟状态都不是网络权重的一部分，导出外骨骼策略时仍需保留训练时的观测归一化、
信号单位、通道顺序及控制周期。

当前 PhotoHipExo 配置 `terrain=null`：组合器生成无限平面，编译后地面及足部碰撞几何的
`friction=[1, 0.005, 0.0001]`，`condim=3`，`priority=0`。
三项依次为滑动、扭转和滚动摩擦参数；当前 3 维接触只启用法向和两个切向约束，
不启用扭转/滚动摩擦自由度。本次没有更改地面摩擦或增加摩擦随机化。

当前训练采用 PPO，并以参考动作的姿态/速度误差构造模仿奖励，不是监督学习控制标签。
目标速度支持范围采样与变化，但参考轨迹每控制步仍只前进一帧；代码仅按
`target_velocity / reference_forward_velocity` 缩放参考速度，没有同步重定时参考姿态。
所以扩大目标速度范围并不等于获得可靠的任意速度步态。需要进一步实现相位重定时或使用多速度参考，
再在所需速度范围训练、验证。本次保持目标速度 1.25 m/s 和现有参考推进方式。

## 本轮验证

相关的随机化、训练统计、评估边界测试共 26 项通过。
另使用长程配置缩短采样长度，完成双环境 192 步端到端检查：
第 128 步触发 relaxed 评估并生成视频，训练继续到 192 步，最终权重及日志均保存成功。
未收敛策略没有完整步态周期时，相应周期统计会跳过并记录在报告中，不中止训练。
此验证不表示 5000 万步训练已经完成或步态已收敛。
