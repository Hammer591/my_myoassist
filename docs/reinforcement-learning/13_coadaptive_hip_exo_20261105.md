# 从冻结助力模型进行人体与外骨骼协同微调

## 论文结论及实现边界

参考用户提供的 Yuan 等（2026），*Multi-Agent Reinforcement Learning of Co-Adaptive Hip Exoskeleton Control (MARL-Exo) for Metabolic Energy Reduction*。

论文没有报告“固定人体策略 vs 解冻人体策略”的消融，因此不能用它证明冻结人体必然使助力收益有限。
第3页 §II-C 讨论的是“两者从零开始联合训练可能不稳定”，采取人体先学稳定步态、再加入设备同时优化两种策略的两阶段安排。
第4页算法1分别计算人体和设备奖励、GAE及PPO更新；式(3)包含设备动作一阶/二阶差分、幅值、能量、弱人体奖励耦合及可选功率项。
该页还说明：先得到 Metabolic + Power 策略，再从其收敛检查点去掉功率项继续协同训练；部署的是后者。
第3页表I说明每个智能体有自己的critic，第5页表II给出两者均为1e-4的学习率。

当前项目已经额外做过“冻结人体、训练设备”的中间阶段。本实现从该阶段最终模型继续协同微调，是利用已有检查点的工程适配，并非逐项复现论文。

| 项目 | 论文 MARL-Exo | 当前实现 |
|---|---|---|
| 人体 | 304肌肉，残差关节目标→SPD→辅助肌肉网络 | 保留myolegs22直接肌肉控制与现有参考跟踪奖励 |
| 生理能耗 | Uchida/Umberger髋肌代谢功率，150 W归一化 | **显式命名的髋肌激活平方代理**，没有实现或声称Uchida代谢 |
| 优化 | 两个策略、两个价值函数、各自奖励/GAE/PPO目标 | 同样保留这四个分离环节；单个Adam管理互不重叠的两组参数 |
| 人体对称性 | 镜像损失 | 保留来源的结构性镜像投影，并共享对称动作的有效探索标准差 |
| 设备观测 | 18维历史、延迟与滤波 | 保留来源8维瞬时髋角/角速度/足接触，维度与顺序不变 |
| 频率 | 控制50 Hz、物理500 Hz | 保留来源30 Hz、1200 Hz，兼容已有actor |

缺少经校准的肌纤维长度、肌肉质量和纤维类型等参数时，激活平方不能改名为“代谢功率”。
这里只实现协同训练及奖励细化路线；不把文中13.8%的模拟代谢改善或18.3%的人体改善当作本项目预期结果。

## 新训练结构

- 人体和设备动作仍组成同一次24维环境动作，环境每步只推进一次。
- 人体奖励沿用当前步态、肌肉努力及关节限制目标，移除设备专属的三个惩罚项。
- 设备奖励包括髋肌努力代理、请求动作的一阶/二阶差分、幅值、功率项、人体奖励耦合和15 N·m软限超限成本。
- 两个价值网络分别预测人体和设备回报；设备critic只读取设备actor已有的紧凑观测。
- 两个独立GAE、独立优势标准化及独立动作log-probability比率，分别计算clip PPO目标；不是把24维动作当成同一个智能体。
- KL超限后仅停止对应智能体；显式清空该组梯度以防Adam动量继续改变已停止的参数。各组分别进行梯度范数裁剪。
- 人体网络和肌肉动作噪声均可更新；继承的镜像均值投影及有效对称标准差在解冻后继续保持。
- time_limit截断分别对两个价值函数bootstrap；真正终止和reference_exhausted不bootstrap，不跨重置传播优势。

设备代理目标的形式为：

```text
rE = dt * [
    max(1 - mean_hip_activation_squared / effort_reference, -5)
    - 0.2 * sum(delta_action^2)
    - 0.2 * sum(second_delta_action^2)
    - 0.01 * sum(action^2)
] + power_weight * (positive_work_J - 5 * negative_work_J)
  + 0.02 * rH + torque_excess_weight * existing_torque_excess_reward
```

髋肌默认选择双侧hamstrings、glutmax、iliopsoas、rectfem，共8个肌肉执行器；`effort_reference=0.25`是无量纲激活平方的固定尺度，并非150 W。
激活平方及机械功在物理子步积分。奖励中率项乘dt以适配当前30 Hz系统；一阶/二阶差分仍在控制步定义，其权重不应被解释为与论文完全相同的物理平滑强度。
保留原有的机械限位、参考边界及终止标准，不用放宽终止条件隐藏起步失败。

## 阶段A：从已有冻结检查点开始协同微调

配置：`rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_coadaptive_power_10M.json`。
默认来源为已评估的`train_session_20261104-211512/trained_models/model_30015488.zip`。
会复制两个actor、有效动作噪声和原critic作为人体critic；新建独立设备critic及优化器。
阶段计数从0开始，来源步数30,015,488、SHA256及转移方式写入`session_config.json:coadaptation_provenance`和checkpoint。
原检查点不会被覆盖。

```bash
MUJOCO_GL=osmesa MPLBACKEND=Agg MPLCONFIGDIR=/tmp/myoassist_mpl \
OPENBLAS_NUM_THREADS=1 MYOASSIST_NUM_THREADS=2 \
.my_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_coadaptive_power_10M.json
```

默认16环境、每环境512步，合计8192步一轮；batch_size=512、4个epoch、gamma=0.99、GAE=0.99、entropy=0、学习率1e-4、目标KL=0.01。
保留项目0.5梯度范数裁剪作为稳定性措施。默认10M是可调整的工程预算，不代表论文收敛预算。
在其他机器可通过`--config.env_params.coadaptation.init_path PATH`修改来源。

`init_path`表示开始新阶段并重建优化器；`prev_trained_policy_path`表示恢复已有协同训练，二者不能同时设置。
来源检查包括观测语义/顺序、参考、控制频率、关节范围、动作布局与镜像映射，不能只看向量维度。

## 阶段B：去掉功率引导，继续优化努力代理

先检查阶段A的稳定性、配对努力及功率曲线；稳定后选择其检查点。
配置：`imitation_22_PhotoHipExo_L1_coadaptive_effort_10M.json`，唯一奖励变化是`power_weight=0`。
必须显式提供阶段A协同检查点，不能误用冻结PPO文件。

```bash
MUJOCO_GL=osmesa MPLBACKEND=Agg MPLCONFIGDIR=/tmp/myoassist_mpl \
OPENBLAS_NUM_THREADS=1 MYOASSIST_NUM_THREADS=2 \
.my_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_coadaptive_effort_10M.json \
  --config.env_params.prev_trained_policy_path rl_train/results/实际阶段A会话/trained_models/实际协同检查点.zip
```

此时恢复两种策略、两个critic、两个动作噪声及优化器，时间步数延续计数，预算是额外10M。
该阶段名称为Effort，不称Metabolic，以区分本项目代理与论文真实生理模型。

若直接做不带功率项的协同实验，可用阶段B配置的`--config.env_params.coadaptation.init_path`指向原冻结模型；这属于单独消融，不是上述顺序的完整流程。

## 日志、评估及验证

`train_log.json:agent_metrics`保存human/exo各自的奖励、KL、value loss、explained variance、KL停止状态，以及努力/功率/动作差分分项。
训练入口及自动分析器均可识别新策略类型，保存后推理不再依赖原冻结检查点。

```bash
MUJOCO_GL=osmesa MPLBACKEND=Agg MPLCONFIGDIR=/tmp/myoassist_mpl \
.my_venv/bin/python -m rl_train.run_policy_eval rl_train/results/实际协同会话 \
  --no-show --regen --steps 600 --evaluation-mode strict
```

验收应保留三种对照：原冻结策略、当前协同策略开启助力、当前协同人体策略穿设备关闭助力。
对齐初始参考、种子和实际速度；单独统计跌倒、参考耗尽和预算完成，不能把不同长度的跌倒轨迹混入节能均值。
当前来源模型的seed1236存在零助力同样跌倒的问题；协同微调是否解决它必须重新测量。

测试覆盖真实MuJoCo更新前后两个actor/critic变化、人体探索噪声解冻、镜像保持、分离似然及GAE、停止后Adam不再移动该组、差分历史重置、来源不兼容拒绝和保存恢复。
此外用实际30,015,488步模型通过正常CLI完成128步冒烟训练；这仅验证训练链路，不是完整训练或收益证明。

本次验证：26项针对性/回归测试通过，Ruff及git diff --check通过；实际来源检查点的单环境和2个子进程环境各完成128步CLI训练，并对保存后的协同模型完成16步无渲染严格回放。冒烟会话均标记`SMOKE_TEST_ONLY.md`，不应当作完整微调成果。
