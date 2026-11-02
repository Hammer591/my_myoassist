# 从无外骨骼人体检查点训练冻结人体的髋外骨骼

配置文件：`rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_frozen_human_hip35_30M.json`。
训练从新的 PPO 模型开始，不续用人体 PPO 的优化器或价值网络。只把已训练的 22 肌肉人体 actor 和它的动作噪声复制到带外骨骼策略中；人体 actor 权重与人体动作噪声保持冻结。双侧外骨骼使用现有左右共享的小网络，价值网络从头训练，两个外骨骼动作槽仍可学习。训练一开始，外骨骼平均输出为零，初始动作噪声标准差约 `exp(-2)`（归一化控制量）。

先完成和验收无外骨骼人体训练。只有确认它能稳定跟踪参考且不持续抵住 −35° 髋伸展下限，才将其用于助力阶段。以下命令中的检查点路径要替换成实际训练会话的最终模型：

```bash
MUJOCO_GL=osmesa MPLBACKEND=Agg MPLCONFIGDIR=/tmp/myoassist_mpl \
MYOASSIST_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
.my_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_frozen_human_hip35_30M.json \
  --config.env_params.human_actor_init_path rl_train/results/train_session_YYYYMMDD-HHMMSS/trained_models/model_30015488.zip
```

默认总预算为 3000 万合计环境步。必须显式提供来源检查点；配置中 `human_actor_init_path` 为 `null`，因此不会误用旧的人体模型。加载时要求来源使用 `HumanActorCriticPolicy`、76 维观测及 22 肌肉动作；如果检查点旁有训练会话的 `session_config.json`，还要核对参考、观测顺序、坐标范围与时间基准。旧的 44 维人体模型不能作为该阶段的来源。

新的带设备模型沿用人体的髋/踝限位、参考目标和 76 维观测。人体 actor 读取完整观测；外骨骼 actor 只读取当前双侧髋角、角速度和足部接触。现在的设备模型是 `PhotoHipExo_L1` 的双向 ±20 N·m 电机，并不是论文里的单向绳驱设备。

该阶段提供“人体完全冻结、训练外骨骼与价值网络”的对照。后续若进行人体共同微调，应另建配置和训练逻辑、明确不同学习率，并用穿设备但关闭助力的对照验证收益。本阶段没有加入历史观测，也没有声称已完成共同微调。

测试 `tools/tests/test_frozen_human_exo.py` 覆盖源/目标人体输出一致、冻结参数保持、外骨骼网络更新、保存恢复以及来源参考不匹配时拒绝加载。完整 3000 万步的助力效果仍需实际训练和评估。
