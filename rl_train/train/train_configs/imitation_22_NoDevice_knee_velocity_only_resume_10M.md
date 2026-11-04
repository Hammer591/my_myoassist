# 仅提高膝角速度跟踪：从最新检查点续训10M

起点：`/home/hammer/train_session_20261103-200744/trained_models/model_30015488.zip`。
新增10,000,000个环境步，不是PPO更新轮数；保留累计步数，32环境×512步的rollout批量会将终点取整至约40,026,112步。

奖励基线为`imitation_22_NoDevice_hip35_knee3_smooth_symmetric_30M.json`。
唯一奖励变化是左右膝qvel相对权重各由0.1提高到1.0；qvel总系数仍为2。
前向速度奖励恢复1，激活变化奖励0.1，兴奋变化惩罚0.02，拮抗惩罚0.1且仅双髋。
力矩变化率惩罚0.03、结构性镜像和膝+3度边界保留。未启用参考轨迹镜像增强。

检查点已经历组合奖励训练；切换奖励不会抹掉已学习行为，也会保留网络/优化器状态。
因此此实验衡量从最新策略切换到该目标后的适应效果，不能单独证明此前窄峰改善由膝速度权重导致。
要做严格归因，需要各组从调整前的同一个检查点开始，并保留原奖励继续训练对照。

运行（仓库根目录）：

```bash
.my_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/imitation_22_NoDevice_knee_velocity_only_resume_10M.json
```

权重文件不上传GitHub；在其他机器运行时，先放置该checkpoint，并修改JSON中的`env_params.prev_trained_policy_path`为实际路径。
训练会创建新的session目录，不覆盖原始会话。
