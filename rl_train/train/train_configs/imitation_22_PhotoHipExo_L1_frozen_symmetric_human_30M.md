# PhotoHipExo_L1：冻结结构性镜像人体策略

将通过评估的NoDevice人体策略完整迁移至辅助策略，保留原MLP、镜像投影和有效肌肉动作标准差；冻结人体网络与肌肉动作标准差，只优化外骨骼网络和价值网络。人体策略仍依赖当前状态，不是固定播放动作序列；训练时保留来源策略的随机探索分布。

默认新训练30M步，目标1.25m/s，32环境；初始外骨骼均值输出为0。参考定义、观测76维、关节范围与train_session_20261104-182204一致。外骨骼采用现有PhotoHip配置：努力惩罚0.1、动作变化惩罚0.05、超出15N·m软限的惩罚2；软限不是硬件安全限位。

```bash
.my_venv/bin/python -m rl_train.run_train \
  --config_file_path rl_train/train/train_configs/imitation_22_PhotoHipExo_L1_frozen_symmetric_human_30M.json \
  --config.env_params.human_actor_init_path rl_train/results/train_session_20261104-182204/trained_models/model_33685504.zip \
  --config.total_timesteps 30000000
```

这里使用human_actor_init_path只迁移人体策略，不使用prev_trained_policy_path续训整个人体PPO。
检查点旁的session_config.json用于校验观测、参考与关节范围兼容性。在其他机器运行时调整路径。

辅助checkpoint持久化镜像映射，因此推理/加载无需重新读取人体来源zip；仓库加载评估时会重新冻结人体参数。
验证覆盖普通与镜像策略迁移、动作/标准差一致、PPO更新后人体不变且外骨骼更新、保存重载后人体不变；另用实际33,685,504人体检查点完成64步回放，人体动作精确匹配、可训练人体参数为0。
