# 2026-10-27 训练诊断修复日志

本次针对 `train_session_20261026-140410` 与 `train_session_20261021-125938` 的对比诊断修复。修改基线为 `f64b321`。

用户要求：先核对官方输出性能，再修复已发现问题并上传 GitHub；暂不修改惩罚和 PPO；质量以用户测量为准。

## 同日追加：主回放改为宽松单回合

以下更新取代初版日志中“默认 strict”的选择，其余训练修复保持有效。

用户反馈 `train_session_20261027-171915` 的 `analyze_results_933888_00_strict/replay_00.mp4` 频繁重置。检查逐帧 episode ID 与结束原因：200 帧包含 52 个回合、51 次重置，已结束回合均因轨迹偏差终止，每回合仅 2～8 步。原因是初版将默认评估改为 strict，却保留了自动评估 `terminate_when_done=False` 的凑帧行为；这不适合作为展示策略完成过程的主回放。

- 自动评估、命令行与评估器默认使用 relaxed，允许轨迹偏差；仍保留身体高度等结束条件。
- 主回放只运行一次尝试，结束即停，不再 reset 拼接；显式 strict 也使用单回合。
- 严格诊断仍可用 `--evaluation-mode strict`，两种目录和元数据保持明确区分。宽松回放不能当作通过严格训练标准的证明。
- 旧缓存必须模式匹配、记录 `terminate_when_done=true` 且只有一个回合才可复用。命令行显式指定步数或变速时重新采样。
- 元数据增加检查点路径，便于追溯。训练 reset、奖励/惩罚、PPO 与设备质量均未修改。

使用训练包内 **model_933888.zip**（非最新检查点），种子 1234、确定性策略、目标 1.25 m/s、上限 200 步重新运行：连续 15 步后触发 `height`，只有 episode 0，零次中途重置；生成 15 帧、30 fps、0.5 秒视频。这个检查点仍无法稳定行走；视频短是本次尝试的高度终止结果。新文件位于训练包的 `analyze_results_933888_00_relaxed/`，原 strict 记录保留。

新增回归验证覆盖默认宽松越过轨迹误差、显式严格提前停止、单回合不重启、缓存兼容检查、自动分析入口及 CLI 默认值；评估边界测试共 8 项通过；完整 tools/tests 共 64 项及 2 个子测试通过，修改文件 Ruff 检查与格式检查通过。实际检查点评估和视频编码均完成。

## 同日追加：空步态周期统计警告

用户在训练进度 2,113,536、评估加载 model_1982464 时观察到 NumPy 的 `Mean of empty slice`、`Degrees of freedom <= 0` 和无效除法警告。本地未提供该检查点，不能确认该次采样的具体结束原因。使用已重生成的 model_933888 单回合数据复现同类路径：左右完整周期均为 0，`get_toe_off_average` 对空列表求均值；周期绘图随后还会对空数据求标准差。

修改为没有完整周期时抛出明确的“cycle statistics skipped”说明，由现有分析报告捕获并写入 `exceptions`。不计算虚假的零均值，不屏蔽 NumPy 警告；原始时序、接触图与回放仍可生成。需要周期统计的绘图在创建 figure 前验证数据，避免跳过时泄漏图形。

验证：空周期及正常周期均值测试、分析入口保留原始图和记录跳过原因测试、既有评估边界测试，共 11 项通过。使用真实 15 帧回放运行分析并将 RuntimeWarning 提升为异常，确认只记录 5 项缺少周期说明，原始图正常生成；Ruff 及 diff 检查通过。奖励、PPO 和训练参数未修改。

日志中的进度条与评估输出混排不代表模型加载失败。分析在 spawn 子进程中运行，但父训练进程通过 `pool.apply` 同步等待；因此评估期间进度条停留在相同步数是预期行为。这里的空统计不参与 PPO 更新，不能据此断言训练参数出现 NaN。

## 官方参数与模型约定

已核对用户指定官方仓库 main 的提交 `dadd9d543d9a4102f762fb453c67232aff1ed1a9`，读取使用说明书、二次开发手册及新主板电机控制代码。产品表提供峰值输出 20 N·m，但没有明确单侧/双侧口径、连续额定扭矩、峰值持续时间、额定转速/机械功率或扭矩—转速曲线。

**重量采用用户实测，保留现有约 2.55 kg 模型质量和惯量；不使用官方 3.2 kg（含电池）覆盖用户数据。** 保留此前指定的每侧 ±20 N·m 理想执行器。官方峰值不等于已验证的每侧持续输出。

完整来源及适用版本见 [官方输出指标核对](../../MyHipExo/OFFICIAL_OUTPUT_SPEC.md)。

## 修复内容

| 问题 | 修复 | 验证 |
|---|---|---|
| 随机相位被错传为速度上限 | 速度设置使用关键字参数；所有目标随机化使用环境 RNG | 多次 reset 和三种速度模式下，固定 1.25 的边界与实际目标均为 1.25 |
| reference qvel 使用旧回合目标，之后 base reset 又更换目标 | 先准备新目标，再选择参考相位并缩放初始速度；base reset 不再重复采样 | 初始 pelvis_tx 速度等于本回合目标；不同速度模式的同种子 reset 可复现 |
| 正弦/阶跃模式继承前一回合时间 | 正弦初值按新回合 t=0 计算，阶跃计时在 reset 完成后归零 | 经动力学运行后同种子重置仍得到相同初始目标和观测 |
| 多环境使用相同种子 | 每个 worker 工厂使用 `seed + rank` | 3 个工厂分别携带连续且不同的种子 |
| 每次 done 反复转存以前回合的奖励分项 | 转存后清空当前环境的分项累计；将标量复制为 float | 两回合回报 2、4，分项均值正确为 3 |
| 日志写出时丢掉未结束回合的分项 | 只清空已完成回合统计，保留未结束回合分项 | 异步回合跨3个日志窗口仍与实际回报一致 |
| 零完成回合除零、以第一个 worker 的键为准导致遗漏 | 零完成回合输出空分项并记录 completed_episodes=0；合并各 worker 的键 | 空窗口、缺失 info、不同键集合均不报错 |
| 图中模仿项权重因错误调用 `.values()` 回退到 1 | 图表采用与当前环境相同的外层权重和 | 已加权分项2 × 外层权重和1.5，在图中显示3；不改变奖励计算 |
| 图名 proportion 暗示百分比，历史数据无可靠性标记 | 图标题/纵轴改为回合分项及加权贡献；历史记录标注 unverified | 新字段兼容旧 JSON；绘图测试覆盖首条有效记录 |
| 自动评估悄悄关闭轨迹终止限制 | 默认 strict 使用训练配置；relaxed 显式选择；配置使用深拷贝 | 两种阈值与步数限制正确，评估不改训练配置 |
| 连续追加多个回合后按一条轨迹分段 | 保存逐帧 episode ID 和每回合结果，分段算法在每个回合内独立运行 | 传感器及髋屈曲备用分段入口均不跨 reset |
| 初始化评估环境后立即 close | 初始化时保留环境，正常评估结束再关闭 | 实际环境无渲染评估完成并可保存/读取 metadata |
| 旧宽松评估缓存容易与新评估混用 | 输出目录增加 `_strict` / `_relaxed` 后缀 | CLI 提供 `--evaluation-mode` 并标注实际采样帧数 |

## 新日志字段

训练 `train_log.json` 每条记录新增：

- `reward_log_schema_version=2`：标记修复后的累计口径；旧文件未包含此字段时按0读取。
- `completed_episodes`：本日志窗口统计的完成回合数。均值为0且该字段为0表示无样本，不能解释为训练失败。
- `target_velocity_stats`：日志窗口内实际目标的 min/mean/max/samples，覆盖窗口中的所有环境步，包括尚未结束的回合。
- `termination_counts`：高度、时间上限、参考数据耗尽及 `trajectory:<joint>` 等结束原因计数；同一回合可能同时出现多个原因。

评估 `gait_evaluated_data*.json` 的 metadata 新增：

- `evaluation`：strict/relaxed、seed、确定性策略、实际轨迹阈值/回合上限、请求帧数和是否遇到 done 就停止。
- `episode_ids`：每个采样帧所属回合。
- `episodes`：起止帧、长度、回报、terminated/truncated、终止原因、越界关节误差、是否达到评估预算。

`reward_accumulate` 保留原语义：从训练开始的累计量，并非窗口或回合均值。参考数据耗尽时环境仍沿用原有“最终返回回报置零”的行为；此时 dense 与实际回报可能有真实差异，可由 reference_exhausted 终止原因识别。

## 保留项与历史结果

- 未修改任何奖励公式或惩罚系数；qpos 模仿奖励现有的两层权重处理也保留，只修复图表使其准确反映当前计算。
- 未修改 PPO 参数、策略网络、mirror 系数、训练配置 JSON、物理 XML/YAML、设备质量或惯量。
- 未改写历史训练日志、检查点、评估图及视频。历史重复累计的分项无法仅凭汇总日志精确还原，不作伪修复。
- 未启动新的3000万步训练。128步测试只验证训练路径与日志一致性，不说明修复后策略性能已经提升。

修复前统一当前代码、固定1.25 m/s的检查点诊断记录（严格30个起始种子，宽松5个种子）：

| 检查点 | 严格平均控制步数 | 严格中位数 | 宽松20秒完成数 | 宽松速度 RMSE |
|---|---:|---:|---:|---:|
| Hippo 9,977,856 | 47.17 | 49.5 | 5/5 | 0.165 m/s |
| Photo 9,977,856 | 25.83 | 18.5 | 5/5 | 0.213 m/s |
| Photo 29,900,800 | 57.87 | 37.5 | 5/5 | 0.195 m/s |

这是修复前的复测基线，不是修复效果对比。历史依赖/模型没有完整冻结，不能据此对设备能力作单一因果归因。

## 验证记录

测试命令（环境中有 ROS 的 pytest 插件，需关闭自动加载；PYTHONPATH 确保旧 editable 安装的子进程能找到 MyHipExo）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  .my_venv/bin/python -m pytest tools/tests -q
PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  .my_venv/bin/python test_setup.py
```

最终结果：`tools/tests` **59 passed**（另有2个subtest），耗时94.28秒；`test_setup.py` **16/16通过**。包括真实128步 MirrorPPO 更新和落盘日志一致性检查。测试中的 scipy.misc 弃用、无可用X11显示及CUDA驱动兼容提示均为现有环境警告，本次功能测试在CPU/无渲染模式下通过；没有声称验证了视频渲染或真实设备输出。

另使用 AST 对比确认 `_calculate_base_reward`、`_calculate_reward_per_step`、`_calculate_imitation_rewards` 和两个环境的 `get_reward_dict` 与基线完全一致；git 差异确认训练参数、策略/PPO代码以及设备物理配置未修改。

全库 Ruff 检查发现155项既有样式问题，均位于此前的 `MyHipExo/build_model.py`、`build_viewer.py`、`mass_properties.py` 及4个 render 脚本。对 `f64b321` 对应文件复查得到相同155项；全库格式检查另列出8个既有未格式化文件。本次修改的13个Python文件 lint/format 均通过，未为本任务批量改写模型生成与渲染脚本。

## 使用方式

重新启动训练后自动采用修复后的速度和统计；旧检查点可以加载，但原先训练过的错误目标分布不会被代码修复自动消除。

```bash
.my_venv/bin/python -m rl_train.run_policy_eval rl_train/results/train_session_20261026-140410 --evaluation-mode strict --no-show
.my_venv/bin/python -m rl_train.run_policy_eval rl_train/results/train_session_20261026-140410 --evaluation-mode relaxed --steps 600 --regen --no-show
```

定期评估的 `evaluate_param_list` 条目可显式设置 `"evaluation_mode": "strict"` 或 `"relaxed"`，省略时为 strict。短回合严格评估可能没有完整步态周期，属于评估结果，不能用跨 reset 的拼接周期补齐。
