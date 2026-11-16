# 统一 latent planner：2026-11-16 迭代

目标保持为同一个 planner/policy 按命令通过 AH-BFM 执行传球与射门，并以自主运行验证；同环境连续技能切换仍需实现和测试。

## 作者权重审计

本地作者包 `AH-BFM/` 包含 actor、backward map、normalizer 与控制元数据。权重 SHA256 与 manifest 一致：`9d8b9ae1b775c4b57034a49a051e527c8470456b51093ff9f7d6467dceb895d9`。

在两技能各256个保存的仿真状态上，新 PyTorch actor 与现有 ONNX 最大输出差分别约 2.09e-7、3.58e-7。关节顺序、action rescale、控制间隔一致；latent 梯度可通过冻结 actor，actor 参数无梯度。Backward API 通过加载与范数检查，但31个身体点的仿真映射尚未确认，因此暂不使用未验证的语义正则。该包在本地忽略规则中，不隐式公开作者提供的源码或权重。

## 本轮实现

- `native_bfm.py`：作者模型加载、可微 latent-to-action 路径。
- `planner.py`：945维统一真值观测，含AH state/history、yaw局部机器人速度、球相对位置与速度、目标方向/球速、两维技能命令。单个共享网络预测5帧latent，每帧范数16，先执行首帧，50Hz重规划。
- `planner_collection.py`：每个执行控制步记录统一观测、实际latent、expert目标、episode及tick、done；明确标记专家辅助来源。
- `train_latent_planner.py`：平衡两技能，短序列latent回归与首帧解码动作监督，未来窗口用episode掩码防止跨reset。每环境最后25% lanes只用于离线验证。
- `evaluation.py`：同一checkpoint接受两种技能命令，使用现有严格自主评估协议。它是状态条件序列回归planner基线，尚不是flow matching。

训练种子811401（射门）、811402（传球），不使用开发评估种子811314。传球采集原s1自主轨迹；射门采集专家辅助64步反演轨迹，此数据源是引导训练的扩展，不能称为独立射门LPP已修复，也不等于FIBER的自主LPP数据方案。训练后必须自主验证，并针对学生实际状态继续迭代。

## 本轮运行与证据

- `runs/native_bfm_audit.json`：作者模型审计。
- `runs/planner_pass_train_s402/`、`runs/planner_shoot_train_s401/`：采集与元数据。
- `runs/planner_v1_status.json`：调度状态，驱动检查采集进程实际存活，不凭旧状态文件重启。
- `runs/planner_v1_train/`：训练完成后生成。
- `runs/planner_v1_results.json`：自主评估完成后生成，不以离线loss代替成功率。

本轮新增网络调用及schema测试后，24个单元测试通过。当前尚无本轮自主成绩；任务保持进行中。

附加验证：真实采集数据的一步训练 smoke 完成，模型保存/加载正常；2环境10控制步的仿真 smoke 确认统一 planner 经评估入口自主调用，`expert_assisted_oracle=false`。该 smoke 未完成任何 episode，不报告其成功率。新增跨 reset 序列标签测试通过，三个 planner 测试全部通过。

## 首次自主评估与闭环修复

v1 统一planner的射门开发评估：有效触球0/16、方向0/16；13个超时、3个wrong_foot_kick。离线验证动作MSE约0.008，自主状态平均约0.245。传球评估仍在进行。不能以离线训练结果声称技能成功。

定位到起步欠拟合：在原训练轨迹本身，射门第0帧动作MSE为0.2064，第1帧0.0114，第20帧0.00529。reset时AH历史为空，这些稀有起步状态被平均损失掩盖。追加25% reset状态分层采样，并独立训练 `planner_reset_balanced` 作为同数据修复对照。

已启动 `planner_dagger_r1_pass` / `planner_dagger_r1_shoot`：从planner v1自主访问的状态查询标签，10% episode混合老师执行，其余执行学生。`z`记录真实执行latent、`label_z`记录监督latent；测试确认二者不同且纯学生配置不执行老师动作。下一轮保留原始数据并加入这些闭环状态，训练与评估由 `run_planner_refinement.py` 驱动。

新增单一部署入口 `SoccerPolicy(checkpoint)(observation, skill, target_direction, target_speed)`，可在同一batch混合技能命令，返回关节目标与需要反馈的AH动作。入口不加载专家/技能projector、不运行oracle；评估已接入该入口。混合调用已通过API smoke，完整任务效果仍待验证。

reset分层采样对照已完成：原射门训练第0帧动作MSE从0.2064降至0.00621，但自主有效触球仍仅1/16、方向0/16（13超时、2wrong_foot、1正常踢后结束）。说明稀有起步状态是已修复的局部问题，不能解释全部闭环失败；继续使用学生状态纠偏。25项测试与作者actor兼容性审计再次通过。
