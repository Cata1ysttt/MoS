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

## 观测补全与下一轮

v1完整自主结果：传球触球3/16、方向0/16；16个终止包括14个missed_ball、1个wrong_foot、1个anchor_ori。DAgger r1射门仍为有效触球0/16、方向0/16（5个wrong_foot、7个anchor_pos、4个anchor_ori），未达到目标。

代码核对确认传球专家具有launch/kicked阶段及机器人场地位置/朝向观测，945维初版没有这些变量。新增可版本化 `soccer_state_v2_950`，补充两维事件阶段、相对env原点的xy与yaw；旧checkpoint继续走旧schema，不伪造兼容。新采集记录身体位置、姿态、link线速度、角速度与body名称，为backward输入映射验证提供原始数据。

已启动新种子811451/811452的v2采集，混合50% teacher episode和50% r1学生episode，明确区分执行latent与标签。新schema数据重新训练，绝不将旧945维数据冒充950维。下一轮动作辅助损失改为拟合latent teacher真正解码的动作，避免一面模仿s1的latent、一面被原专家动作监督推向不同控制；这是额外实验变量，不解释为单独观测补全的因果结果。

`run_planner_v2.py` 驱动采集后6000更新和同checkpoint两技能自主评估。当前26个测试通过；v2尚无自主成绩。第一版代码已推送main提交 `8c06ab1`；后续schema修复仍在持续迭代。


## 最新快照：语义映射、flow 与动作监督

以下更新覆盖上文各阶段“进行中”的历史状态。v2 已完成：自主传球触球2/16、方向0/16；射门3/16、0/16。加入 backward prior 的回归 v3：传球0/16、0/16，射门1/16、0/16。所有结果使用开发种子811314，没有作为训练种子，不是新种子泛化验证。

已加载配套 backward encoder，候选身体映射为 `candidate_g1_31_xz_head035`：31个身体点、X/Z旋转表示、躯干上方0.35m虚拟头部。通过平移/全局yaw不变性及数值审计，但没有作者输入构造的黄金参考，不能声称完全一致。语义标签优化保留原执行latent，独立存储监督label；均方语义项权重0.01是本项目实验设置。

新增条件 endpoint flow MLP：Gaussian插值训练、预测clean endpoint、部署固定噪声8步Euler采样，最终latent归一化为范数16。它使用仿真真值和5帧短序列，不是完整FIBER DiT/感知实现。初版flow自主传球0/16、0/16，射门2/16、0/16。扩大采集到每技能64环境、1000控制步，加入显式命令编码后，flow wide自主传球与射门均为触球0/16、方向0/16。

独立专家辅助对照 `runs/semantic_oracle_shoot_diagnostic/summary.json`：64步动作预对齐再64步语义优化，射门触球15/16、方向12/16。它每步使用专家，不是planner成绩；只表明该actor与候选语义约束存在可执行的控制解。

动作拟合审计 `runs/flow_wide_action_fit.json` 显示，flow wide在已采集射门状态上仍有明显解码动作误差，例如留出环境第1帧腿部关节目标MSE约0.05345。新增 `train_flow_planner.py --action-weight 100 --leg-weight 2`，直接反传完整部署8步采样和冻结AH actor，监督语义标签实际解码的物理关节目标。这是足球任务额外损失。`run_planner_flow_action.py` 已启动10000更新及同checkpoint两技能自主评测；未完成前不计入成功结果。

本次上传包含实现、实验驱动、诊断和结果报告；作者提供的 `AH-BFM/` 包、权重、轨迹和运行日志均由Git忽略。33项测试通过。同环境不reset连续切换尚未验证，统一planner目标仍未完成。最新完整自主表由 `scripts/summarize_planner.py` 生成。

动作辅助版本随后完成10000更新；同checkpoint自主射门评测已完成：有效触球1/16、方向0/16，仍不达标。传球评测正在运行。不能将动作拟合损失下降解释为闭环任务成功。
