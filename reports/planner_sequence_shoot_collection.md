# 连续传球后射门的数据采集

来源门控对照中，两个新种子的r3与候选都在35/64个环境里于射门触球前失败。此前射门训练主要来自单任务静止球起步，未直接覆盖统一策略传球之后的身体状态与连续观测历史。本轮收集真实handoff后的学生访问状态，后续用于射门分支适配。

`probe_soccer_sequence.py --record-shoot-states`被动保存每个控制步仍存活且skill为shoot的v4输入、执行latent、环境编号、tick和源动作的scale/offset。不保存失败后重置的新episode，不加载专家，不改动作、机器人、球或历史。失败轨迹上的失败前状态也保留，不只收集成功样本。

`label_sequence_shooting_states.py`离线读取保存的1030维shoot policy历史及150维RMA历史，查询原ONNX射门专家。标签使用源环境的完整动作映射：TrackingKickEnv.step先将raw action裁剪到±10，再由JointPositionAction应用scale/offset，物理目标不再裁剪，最后按AH关节顺序排列。最初版本漏掉了第一步，修正记录见下文。标签记录专家与源状态文件SHA256。此过程不执行任何专家动作，不产生新的任务成功率。

测试：104项通过（5.29秒）。新增测试检查存活/技能筛选、原输入不变、CPU快照独立、关节映射、单调tick以及随机数流不变。真实模拟器的被动性验证已通过，见下文；单元测试本身不作为闭环效果证据。

驱动`scripts/collect_sequence_shooting.py`先重放种子815101、32环境、1000步的r3连续基线，要求summary、protocol和scene contracts与关闭记录时全部一致。该组标记为diagnostic。验证成功后使用训练种子815201、256环境、1000步自主采集，并离线标注；训练与诊断数据分目录且带purpose元数据。

本轮源policy仍为r3，SHA256为`5cdc03e0d5d5c44a20626d03c4a4088fab0d6f2ab5b229b66466c18bdade2133`。权重不更新，尚无新的射门适配器训练或效果结论。

状态：`runs/planner_sequence_shoot_r1_status.json`，启动PID3118983。采集结果和作者权重仅保留本地。


## 被动记录闭环验证完成

种子815101的重放已经完成。开启记录后，完整summary（含各次触球、切换和失败记录）、protocol与scene contracts均与此前关闭记录的r3基线完全一致。保存了30个handoff环境的3324条射门状态，随后离线标注成功；原专家最大绝对raw action为7.55795，没有超过10的值。

该组purpose=diagnostic，不用于训练。独立种子815201、256环境的正式采集已完成，结果见下文。


## 正式采集完成及标签修正

256环境中235个进入射门阶段，保存27022条状态，其中26074条触球前、948条触球后。没有筛掉失败轨迹；未进入射门阶段的环境没有对应射门状态。该轨迹是训练数据来源，不作为新的独立验证。

检查原始动作范围发现8个raw action分量超过10。核查当前`TrackingKickEnv.step`源码确认其在action manager之前裁剪到±10，因此最初离线标注的无原始动作裁剪映射有误。已保留旧文件，并生成`shoot_labels_clamped.pt`：8个状态各有1个目标分量修正，最大修正0.35875rad；所有输入与专家raw输出保持，修正目标与原生Bridge映射逐值一致。复现脚本`scripts/audit_sequence_label_clamp.py`，源环境文件SHA256和新旧标签SHA256见`reports/planner_sequence_shoot_label_clamp_audit.json`。

旧`shoot_labels.pt`的SHA256为`816dd2706a0ea11d8186437b9dfd4fc606eb24227e7510bb4acebe76667ee851`，不用于训练。修正版SHA256为`907125fa46b26e6e12a9ee7a675522efe120f4bd079ff7e8059a293032ee6e9d`。训练入口拒绝未修正映射、diagnostic数据和已保留的评测种子，并核对源数据、收集policy与标注专家身份。

当前测试共107项通过；真实数据两轮训练短跑与完整SoccerPolicy导出重载通过。正式射门适配训练已启动，详见`reports/planner_sequence_shoot_adapter.md`；尚无新模型的闭环改进结论。
