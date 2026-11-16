# 恢复阶段的速度指令敏感性

传球命令配置的目标速度范围为2–3 m/s。现有射门后切回传球分支的恢复对照保留6 m/s目标速度。本审计检查在已记录传球触球后状态中，只改变统一planner末尾速度输入会造成多大输出差异。

选取r2/r3传球数据最后四分之一环境，ball_kicked为真，每份均匀取最多512行，共1024个状态。记录速度范围2.0089–2.9826 m/s。状态SHA256：`e3bdd911ad3b35e22f5b0463f65fadcf335eebfa0996f08e671c5d8d93dc2e95`。

| 模型 | 改为2.5时关节目标RMS差（rad） | 改为6时关节目标RMS差（rad） | 改为6时腿部RMS差（rad） |
|---|---:|---:|---:|
| planner_history_r3_contact_train | 0.0005374 | 0.0178572 | 0.0191588 |
| planner_history_r5_pass_adapter_control_train | 0.0005493 | 0.0184318 | 0.0201855 |
| planner_pass_ppo_shared_scene | 0.0005429 | 0.0177486 | 0.0190205 |

输入干预在三个模型中均产生明显不同的输出幅度，说明“切回传球分支后保留射门速度”不能假设为无影响。它仍不足以证明失败原因，也不证明将恢复速度改为2.5会改善闭环。

限制：这些是原传球记录中的触球后状态，不是实际射门后的状态；历史观测保持原值，因而只度量即时速度输入变化。关节目标为AH-BFM解码后、场景裁剪/延迟之前的命令；未进行环境回放、参数更新或改变任务评分。接下来的有效验证应是明确标记的闭环恢复指令对照，并保持已完成射门的真实6 m/s评分目标。

来源：`scripts/audit_recovery_speed_conditioning.py`已对三份完成的checkpoint运行；原始结果为`runs/planner_recovery_speed_conditioning.json`和`runs/planner_shared_recovery_speed_conditioning.json`。

## 已安排的闭环对照

`probe_soccer_sequence.py --post-shoot-recovery-steps 10 --post-shoot-recovery-speed 2.5`只在观察到射门触球并等待10步、切回传球分支后覆盖planner速度输入。环境command、已记录射门目标速度及评分不变；不重置机器人、历史或球。未提供速度选项时保持原行为。

对照冻结r5 control checkpoint，源射门物理、16环境、1000步，场景与独立第二球种子813201。两组均启用10步后切回传球分支，唯一配置差异是planner恢复速度是否设为2.5。重新运行两组以检查初态哈希、场景配置及首次干预前的事件记录一致，不把旧记录缺失的初态哈希补作已验证。

`audit_recovery_speed_ablation.py`检查上述对照条件；恢复日志分别记录环境目标与planner目标，`audit_sequence_recovery.py`拒绝将射门评分目标改为2.5，也拒绝缺失或错误的覆盖记录。原有完整方向/规定脚/速度/持续站立指标保持。两组排在当前共用场景评测及原专家延时诊断之后，结果尚未产生。

验证：完整86项测试通过；随后新增的3项计数/评分保护回归测试通过。真实旧连续记录的计数重算通过，构造的评分目标改变和错误planner指令均被审计拒绝。速度覆盖的实际闭环效果仍待排队运行，不能由这些检查推断有效。

状态：`runs/planner_recovery_speed_ablation_status.json`；预定输出`planner_r5_recovery_speed_baseline`、`planner_r5_recovery_speed_override`。排队的原专家诊断未启用本选项，执行代码版本记录补充在`runs/source_shoot_continuation_protocol_v2.json`，原协议文件保留。
