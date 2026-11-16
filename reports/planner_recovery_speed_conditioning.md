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
