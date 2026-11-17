# 射门切换时的关节目标变化：历史诊断与待采集验证

此前近球组的准确射门比例较低，值得检查进入射门时的控制衔接。这里先在既有r3训练采集的235个handoff、后续真实状态上比较下一条AH关节目标与上一帧记录的AH目标。`Bridge.observe`的64:93已经是scaled action，前一目标使用它直接乘scale，不再乘6。比较的是控制命令，不是机器人在20ms内实际移动的关节角度。

使用当前完整球历史候选的同一共享prior/AH-BFM，保留记录的本体和AH历史，分别重建已执行latent的输出、原始球历史prior、完整球历史prior，以及以当前backward编码为目标的输出。没有执行新动作或查询专家。

| 状态组 | 状态数 | 记录latent重建的关节RMS变化中位数 | 完整球历史反事实prior | 当前backward目标反事实 |
|---|---:|---:|---:|---:|
| 首次handoff | 235 | 0.13358 rad | 0.45558 rad | 0.11372 rad |
| 触球前第1–9步 | 2115 | 0.06072 rad | 0.27679 rad | 0.14912 rad |
| 触球前第10–49步 | 9308 | 0.05893 rad | 0.06563 rad | 0.19170 rad |
| 触球前第50步以后 | 14416 | 0.10070 rad | 0.10070 rad | 0.22314 rad |

原始球历史prior的中位数与记录latent重建一致。完整历史反事实在handoff的RMS变化P95为0.6674rad，最大单关节目标差约1.9894rad。当前backward目标在handoff有较小典型变化，但后续多数阶段更大；不能把它当作持续射门控制器，也不能把它与reset时保存的站立目标混淆。

这些是旧r3实际走过的状态，反事实动作不会改变其后续轨迹。因此不能声称当前完整球历史控制器实际有相同的首步变化，或认定这是失败原因。它提供了需要在线核实的衔接问题；不会仅凭该统计回退球历史修正或加入平滑。

已排队采集当前完整球历史基线的32回合、20秒连续执行，诊断种子817201、同一物理/指令/球采样/评分，不加载专家。被动保存真实shoot输入和实际发出的latent，结束后自动重建命令变化并记录原任务结果。数据目的为diagnostic，不用于声称泛化改善。采集等待有效传球触发版本的固定对照结束，以免修改或打断当前实验。

队列 `queue_shoot_handoff_collection.py`，状态 `runs/planner_full_history_handoff_collection_r1_status.json`；新集合尚未完成。队列每次检查实际父PID和终态，运行前后校验冻结源码与模型。协议 `planner_full_history_handoff_collection_protocol.json`，历史结果 `planner_shoot_action_jump.json`，诊断脚本 `audit_shoot_action_jump.py`。
