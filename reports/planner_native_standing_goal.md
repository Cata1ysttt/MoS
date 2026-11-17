# 源任务初始站姿与backward目标诊断（已完成）

前一轮在默认BFM关节目标下，直接控制和latent求解均0/16存活。新诊断固定实际reset得到的站姿，检验失败是否与目标姿态选择或控制方式有关。仍是基础站立诊断，不是足球或射门后恢复成绩。

固定源射门场景、16环境、种子816401（复用开发对照）、20秒；全部失败条件保留。三组使用相同初态和物理：

- `direct_reset_pose`：直接保持各环境reset时实测的关节姿态，仅作物理对照。
- `latent_reset_pose`：每步64次求解latent，目标为同一实测关节姿态，再由当前作者AH-BFM解码。
- `backward_stationary`：保存reset时的身体姿态，将backward输入中的关节、根部角速度及全部身体线/角速度置零，一次编码得到固定目标latent；后续不求解动作目标，由同一AH-BFM根据实时身体状态输出反馈动作。关节误差只作为诊断量，不是该组的优化目标。

所有latent来自当前作者checkpoint，未使用旧z_bank。使用的 `candidate_g1_31_xz_head035` backward观测映射仍未获作者golden vector验证，因此结果只适用于本实现。保存 `goal.pt` 供输入及latent审计，不将权重或运行数据上传Git。

`run_native_standing_goal_diagnostic.py`要求前一轮已完成，先做2环境4步的backward模式smoke，再执行三组对照；冻结运行代码，核对初态、场景和完整失败计数。仿真继续复用现有串行锁，未修改正在运行的足球确认评测代码。backward模式真实仿真smoke已通过，2/2环境执行4步无失败；记录见 `planner_native_standing_goal_smoke.json`。三项20秒对照现已全部完成。

## 反馈控制成功保持基础站立

初态和完整场景检查通过，16环境均计入分母；没有移除任何失败条件。

| 控制方式 | 20秒存活 |
|---|---:|
| 直接保持实际reset关节姿态 | 0/16 |
| 64步latent逼近实际reset关节姿态 | 0/16 |
| 固定backward目标、AH-BFM实时反馈 | 16/16 |

成功组在最后采样时刻（19.8秒）的水平位移均值0.144米、最大0.255米；不是锁住物理状态或重置后重新计数。完整结果与仅统计首次失败前轨迹的分析见 `planner_native_standing_goal_status.json`、`planner_native_standing_goal_analysis.json`。

额外比较保存的direct和backward两组 `goal.pt`：初始state、privileged_state与latent逐值相同，关节速度、根部角速度、身体线/角速度本来都为零。因此本场景中，成功组的目标可以直接从首次观测的backward特征保存，无需额外姿态输入或旧版z_bank。

这是一次复用开发种子的基础站立结果，尚不证明能从真实射门后的动量中恢复。下一步仅在同一策略观察到射门触球并收到恢复指令后调用该目标，保持传球和射门原输出；见 `planner_initial_goal_recovery.md`。
