# 当前AH-BFM的基础站立诊断（已完成）

连续候选仍有射门后高度过低和姿态失败，先检查当前骨干是否能在相同源射门物理中保持基本站姿。此诊断不改变正在运行的新种子确认代码，不计入足球成功，也不证明能处理射门后的动量。

现有 `assets/z_bank.json` 的meta声明来源为MuJoCo校准的 `BFM-0.5-75275`、actor ONNX SHA256 `93b4e4becefb3a9f6096c1316ca287950a114c8e119cf74f311b90f04a20f9da`。当前作者AH-BFM为另一份原生checkpoint；没有证据证明旧stand latent兼容，诊断不使用该向量。来源记录见 `planner_standing_bank_provenance.json`。

两组使用源射门reset、物理、动作和随机化，保留全部失败终止，移除触球后的计划结束、延长超时至完整20秒，禁用球重采样。两组16环境，固定种子816401，并核对完整初态和场景一致。

- `direct_default`：直接发送Bridge.default关节目标，是绕过BFM的物理对照，不是可部署统一策略。
- `latent_default`：每步对同一固定关节目标做64步latent求解，再由冻结作者AH-BFM输出动作。仅优化latent，不训练骨干、查询专家或直接执行目标关节角。初始latent来自当前checkpoint的backward map，之后以前一latent热启动；映射仍标注为未获作者golden vector验证的候选映射。

先运行2环境4步的latent smoke，成功后才继续两项20秒对照。已完成脚本语法检查，真实Isaac Sim中的2环境4步smoke也已通过，2/2未失败，CUDA graph latent求解正常执行（见 `planner_native_standing_smoke.json`）。驱动 `run_native_standing_diagnostic.py` 已启动，通过现有Isaac Sim锁串行执行。记录首次失败、根部高度/倾斜、位移、目标拟合误差，并将失败环境永久排除出存活计数。两项完整闭环现已结束，结果见下方；没有将此控制器集成到足球策略。

## 完整结果：默认关节目标不能维持站立

初态和完整场景配对检查通过。两组初始高度均为0.76米，倾斜为0。

| 控制方式 | 20秒存活 | 姿态超限 | 高度过低 | 首次失败时间中位数 |
|---|---:|---:|---:|---:|
| 直接默认关节目标 | 0/16 | 15 | 1 | 1.28秒 |
| 64步latent目标求解 | 0/16 | 16 | 0 | 1.29秒 |

失败时间按零基tick对应控制步结束计算，即(tick+1)×0.02秒；两组范围分别为1.14–1.38秒、1.16–1.40秒。全部失败均为物理姿态/高度终止，没有计划结束或超时。目标关节误差仅统计首次失败之前的采样，直接组为0，latent组中位MSE约1.08e-5 rad²、均值约0.00221 rad²（失稳状态误差变大）；不混入重置后轨迹。原始状态与分析见 `planner_native_standing_status.json`、`planner_native_standing_analysis.json`。

该结果排除了直接采用这套默认关节目标作为恢复方案的依据，不能据此认定AH-BFM没有站立能力。后续诊断改用源任务真实初始姿态，并区分关节目标保持和固定backward目标编码的反馈控制。
