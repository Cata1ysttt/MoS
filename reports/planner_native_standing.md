# 当前AH-BFM的基础站立诊断（运行中）

连续候选仍有射门后高度过低和姿态失败，先检查当前骨干是否能在相同源射门物理中保持基本站姿。此诊断不改变正在运行的新种子确认代码，不计入足球成功，也不证明能处理射门后的动量。

现有 `assets/z_bank.json` 的meta声明来源为MuJoCo校准的 `BFM-0.5-75275`、actor ONNX SHA256 `93b4e4becefb3a9f6096c1316ca287950a114c8e119cf74f311b90f04a20f9da`。当前作者AH-BFM为另一份原生checkpoint；没有证据证明旧stand latent兼容，诊断不使用该向量。来源记录见 `planner_standing_bank_provenance.json`。

两组使用源射门reset、物理、动作和随机化，保留全部失败终止，移除触球后的计划结束、延长超时至完整20秒，禁用球重采样。两组16环境，固定种子816401，并核对完整初态和场景一致。

- `direct_default`：直接发送Bridge.default关节目标，是绕过BFM的物理对照，不是可部署统一策略。
- `latent_default`：每步对同一固定关节目标做64步latent求解，再由冻结作者AH-BFM输出动作。仅优化latent，不训练骨干、查询专家或直接执行目标关节角。初始latent来自当前checkpoint的backward map，之后以前一latent热启动；映射仍标注为未获作者golden vector验证的候选映射。

先运行2环境4步的latent smoke，成功后才继续两项20秒对照。已完成脚本语法检查，真实Isaac Sim中的2环境4步smoke也已通过，2/2未失败，CUDA graph latent求解正常执行（见 `planner_native_standing_smoke.json`）。驱动 `run_native_standing_diagnostic.py` 已启动，通过现有Isaac Sim锁串行执行。记录首次失败、根部高度/倾斜、位移、目标拟合误差，并将失败环境永久排除出存活计数。当前尚无闭环结果，任何收益都需另行集成并验证射门后恢复，不能直接提升默认模型。
