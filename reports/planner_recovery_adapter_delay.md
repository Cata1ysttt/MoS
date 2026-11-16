# 对齐恢复适配器的启用时序

恢复训练在观察到射门触球并等待10个控制步后，才切到传球分支并更新适配器。原通用调用只检查传球skill与ball_kicked，普通传球一触球就会启用它。新控制选项将适配器的最早启用时间对齐到触球后10步，同时保留真实“已触球”观测及冻结模型的动作。

`SoccerPolicy(..., post_kick_adapter_delay=10)`通过单独的布尔mask控制适配器修正，不篡改输入状态。每个环境独立计时，持续观察到contact的前10次调用仍使用冻结模型，第11次调用可使用修正。球事件清除contact或`reset(done_mask)`均重启计时。必须保持固定环境槽位，每个控制步调用一次。

原参数默认仍为0，以保留已运行对照的语义。原任务与连续评测入口新增`--post-kick-adapter-delay`并在结果中记录。原任务评测也在环境重置时清除统一policy的运行状态。冻结模型或未启用延迟时行为保持；旧的非阶段适配器不能误用此选项。

验证：99项测试通过（4.81秒）。在8个真实留出触球后状态上，完整SoccerPolicy的latent、关节目标、动作前10次调用与r3逐值相同，第11次与未延迟的恢复模型逐值相同；reset会重新开始等待。输入中的ball_kicked始终保持真实的true，checkpoint哈希未变。来源：`runs/planner_shoot_recovery_delay_native_audit.json`。

`scripts/evaluate_recovery_delay.py`使用同一份完成的恢复checkpoint，先评测延迟设置下的原传球/射门任务，再重新运行连续场景的0步/10步两组，检查初态、物理与任务配置、首次接触前记录一致，并保留完整方向/规定脚/速度/持续站立评分。权重与环境目标不变，种子仍为重复开发种子。连续记录新增真实contact上升沿，避免以晚于接触检测的球速记录推断干预开始时间。

闭环评测已启动，状态为`runs/planner_shoot_recovery_ppo_pilot_delay10_status.json`。完整结果尚未产生；不能由单元测试或固定状态比较宣称精度恢复或持续成功增加。
