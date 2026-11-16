# 仅在射门触球后使用恢复适配器

恢复PPO在射门触球后训练，但原部署门控只检查传球skill与ball_kicked，也会在普通传球触球后介入。此前把启用时间推迟10步，传球方向达标仍为3/16（r3为6/16），完整连续成功仍为1/16。因此本次直接区分接触事件发生时的skill，检验恢复模块的适用范围是否造成传球退化。

`SoccerPolicy(..., post_kick_adapter_delay=10, recovery_adapter_shoot_only=True)`记录每个环境首次观察到触球时的skill。只有该skill为shoot，后续切换到pass时才允许恢复适配器修正。切换本身不重置机器人、历史或接触来源；ball_kicked清除或policy.reset(done_mask)会清除来源。每个控制步调用一次，保持固定环境槽位。中途新建policy不能从传球状态推断此前的射门事件。此选项默认为false，原checkpoint行为不变；不适用于旧的非触球后适配器。

没有修改planner输入、任务目标、奖励、评分或权重。门控只作用于恢复适配器，射门与普通传球仍由同一个共享planner/AH-BFM执行。

验证：102项测试通过（5.29秒），覆盖混合环境、skill切换、换球、局部/全部重置、endpoint和latent两种适配器。完整SoccerPolicy在8个真实留出传球触球后状态上，连续12次调用的latent、动作和关节目标与r3逐值相同；人为设置一半槽位的射门命令历史后切回pass，仅这些槽位与适配模型输出相同，其他槽位仍与r3相同。重置清除来源，输入和权重不变。这是固定状态的部署路由检查，不是射门后物理恢复成功的证据。脚本为`scripts/audit_recovery_origin.py`，产物为`runs/planner_shoot_recovery_origin_native_audit.json`。

闭环驱动`scripts/evaluate_recovery_origin.py`已启动：

- 先在原开发种子811314运行传球、射门各16个episode，并要求episode记录和首触球速度记录均与r3一致。
- 随后按预先选定的新种子815101、815102，分别运行32环境的r3/候选连续对照，共64个环境/模型。两组均使用源射门物理、20秒、无latent插值、射门后10步切回pass；候选附加接触来源门控。
- 核验同种子初始张量、场景与协议一致，以及首次恢复干预前的观测事件一致。完整成功仍要求两次触球的方向不超过10°、规定脚、前向球速误差不超过30%，且未失败/重置持续到20秒。

状态：`runs/planner_shoot_recovery_ppo_pilot_shoot_origin_status.json`，启动PID3072006。权重SHA256为`4ffe85c79ce25121fc26d4bbba570978b92a607a237895a34229e2b492ff9664`。闭环结果尚未完成，不据单元测试宣称成功率改善，默认模型仍为r3。
