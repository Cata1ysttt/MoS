# 连续射门任务回报学习的latent修正接口

完整球历史修正增加触球和存活，但完整成功未提升；继续逼近内部关节目标的射门求解又使两组开发完整成功11/64降至5/64。新方向保留完整球历史版本的初始传球、共享先验及初始backward恢复目标，只在观察到传球触球后、当前为射门命令且球未触球时，学习一个latent修正。目标是后续用实际连续任务回报训练，当前接口尚未经过任务训练。

`PostPassLatentAdapterPolicy` 继承已验证的恢复控制器，由 `load_soccer_policy` 根据 `post_pass_latent_adapter` 元数据加载。修正层接收完整球历史处理后的v4观测与基础latent，输出256维修正后保持基础latent半径；最后一层零初始化，输出逐值保持。推理仍由同一冻结AH-BFM输出动作，不加载或查询专家。

观测历史只由父控制器推进一次；适配层从该步保存的年龄和首测量重构同一份输入，不再次更新状态。提供本步 `last_adapter_input`、`last_base_latent` 与 `last_adapter_active`，供后续PPO缓存。小批量优化必须直接使用缓存和修正层，不能重放有状态policy来计算训练分布。reset清除阶段门控和临时缓存。

172项测试通过，包括零初始化逐值保持、构造不改变RNG、可训练性、latent半径保持及相消修正回退。16个真实历史输入配合显式阶段标记进一步验证：

- 原生传球和射门逐值保持；传球后51个射门控制步的零适配输出逐值保持。
- 重构输入与独立历史处理器每步一致，历史不会重复推进。
- 射门触球后及恢复输出保持。
- 刻意加入非零修正后，混合批次仅符合条件的行变化，部分reset后的原生射门行保持。
- 原参数不变、加载CPU/CUDA RNG保持，checkpoint往返一致。

零初始化checkpoint `runs/planner_post_pass_latent_adapter_zero_r1/diagnostic.pt`，SHA256 `81dc1618e1871913e32decde4611993bda56423a3c723bee82ace4ffa393233b`。适配层373888个参数；部署加载默认冻结，训练阶段需只启用这些参数。详见 `planner_post_pass_latent_adapter_smoke.json`。

已启动零初始化闭环检查：原生传球与射门各16环境，随后种子816201、32环境、20秒连续回合。要求原生逐回合/首触球记录保持，连续结果除checkpoint路径/哈希、延迟和显式adapter元数据外全部与既有完整球历史对照一致；同时审计实际激活区间。协议见 `planner_post_pass_latent_adapter_zero_protocol.json`，状态为 `runs/planner_post_pass_latent_adapter_zero_r1_screen_status.json`。

当前没有训练后的checkpoint或性能提升结论。下一步需要接入有状态基线的PPO数据缓存、真实动作历史推进、reset和任务奖励，再做训练冒烟与独立评测。默认仍为r3，未替换。
