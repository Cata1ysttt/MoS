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

当前没有训练后的checkpoint或性能提升结论。已接入有状态基线的PPO缓存、真实动作历史推进、reset及任务奖励，短训练已排队，仍需实际运行验证与独立评测。默认仍为r3，未替换。

## 连续任务训练接入（待运行验证）

新增 `train_post_pass_latent_ppo.py`：每个真实观测只调用一次有状态policy，缓存处理后输入、基础latent和适配激活标记。PPO小批量更新仅调用修正层，不重放历史、求解器或恢复状态；每轮检查有状态调用计数保持，训练结束要求总调用数恰好等于控制步数加一次初始观测。实际采样动作经AH-BFM执行并推进Bridge历史，done后同时重置阶段、Bridge与policy。

只有传球后、射门触球前的状态参与actor更新与高斯探索；其余阶段latent逐值等于基础控制器。共享planner、目标预测器、AH-BFM和归一化缓冲保持冻结。使用既有连续任务六项奖励，加既有联合奖励系数（两次准确触球额外10，联合成功且存活至超时额外20）；critic使用阶段、剩余时间和十维事件记忆，部署输入不增加这些特权事件信息。有限时域不做超时bootstrap。

本轮短训练固定种子817100、32环境、16×32控制步、8秒训练时域，共16384环境步。要求真实发生射门阶段、actor更新、超时reset，冻结参数保持，导出后原生技能路由保持且连续射门输出与训练修正层一致。它不是20秒性能评测，也不能推广。训练种子817101预留给后续正式训练，当前不会自动启动正式训练。

`queue_post_pass_latent_ppo_smoke.py` 先检查零初始化闭环任务对应进程确实存活；只有该任务成功结束且连续结果完全保持，才启动短训练。冻结相关源码和checkpoint哈希，失败保留输出。当前零初始化的两项原生回归已通过、适配分支零激活；连续回归尚在执行。训练排队状态为 `runs/planner_post_pass_latent_ppo_smoke_r1_queue_status.json`。训练脚本目前仅通过语法检查，运行正确性仍待短训练验证。
