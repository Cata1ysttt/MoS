# 连续射门PPO：100轮与300轮训练时长对照

前轮相对于r3增加双触球（23/64→31/64），但准确射门仍8次、完整成功仍2/64。新增触球没有转化成精度收益。本轮先检验训练时长，不同时改变奖励、物理或评分。

## 固定条件与恢复范围

从`runs/planner_continuous_shoot_ppo_r1/final.pt`继续200轮，源SHA256为`030bfc64a7153afa44c001f2cbd261e625c42d663f386021b89f3e7d9621b6fb`。原始冻结基座和KL锚仍是r3；共享planner和AH-BFM不训练。保留256环境、32步rollout、4epoch×4minibatch、学习率4e-5/3e-4、std=.2、KL锚系数.01，以及前轮连续任务奖励、阶段时序与失败条件。新增1638400环境步，累计2457600步、300轮；仍取最后一轮，不根据评测挑中间权重。

恢复planner、critic、两个Adam优化器和专用探索/小批次随机数生成器。仿真从新回合开始，机器人状态、历史与仿真随机数不恢复；因此这是学习状态的接续，不声称与一次不间断300轮运行逐步一致。训练种子保持815401。

恢复入口拒绝更改原始基座、奖励版本、超参数、烟雾测试标记或未完成保持性检查的来源。奖励/阶段/flow/policy实现SHA、作者actor SHA、原任务配置SHA和完整场景配置均与来源核对。源权重与学习状态文件在训练前后校验SHA。

## 恢复验证已通过

122项测试通过（5.36秒），包括原有的下一次Adam更新及探索样本一致性测试，以及新的连续任务配置、训练累计计数与拒绝条件检查。

32环境短跑从4轮烟雾检查继续4轮，累计8轮/32768环境步。planner、critic、两个优化器与探索随机数均准确恢复；独立审计确认每个Adam参数的step都增加了实际新增更新次数（actor24、critic32）。本段26次handoff保持机器人和历史状态，1818个shoot学习步，26次时限结束，导出一致且共享参数不变。奖励账目、源物理/失败条件和128个真实留出状态的路由检查通过。上述数据仅验证实现，不作为性能结果。见`planner_continuous_shoot_ppo_resume_smoke.json`。

## 正式实验已启动

驱动PID3365289，状态`runs/planner_continuous_shoot_ppo_r2_status.json`；训练目录`runs/planner_continuous_shoot_ppo_r2`。启动命令为：

```bash
python scripts/run_continuous_shoot_ppo.py --tag planner_continuous_shoot_ppo_r2 --resume-run /home/jiongye/fiber-soccer/runs/planner_continuous_shoot_ppo_r1 --iterations 200 --seeds 815601 815602
```

训练结束先完成导出/奖励/恢复步数审计与原传球、射门任务检查，再用预定新种子815601、815602，各32环境比较源100轮候选与累计300轮候选。两组都从同一传球前缀开始，失败后统一零动作；方向≤10°、规定脚、前向速度误差≤30%和20秒无失败标准保持。

本轮连续表格中的baseline是100轮候选，不是r3。驱动与汇总显式记录该比较基线的路径和SHA，避免与前轮r3对照混淆。默认部署仍为r3；尚无本轮训练完成或闭环结果。

并行完成了上一轮首触球奖励的离线核查：准确射门平均触球奖励13.43，规定脚但不准确为2.61；候选有10次仅因方向落在10°–20°而未达标。该分析不改训练，详见 `planner_continuous_shoot_reward_signal.md`。
