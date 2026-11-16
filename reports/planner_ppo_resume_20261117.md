# PPO续训状态与验证

`train_latent_ppo.py --resume-run <完成的PPO目录>`恢复已训练的latent适配器、critic和两个Adam优化器。`--checkpoint`继续指向原始冻结r3，KL约束仍针对原始模型，不随续训阶段改变锚点。源checkpoint、原始共享张量、环境配置与AH-BFM文件hash、模型结构及训练选项均检查；目标和超参数不匹配时拒绝续训。

模拟器和观测历史重新初始化，明确不声称恢复物理轨迹。新格式保存独立探索/打乱generator状态及它对应的policy hash；旧格式缺少generator状态时使用记录在resume_audit中的确定性fallback种子。累计训练轮数与环境步数保留，追加轮数单独记录。原始源文件不被覆盖。smoke标记必须一致，不能通过续训将短测试权重转换为正式候选。

## 已完成验证

单元测试验证恢复前后在相同数据上的下一次Adam更新逐值一致、探索随机数序列一致，并拒绝改变原始模型、奖励目标或smoke状态的配置。全套83项测试通过。

真实模拟器分别验证旧格式恢复和新格式generator恢复：

| 阶段 | 累计轮数 | 累计环境步 | Adam步数 | 探索generator恢复 |
|---|---:|---:|---:|---|
| 原短测试 | 2 | 256 | 8 | 源文件未保存 |
| 旧格式续训 | 4 | 512 | 16 | 明确记录fallback |
| 新格式再次续训 | 6 | 768 | 24 | 逐值恢复 |

两个续训检查各实际处理32次超时；恢复时planner、critic和优化器状态逐值核验，训练后共享张量及射门探针保持、重新加载SoccerPolicy检查通过。它们仍是smoke运行，不作为任务成功率或正式候选。

运行目录：`planner_pass_ppo_resume_smoke_legacy`、`planner_pass_ppo_resume_smoke_rng`；调度完成状态为`planner_pass_ppo_resume_smokes_status.json`，优化器步数另存于`planner_pass_ppo_resume_optimizer_steps.json`。

正式续训从`planner_pass_ppo_no_reference`追加200轮已完成，累计300轮、2457600环境步；共享/射门保护及部署加载检查通过。两技能评测已完成：射门16/16触球、11/16方向达标；传球14/16触球、2/16方向达标，未优于默认模型。完整效果以`planner_latent_ppo_20261116.md`记录为准。
