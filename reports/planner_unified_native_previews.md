# 同一checkpoint的原生传球与射门预览

以下视频来自完整球历史研究基线 `runs/planner_new_ball_task_history_r1/diagnostic.pt`（SHA256 `0d98300f2b299da3af5d229f8d8534a6670e9ef692c76895cde98ae232136514`）已有的原生任务评测。两个任务使用同一checkpoint、统一latent接口与冻结AH-BFM；它不是正在训练的1000轮候选，也没有替换默认r3。

- [传球环境0：40秒完整原生回合](unified_policy_previews/pass_env0.mp4)
- [射门环境0：约1.4秒原生回合](unified_policy_previews/shoot_env0.mp4)

固定展示环境0，没有根据成功与否挑选最好回合。传球保留整个40秒回合，其中包含原任务的多次球事件；没有拼接机器人reset后的片段。射门在原任务的`event_kick_elapsed`条件触发后正常结束，不能把短视频结束理解为摔倒，也不能据此证明射门后20秒稳定。

这些是Isaac记录状态的重绘，MuJoCo只用于模型前向计算与画面渲染，不做动力学推进。不是sim2sim实验。渲染文件核验为H.264、320×240、10fps，分别400帧和14帧；已抽查传球3.5秒与射门0.8秒画面，机器人、球和时间标记可见。视频编码时长受采样帧数取整影响。

这组已有原生评测每技能16环境：传球16/16触球、7/16方向≤10°；射门16/16触球、11/16方向≤10°。单段视频不代表总体准确率，也不是连续“传球→射门”的演示。连续任务的128回合确认仍只有9/128完整成功，见 `planner_new_ball_task_confirmation.md`。

使用现有本地Python环境和 `render_trace.py --run ... --env 0 --output ...` 重绘；新增output参数允许将视频保存到记录目录以外，不改原始轨迹或权限。运行中的训练源码保持冻结。视频与抽查帧保留本地，不随Git发布；源记录、渲染脚本与视频哈希及编码信息见 `planner_unified_native_previews.json`。
