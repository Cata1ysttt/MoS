# 相同新球输入下的射门专家诊断（运行中）

完整球历史初始化在独立确认中提高触球与存活，却没有提高完整成功。既有原专家接管直接读取环境的原历史，并未使用完整球历史初始化，不能凭旧接管结果排除修正输入后的专家能力。

本诊断固定两个已使用开发种子816201、816202，三组各64回合、20秒：

- `unified`：完整球历史候选，经同一AH-BFM执行所有动作。
- `expert_raw`：相同初始传球与恢复策略，射门阶段由原ONNX专家读取原历史后输出原始环境动作。
- `expert_full`：与expert_raw相同，但原专家接收完整新球历史初始化后的输入。

两种专家接管组使用原恢复checkpoint，传球与恢复保持相同控制逻辑；统一版本的额外历史模式只作用于射门阶段。三组保留相同物理、初态、球采样、handoff和评分，全部失败计入分母。统一组必须逐项复现上一轮开发结果。专家组明确记录 `expert_loaded=true`、`one_policy_instance=false`、`shoot_controller=original_expert`，不会计作自主统一policy成绩，也不会自动推广为部署策略。

新增 `--shoot-expert-history-mode` 仅在显式专家诊断时允许使用；禁止专家接管与原planner latent训练状态记录同时启用，以免保存未执行的latent作为标签。保留原始ONNX动作的顺序、幅度与实际环境动作接口，AH历史根据实际执行动作推进。每步检查传给诊断器的基础射门输入与环境观测一致。专家历史单独记录并审计，机器人本体/AH历史不重置。

原ONNX SHA256 `38c72a52330a9bdfda663e6312d3d451b7b55dec19d720ee818b12eddf41c43c`。167项测试通过；真实ONNX的16个原生状态输出保持，16个历史handoff状态配合显式阶段标记的完整历史输出与手工直接调用ONNX逐值一致。见 `planner_shoot_expert_history_routing.json`。这些仅是路由验证，不是新的专家闭环性能。

本实验在四种子确认完整结束、执行审计和冻结哈希核验之后才接入。当前已启动，尚无完整结果。运行状态 `runs/planner_shoot_expert_history_study_r1_status.json`，协议见 `planner_shoot_expert_history_protocol.json`。若专家在修正输入下可靠而统一policy仍差，才有依据进一步检查latent映射/监督；若专家也不可靠，则需继续处理连续上下文本身，不能盲用专家标签。
