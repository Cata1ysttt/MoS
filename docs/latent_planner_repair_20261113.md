# 传球与射门的 latent planner 修复路线

目标：部署时由一个条件 planner 根据机器人/球状态、技能命令和目标，生成可执行 latent 序列；同一个冻结 AH-BFM 在 Isaac Lab 中完成传球与射门。专家和技能 projector 仅用于训练数据生产，不依赖部署时切换两个专家。

## 实测状态与方法纠偏

2026-11-16 状态更新：joint_v2 三轮已完成，最新结果见 `reports/joint_v2_debug_20261113.md`。以下第一轮诊断保留为修复方案的依据。第一轮同一 checkpoint：传球方向 5/16、触球 15/16；射门方向 1/16、触球 5/16。原传球 s1 同场景方向 8/16、触球 16/16。专家方向分别 13/16、14/16。共享 projector 尚未成为合格的数据生成器，也不能把这组结果当成统一 planner 的结果。

射门在线专家辅助 latent 反演：触球 16/16、方向 7/16，全为正常 event_kick_elapsed 结束。它支持修复 latent 推断，但不是自主成绩、不是能力上界，也没有追平专家方向准确率。

FIBER 依据：arXiv:2610.24675v2，III-C/III-D、Appendix C/D。其顺序是独立技能投影、投影策略闭环采集、统一离线 flow planner。当前共享 projector 作为对照保留；主路线不以先训练成功一个共享 projector 为前提。

## 阶段 1：分别得到可靠的可执行技能投影

- 保留传球 s1，单独修复射门 projector；两者共用完全相同的 AH actor、关节映射、动作范围、PD 与历史接口。
- 射门以专家动作反演 latent 作为训练辅助，在学生状态上做 DAgger。反演结果不能直接成为部署能力的证据。
- 训练反演使用连续初值及短时间窗口动作误差；比较 latent 连续性约束，接触阶段降低平滑约束，避免抹掉踢球冲量。
- 同一 actor 输出存在动作等效 latent 的可能。配套 backward encoder 若可获得，使用其原生正则；否则把连续性、固定反演初值与窗口动作监督明确标为替代实验，不能声称等价原论文。
- 使用训练/开发/最终测试分离的种子和完整 episode。先确保传球不低于 s1，再提升射门稳定性、方向和速度。最终测试建议每技能至少 3 个新种子、每种子 64 个 episode，并报告区间。

## 阶段 2：先解决执行频率，再采 planner 标签

当前 checkpoint 的 latent_interval=1，即每 20 ms 更新。论文的 10 Hz latent 和 50 Hz actor 不能直接套用：把现有 z 每五帧抽样或保持，会改变执行的控制策略。

建议先用状态条件的小型序列 planner，预测未来 0.2–0.4 s 的 50 Hz latent（10–20 帧），每次执行首个 20 ms 命令后重规划，检查可学习性与计算耗时。随后测 2 帧/5 帧执行块的稳定性和接触精度；如推理不满足 20 ms 预算，则重新训练并验证对应低频投影，或研究高频 latent feedback。推理预算、执行频率与采集频率必须一起验证，不能只改配置降频。

Flow planner 与单步/序列回归基线使用同一数据和时间契约。插值只能在已经闭环验证过的执行约定中使用。

## 阶段 3：重采统一的完整轨迹

当前 student_replay 保存 conditioned expert observation、BFM state/history、expert target；没有完整 episode 标识和实际执行 latent。evaluation 的 action_diagnostics 每五帧保存一次，而其 latent 每帧更新。这两类文件都不能直接充当完整 planner 序列数据。

新增记录契约（每个实际执行的控制步）：

- episode_id、tick、seed、skill、source checkpoint/actor hash、dt、latent_interval、执行/插值规则；
- 统一 planner observation：关节位置/速度、躯干角速度、重力方向、历史命令、球相对位置/速度及可见性；
- command：技能 pass/shoot、目标点/方向、目标出球速度、有效性 mask；坐标统一到机器人 yaw frame，量纲明确；
- pre-step 的 BFM state/last/history、真正执行的 z、实际关节目标、done/reset；
- 成功、方向/速度误差、踢球后稳定性用于质量分析或训练权重，不作为可见输入；
- 标记 expert-assisted/teacher-action 样本，默认 planner 正例只用自主 projector 的实际闭环执行数据。

两种专家观测不直接 padding 后当统一观测。当前 pass 的 policy/RMA 是 1100+100，shoot 为 1030+150；球位置维度、历史及相位字段不同。planner 使用独立的规范化 schema。初版使用仿真真值可验证控制结构，但须标记为 privileged-state prototype；实际感知能力需要另行验证。

轨迹按完整 episode 和种子划分；窗口不得穿过 reset。保留恢复过程和困难成功轨迹，不只截取触球前的漂亮片段。两技能平衡采样，记录各自覆盖范围。

## 阶段 4：训练统一 planner

部署：统一状态/历史 + 技能/目标命令 -> 一个 planner -> latent 序列 -> 冻结 AH-BFM -> 关节目标。

训练：分别执行 P_pass / P_shoot -> 统一轨迹数据集 -> 同一个 planner。

首先用显式 skill + goal command 验证可控性。若目标还包括自主决定何时传、何时射，需要额外战术条件/奖励或决策器；这不是增加一个技能标签就自然获得的能力。

同数据比较：单步 latent 回归、短序列回归、条件 flow matching。可加入解码动作辅助损失，但须标记为足球任务扩展；FIBER 报告的 planner 训练未使用该辅助项。不能用 B(s) 替换实际执行 latent 作为 planner 标签。

## 阶段 5：同环境技能切换与验收

当前传球/射门分别在两个环境中评测，不能证明技能连贯切换。需建立一个物理/PD/观测契约一致的 soccer 环境，覆盖滚动球传球和静态/低速球射门，并重新验证专家在此环境中的适用范围。

切换时保留真实机器人状态及 AH 的动作/历史缓存；作废旧 planner 未执行序列并重新规划，不能重置机器人或清空历史来伪装切换成功。采集接近、调整、踢球、恢复阶段，分别测试接触前、接触中、踢后命令变化；对不可立即执行的切换定义清晰的承诺窗口/恢复阶段，并使 planner 能观测当前执行阶段。

逐层报告：专家 -> 独立 LPP -> 同一 planner 的两种命令 -> 不 reset 的连续切换。指标包括触球、方向、球速、跌倒、踢后稳定、命令响应延迟、推理时间。建议 planner 初步验收目标为每技能保留其独立 LPP 至少 90% 的方向成功率（拟定门槛，非现有结果）；最终效果仍与专家比较。

最小定位消融：独立 vs 共享 LPP；纯动作 vs 连续/语义正则；每帧 vs 2/5 帧执行；回归 vs flow；有/无切换轨迹。

## 当前执行边界

现有 joint_v2 三轮训练已完成，结果保留作为对照。本文定义下一条主修复路线；统一 planner、轨迹 recorder 和同环境切换尚未实现，本文不宣称端到端能力已经完成。
