# MoS — Master of Soccer

利用身体基础模型（BFM）的运动能力，研究人形机器人的足球交互与多技能控制。
当前统一策略用传球和射门专家的动作监督训练同一个 latent planner，部署时调用冻结的 AH-BFM。

**这是独立研究代码，不是 FIBER 官方实现。当前尚未追平专家准确率，
同一 checkpoint 已有自主传球和射门结果；传球精度与连续切换仍待改善。**

## 当前方法

```text
任务观测（机器人、球、目标） → 统一 latent planner → 256维 latent → 冻结 AH-BFM → 关节目标
                                      ↑
                  学生实际访问状态上的专家动作监督 + 专家轨迹回放
```

- 仅更新 projector，保留 actor 的关节顺序、历史、缩放和控制契约。
- 初期 LPP 使用动作监督；现已接入作者配套 backward encoder，并实验性加入语义标签。身体映射尚未经作者参考实现确认。
- 在线训练与评估使用 Isaac Lab / Isaac Sim 中的外部 g1-rollingball 环境。
- SwanLab 记录训练。MuJoCo 视频工具重绘已记录的 Isaac 状态，不代表 MuJoCo 动力学评估。
- 内部 Python 包保留名称 `fiber_soccer`，以兼容已有实验脚本和环境。

## 目录

| 路径 | 内容 |
|---|---|
| `src/fiber_soccer/` | latent projector、控制接口、训练、评估与诊断 |
| `scripts/expert_baseline.py` | 专家评估、projector训练与自主评估入口 |
| `scripts/verify_actor.py`, `verify_real_inputs.py` | actor转换一致性检查 |
| `scripts/compare_paired.py`, `report_pass_quality.py` | 配对评估与传球质量统计 |
| `scripts/play_compare.py`, `render_trace.py` | 轨迹重绘 |
| `scripts/run_*.py`, `continue_*.py`, `distill_*.py` | 历史实验配方；含本机路径和特定实验依赖，运行前需修改 |
| `tests/` | 独立于Isaac的计算与接口测试 |
| `docs/` | 实验记录与研究笔记；历史记录不是最新状态摘要 |
| `reports/` | 组会材料；视频链接指向本地未纳入Git的文件 |

## 安装与外部依赖

Python >=3.11。复用已有兼容环境，不需要为本项目另建Isaac镜像：

```bash
python -m pip install -e '.[test,train,video]'
python -m pytest tests -q
```

这只安装Python组件，不安装Isaac Sim或g1-rollingball。完整运行还需要：

1. 已配置的 Isaac Lab / Isaac Sim，以及提供
   `Tracking-Flat-G1-Rolling-Pass-Event-v1-stage2` 的 g1-rollingball 环境。
2. 传球专家 checkpoint 及同目录 `params/env.pkl`、`params/agent.pkl`；
   环境代码版本须与保存的配置兼容。
3. AH-BFM actor与配套资源，详见 [资源说明](docs/assets_setup.md)。
4. SwanLab 登录或在运行环境中配置其离线模式。不要提交凭据。

模型、专家数据、实验日志和视频不随本仓库发布。仓库不是无需外部依赖的一键复现包。

## 运行

在配置好Isaac的Python环境中，将仓库 `src` 和外部环境包加入 `PYTHONPATH`。
`MOS_ASSETS_DIR` 默认为仓库下的 `assets/`，可指定绝对路径。

```bash
# 以下路径需换成自己的实际路径；输出目录必须尚不存在。
python scripts/expert_baseline.py --headless --device cuda:0 \
  --checkpoint /path/to/expert/model.pt --evaluate \
  --num-envs 64 --steps 2000 --seed 810127 --output /path/to/new_expert_eval

python scripts/expert_baseline.py --headless --device cuda:0 \
  --checkpoint /path/to/expert/model.pt --train-lpp \
  --num-envs 64 --steps 64000 --latent-interval 1 \
  --output /path/to/new_training_run

python scripts/expert_baseline.py --headless --device cuda:0 \
  --checkpoint /path/to/expert/model.pt --evaluate \
  --projector-checkpoint /path/to/new_training_run/latest.pt \
  --num-envs 64 --steps 2000 --seed 810127 --output /path/to/new_student_eval
```

已有Docker环境可通过 `scripts/container_exec.sh` 调用上述脚本。支持
`MOS_CONTAINER`、`MOS_CONTAINER_ROOT`、`MOS_ROLLINGBALL_ROOT`、`MOS_ISAAC_PYTHON`；
默认值保留原本地runtime。脚本参数中的路径必须是容器内路径。
`MOS_SWANLAB_PROJECT` 默认为 `MoS`，`MOS_SWANLAB_WORKSPACE` 可选。

训练前须核对专家环境、球参数与actor转换一致性。训练入口检查
`verification.json` 和 `real_verification.json`；不要手工伪造审计结果。

## 已验证结果与限制

同一64场景开发集，第一球事件末方向误差≤10°：专家59/64，保留的自主s1为40/64；
继续训练对照37/64，混合辅助回放31/64。未采用后两者。
独立128场景：专家108/128，自主s1为68/128。
这是方向指标，不是方向、力度和稳定性联合成功率。

[实验细节](docs/passing_optimization.md) ·
[组会材料](reports/group_meeting_2026-11-11/meeting.md) ·
[METATRON讨论](docs/metatron_research.md)

## 来源与权重

方法参考 [FIBER](https://fiberaa.github.io/) / [论文](https://arxiv.org/abs/2610.24675)。
AH-BFM演示资产与外部专家、机器人模型的使用须遵循各自许可。
本仓库不重新分发这些第三方资产，不声称复现了完整FIBER。

## 2026-11-16 实验状态

传球/射门共享 projector 的三轮闭环训练已完成。按两技能中较低方向成功率选择的第2轮，在同一开发种子、每技能16个 episode 上：传球触球16/16、方向3/16；射门触球11/16、方向5/16。原传球 s1 同场景方向8/16；两个专家分别13/16、14/16。当前模型尚不具备可靠的双技能表现。

详见 [联合实验完整结果](reports/joint_v2_debug_20261113.md)、[初版对照](reports/pass_shoot_pilot_20261113.md) 和 [latent planner 修复路线](docs/latent_planner_repair_20261113.md)。下一条主路线是独立技能投影、完整闭环 latent 轨迹、统一 planner 和连续切换验证；planner 已实现，但自主双技能效果尚未达标。UFO 诊断代码与 [诊断报告](reports/ufo_standalone_20261111.md) 一并保留。

仓库只包含代码、测试和报告；模型权重、专家数据、仿真资源和运行输出需按本地环境准备，不随 Git 上传。

## 统一 planner 迭代（2026-11-16）

已接入作者本地提供的 `AH-BFM/` actor，验证与原 ONNX 控制兼容。初版 `LatentPlanner` 使用统一945维真值状态/历史/目标/技能命令，输出5帧 latent，50Hz执行首帧并重规划。两技能共用一个 checkpoint，不在推理时调用专家或在线 oracle。

统一加载入口：`fiber_soccer.planner_policy.load_soccer_policy(checkpoint, device)`，按checkpoint配置加载前馈或内部目标求解控制器；普通前馈模型仍可直接使用`SoccerPolicy`。调用 `policy(observation, skill="pass")` 或 `skill="shoot"` 返回同一AH关节顺序的 `joint_targets`、`action_scaled`、`action_normalized` 与 `latent`。观测使用 `planner.observation` 的schema，调用方必须按实际执行动作维护AH历史。skill也可为逐环境的0/1张量。原型使用仿真真值，未声称真实感知能力。

最新进展见 [当前状态](reports/planner_current_status.md)。仅初始传球使用16步CUDA graph修正的候选，在四个新种子、每模型128回合的连续确认中，准确传球/准确射门/完整成功为50/24/7，r3为31/16/2；但候选存活23/128低于r3的27/128，稳定性仍未解决，默认保持r3。同一policy保存初始backward目标用于射门后恢复的候选，已在两组开发连续对照中把完整成功3/64提高到7/64、存活10/64提高到28/64，实际进入恢复的27个环境全部持续到20秒；射门触球和单项准确率未提高，四个新种子的三模型确认正在运行。新增恢复记忆要求从物理reset后的首次观测开始调用、保持批槽位并执行 `policy.reset(done)`。所有扩展均通过 `load_soccer_policy` 加载；不同控制器共用技能、方向、速度参数，不查询专家动作。详见 [恢复候选](reports/planner_initial_goal_recovery.md) 与 [内部目标求解实验](reports/planner_predicted_target_solver.md)。

当前保留的开发集候选是 `runs/planner_history_r3_contact_train/best.pt`：2406维v4历史输入、条件endpoint flow、50Hz预测latent并执行首帧，两技能使用同一个checkpoint。开发种子811314、每技能16环境：传球14/16有效触球、6/16方向≤10°；射门16/16、11/16。推理不调用专家或在线latent优化。单实例混合任务调用已核对，与分别调用的关节目标和latent一致。

该候选还不能作为稳定双技能控制器：此前r2历史版在新种子812901上，传球仅17/32触球、1/32方向达标（专家32/32、27/32）；射门32/32、14/32（专家32/32、25/32）。r3触球版已完成种子812902、32环境/技能的完整配对验证：传球28/32触球、5/32方向达标（专家32/32、31/32）；射门28/32、16/32（专家30/32、22/32）。传球精度仍明显不足。原专家物理目标监督续训另在种子812903上完成配对：传球32/32触球、7/32方向达标（专家32/32、30/32），射门26/32、20/32（专家32/32、25/32）。不同种子的成绩不能直接归因于模型变化。

模型实现同时保留945/950/1206维旧输入、2406维历史输入，以及单步latent回归对照。部署必须使用checkpoint的 `policy.planner.schema`；含backward先验的schema还需传入 `backward_model=policy.actor.model`。单步latent对照的开发集传球为10/16、1/16，射门14/16、4/16，当前不替代flow候选。离线动作回归只用于诊断，不是部署策略。

完整实验见 [迭代记录](reports/planner_iteration_20261116.md)、[自主结果](reports/planner_autonomous_results.md) 和 [新种子配对验证](reports/planner_fresh_seed_results.md)。作者提供的 `AH-BFM/` 权重仅在本地使用，不随仓库发布。


实验性v4输入需要向 `planner.observation` 额外传入 `task_observation=torch.cat((obs["policy"], obs["rma_obs"]), -1)`，同时传入配套 `backward_model`。这些是当前任务配置下的观测管理器历史，不是专家动作；不能把两种任务的原始历史布局任意互换。v4已接入同一个 `SoccerPolicy`，开发种子闭环评测及单实例混合调用已通过执行检查；已有新种子验证暴露传球方向问题，技能切换已加入可选0.2秒latent过渡：8环境/4秒探针曾由6/8提升到8/8无重置，但新种子32环境/10秒压力测试中直接与插值切换均为0/32无重置。插值减小瞬间动作突变，尚未实现持续切换；早期完整传球后射门序列测试当时未出现双球方向同时达标，后续已有此类回合，但稳定完成率仍低；最新数据以当前状态报告为准。输入对照见 [观测历史实验](reports/planner_input_comparison.md)。

连续环境可选择 `SoccerPolicy(checkpoint, device, transition_steps=10)`，在50Hz控制下对技能变化插值0.2秒。该模式有逐环境运行状态：每个控制步调用一次、保持batch槽位与环境对应，并在环境重置后调用 `policy.reset(done_mask)`；更换整个batch前调用 `policy.reset()`。观测管理器两套任务历史与AH动作历史仍由调用方维护。默认 `transition_steps=0`，既有单技能评测使用该默认设置。短程配对结果见 [技能切换验证](reports/planner_skill_switch_20261116.md)。


实验性传球适配器支持两种修正位置：默认 `--adapter-space endpoint` 在flow迭代中修正预测终点；`--adapter-space latent` 在冻结flow完成后只修正首个执行latent。后者训练需同时指定 `--skill-adapter pass --adapter-observation-input --flow-weight 0 --action-target expert`，使用独立的 `flow_latent_adapter_v1` checkpoint类型，仍由同一个 `SoccerPolicy` 加载。零初始化保持原输出，训练只更新传球适配器，保留latent半径和其余未来帧。该模式已完成20000次更新与真实状态保持性检查；开发评测传球15/16触球、5/16方向≤10°，射门16/16、11/16，未替换默认候选。
