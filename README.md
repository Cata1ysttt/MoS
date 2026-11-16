# MoS — Master of Soccer

利用身体基础模型（BFM）的运动能力，研究人形机器人的足球交互与多技能控制。
当前实现将 G1 传球专家投影到冻结 AH-BFM 的 latent 空间。

**这是独立研究代码，不是 FIBER 官方实现。当前尚未追平专家准确率，
已有统一状态条件 latent 序列 planner 原型，正在闭环纠偏；自主双技能效果仍未达标。**

## 当前方法

```text
任务观测（机器人、球、目标） → projector → 256维 latent → 冻结 AH-BFM → 关节目标
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

部署入口：`fiber_soccer.planner_policy.SoccerPolicy(checkpoint, device)`；调用 `policy(observation, skill="pass")` 或 `skill="shoot"` 返回同一AH关节顺序的 `joint_targets`、`action_scaled`、`action_normalized` 与 `latent`。观测使用 `planner.observation` 的schema，调用方必须按实际执行动作维护AH历史。skill也可为逐环境的0/1张量。原型使用仿真真值，未声称真实感知能力。

现同时保留序列回归基线和条件 endpoint flow MLP。观测支持945/950/1206维版本；部署必须使用 checkpoint 的 `policy.planner.schema`，1206维观测还需向 `planner.observation` 传入 `backward_model=policy.actor.model`。已加入reset分层采样、学生状态DAgger和候选 backward 语义标签。扩大数据 flow 版本曾在两技能上均失败。最新完成的学生状态纠偏版本，在开发种子811314、每技能16场景上：传球有效触球5/16、方向2/16；射门5/16、3/16。两项使用同一checkpoint且无oracle，但成功率仍低。标签置信度加权对照及新种子32环境配对验证正在运行，不能部署为已验证的稳定双技能控制器。见 [迭代记录](reports/planner_iteration_20261116.md) 与 [自主结果](reports/planner_autonomous_results.md)。同环境连续切换与新种子成功率仍待验证。作者包仅在本地使用，不随仓库发布。
