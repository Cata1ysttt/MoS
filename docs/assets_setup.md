# 外部资源与运行前核查

资源目录由 `MOS_ASSETS_DIR` 指定，默认仓库 `assets/`。该目录不纳入Git。
当前训练入口使用：

- `actor.onnx`：官方演示AH-BFM actor，需核对版本和来源。
- `actor_meta.json`：关节名称与动作缩放。
- `pickup_scene.json`：与导出actor契约一致的默认关节位置等配置。
- `z_bank.json`：包括stand latent，不能替代backward encoder。
- `expert_rewards.py`：匹配已保存专家环境的奖励实现快照；须从自己有权限的g1-rollingball环境获取。
- `verification.json`、`real_verification.json`：由转换校验生成的结果，不是下载权重的一部分。

模型来源：https://fiberaa.github.io/ ，网页资源清单位于
https://fiberaa.github.io/public-models/manifest.json 。请先核对清单与使用许可，
不能将任意同维latent actor与这些元数据混用。

`scripts/verify_actor.py` 和 `scripts/verify_real_inputs.py` 是本地验证工具；
使用前阅读其输入约定，后者需要真实rollout样本。训练入口保留既有工程误差门限，
历史严格数值误差与实际物理目标误差的区别记录在 `docs/status.md` 中。

外部专家checkpoint必须连同原始env/agent配置、环境实现一起使用。
本仓库没有包含这些资产，也没有完整的基础BFM预训练流水线。
