# Passing + shooting pilot, 2026-11-13

Request: incorporate `body_yaw_49999.onnx` shooting expert with passing, retaining
Isaac Lab and AH-BFM and borrowing FIBER's action-aligned representation.

Remote fetch completed; g1-rollingball main matches origin/main at 1d3562d.
Existing user modifications retained. Remote default master is older; no switch
or reset was performed. Current main removed the historical shooting env class.
The existing mounted batchmpc_issaclab kick config is loaded as a separate module,
using current Isaac/whole_body_tracking runtime and body-yaw observations.
This is a reconstruction, not a recovered exact training run. Teacher autonomous
performance is necessary to validate it; ONNX shape matching alone is insufficient.

Shooting export: inputs obs1030, rma_obs150, time_step1; action29. Metadata has
body-yaw task observations, nominal PD, joint ordering, static ball and target
speed6m/s. Export metadata does not encode all training config/reset parameters.
Pass dataset: existing expert_dense_v1, 2000 steps x16 lanes, input1200.
Shooting collector: 500 steps x16 lanes, actor1030+RMA150. Trace has frozen-BFM
inputs and expert physical joint targets. The first collection used default ORT
thread settings; subsequent launches cap intra/inter-op threads at1.

## Pilot scope

A single task-conditioned projector outputs256D norm16 latents to frozen AH-BFM.
Pass/shoot observations are padded to1200 and appended with explicit two-class
one-hot skill conditioning. Expert observation semantics remain skill-specific;
this is a privileged observation baseline, not a unified perception interface.
Each optimizer update uses64 samples per skill with equal weight. Lanes0:12 train;
12:16 validate (all their frames withheld). Fresh projector; AdamW1e-4;1000 updates.
Full actor is frozen. SwanLab online logs per-skill action losses. No backward
encoder or semantic regularization is claimed. No flow-matching planner is claimed.

This deliberately tests joint projection first, a simpler experiment than the
paper's separate LPP -> closed-loop latent dataset -> offline flow planner.
Separate fresh shooting projection and the existing passing s1 remain necessary
future controls if joint performance is poor. Do not attribute failures to
multitask interference without those matched controls.

## Evaluation

Driver schedules16 scenes per skill with new seed811314, up to2000 steps,
expert and shared-projector autonomous arms. All evaluation scenes are excluded
from training. First episodes captured before reset; report both skills
separately, including terminations and first kick speed/direction. Raw teacher
collection summaries count repeated episodes and are NOT first-episode rates.

Artifacts:
- runs/shoot_expert_probe_v1/{audit.json,teacher_records.pt,summary.json}
- runs/pass_shoot_joint_pilot_v1/{config.json,metrics.jsonl,latest.pt}
- runs/pass_shoot_{shoot,pass}_{expert,joint}_eval_v1/
- runs/pass_shoot_pilot_status.json

Reference: https://arxiv.org/html/2610.24675 (FIBER), especially action-aligned
projection and closed-loop latent rollout aggregation. Full paper pipeline is
not established by this pilot.

## Training completed

1000 optimizer updates completed; SwanLab run `n7g5fab7` uploaded328 records.
Sampled heldout physical-target MSE: pass0.03413→0.00976, shoot0.09912→0.01803
(rad²). This measures heldout teacher states only, not autonomous skill success.
Relevant existing tests: test_evaluation_audit, test_lpp, test_pass_measurement —6 passed.
Shooting ONNX SHA256: `38c72a52330a9bdfda663e6312d3d451b7b55dec19d720ee818b12eddf41c43c`.

Shooting teacher evaluation completed:16/16 kicks,14/16 direction within10deg.
15 terminations are expected event_kick_elapsed,1 is anchor_pos; do not report
16 falls. Scene hash `fcbdafe56d14e9fa0393388dfdff678854f6f17436b95271f529d22fa4fd2969`.

Shooting joint evaluation completed with identical initial-state SHA256 as
teacher:3/16 quality kicks,0/16 direction success. Termination reasons:
11 timeouts,2 wrong-foot terminations,2 normal post-kick ends,1 anchor orientation.
Do not equate event_counts (5 total kick events) with the first-episode
quality_kick_success count (3). Autonomous mean target MSE0.4143 versus sampled
heldout teacher-state MSE0.0180; mean target clipping fraction0.1910. These are
different state distributions, supporting further on-policy data aggregation,
not proving that joint training itself causes interference.

## Final paired evaluation

| Skill | Policy | Kick success | Direction <=10deg | Termination reasons |
|---|---|---:|---:|---|
| pass | expert | 16/16 | 13/16 | {'time_out': 16} |
| pass | joint | 1/16 | 0/16 | {'missed_ball': 14, 'anchor_ori': 1, 'anchor_pos': 1} |
| shoot | expert | 16/16 | 14/16 | {'event_kick_elapsed': 15, 'anchor_pos': 1} |
| shoot | joint | 3/16 | 0/16 | {'wrong_foot_kick': 2, 'event_kick_elapsed': 2, 'anchor_ori': 1, 'time_out': 11} |

Both skill comparisons have identical within-skill initial-state hashes.
