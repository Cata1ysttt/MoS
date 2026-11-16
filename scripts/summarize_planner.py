"""Summarize only finished autonomous episodes, with assistance provenance."""
import json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1];runs=root/'runs'
names=[('Original pass s1','joint_v2_pass_s1_control'),('Unified v1 pass','planner_v1_pass_eval'),('Unified v1 shoot','planner_v1_shoot_eval'),('Reset-balanced shoot','planner_reset_balanced_shoot_eval'),('DAgger r1 pass','planner_dagger_r1_pass_eval'),('DAgger r1 shoot','planner_dagger_r1_shoot_eval'),('Unified v2 pass','planner_v2_pass_eval'),('Unified v2 shoot','planner_v2_shoot_eval'),('Backward residual pass','planner_v3_pass_eval'),('Backward residual shoot','planner_v3_shoot_eval'),('Flow v1 pass','planner_flow_v1_pass_eval'),('Flow v1 shoot','planner_flow_v1_shoot_eval'),('Flow wide pass','planner_flow_wide_pass_eval'),('Flow wide shoot','planner_flow_wide_shoot_eval'),('Flow action pass','planner_flow_action_pass_eval'),('Flow action shoot','planner_flow_action_shoot_eval')]
lines=['# Unified planner autonomous results','', '| Model/task | Episodes | Kick | Direction ≤10° | Termination terms |','|---|---:|---:|---:|---|']
result={}
for label,name in names:
 path=runs/name
 if not (path/'summary.json').exists():continue
 d=json.loads((path/'summary.json').read_text())
 assert not d['expert_assisted_oracle']
 if d['completed']!=d['requested']:continue
 rows=json.loads((path/'episodes.json').read_text())
 terms=dict(collections.Counter(k for row in rows for k,v in row['termination_terms'].items() if v))
 result[name]=dict(summary=d,terms=terms)
 lines.append(f"| {label} | {d['completed']} | {d['quality_kick_success']:.1%} | {d['quality_ball_direction_success_10deg']:.1%} | {terms} |")
lines+=['','All runs use development seed 811314; results are not unseen-seed validation. Assisted training rollouts and incomplete smoke runs are excluded. Normal post-kick termination is not a fall. A single checkpoint must perform both tasks; this table does not claim no-reset skill switching.']
(root/'reports/planner_autonomous_results.md').write_text('\n'.join(lines)+'\n')
(runs/'planner_autonomous_results.json').write_text(json.dumps(result,indent=2))
print('\n'.join(lines))
