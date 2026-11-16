"""Summarize only finished autonomous episodes, with assistance provenance."""
import json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1];runs=root/'runs'
names=[('Original pass s1','joint_v2_pass_s1_control'),('Unified v1 pass','planner_v1_pass_eval'),('Unified v1 shoot','planner_v1_shoot_eval'),('Reset-balanced shoot','planner_reset_balanced_shoot_eval'),('DAgger r1 pass','planner_dagger_r1_pass_eval'),('DAgger r1 shoot','planner_dagger_r1_shoot_eval'),('Unified v2 pass','planner_v2_pass_eval'),('Unified v2 shoot','planner_v2_shoot_eval'),('Backward residual pass','planner_v3_pass_eval'),('Backward residual shoot','planner_v3_shoot_eval'),('Flow v1 pass','planner_flow_v1_pass_eval'),('Flow v1 shoot','planner_flow_v1_shoot_eval'),('Flow wide pass','planner_flow_wide_pass_eval'),('Flow wide shoot','planner_flow_wide_shoot_eval'),('Flow action pass','planner_flow_action_pass_eval'),('Flow action shoot','planner_flow_action_shoot_eval'),('Flow DAgger r1 pass','planner_flow_dagger_r1_pass_eval'),('Flow DAgger r1 shoot','planner_flow_dagger_r1_shoot_eval'),('Flow DAgger weighted pass','planner_flow_dagger_r1_weighted_pass_eval'),('Flow DAgger weighted shoot','planner_flow_dagger_r1_weighted_shoot_eval'),('Flow contact pass','planner_flow_dagger_r1_weighted_contact_pass_eval'),('Flow contact shoot','planner_flow_dagger_r1_weighted_contact_shoot_eval')]
names += [('Action alignment pass','planner_action_alignment_r1_pass_eval'),
          ('Action alignment shoot','planner_action_alignment_r1_shoot_eval'),
          ('Flow continuation pass','planner_flow_continue_r1_pass_eval'),
          ('Flow continuation shoot','planner_flow_continue_r1_shoot_eval')]
names += [(label+' '+skill, name+'_'+skill+'_eval')
          for label,name in [('Recovery r2 replay','planner_flow_dagger_r2'),
                             ('Recovery r2 state only','planner_state_r2_only'),
                             ('Recovery r2 history','planner_history_r2'),
                             ('Recovery r3 history','planner_history_r3'),
                             ('Recovery r3 contact','planner_history_r3_contact'),
                             ('Recovery r3 direct','planner_direct_r3'),
                             ('Recovery r3 action alignment','planner_history_r3_action'),
                             ('Recovery r3 continuation','planner_history_r3_continue'),
                             ('Recovery r3 approach','planner_history_r3_approach'),
                             ('Recovery r3 expert target','planner_history_r3_expert_target'),
                             ('Recovery r3 corrected labels','planner_history_r3_sum_labels'),
                             ('Recovery r4','planner_history_r4'),
                             ('Recovery r4 pass replay','planner_history_r4_pass_replay'),
                             ('Recovery r4 pass adapter','planner_history_r4_pass_adapter'),
                             ('Recovery r4 adapter direct observations','planner_history_r4_pass_adapter_obs'),
                             ('Recovery r4 adapter direct observations expert target','planner_history_r4_pass_adapter_obs_expert'),
                             ('Recovery r4 adapter ball boundaries','planner_history_r4_pass_adapter_boundary'),
                             ('Recovery r4 adapter action only','planner_history_r4_pass_adapter_action_only'),
                             ('Recovery r4 wider pass adapter','planner_history_r4_pass_adapter_wide'),
                             ('Recovery r4 spatially gated adapter','planner_history_r4_pass_adapter_near')]
          for skill in ['pass','shoot']]
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

fresh_lines=['# Paired fresh-seed planner validation','',
             '| Candidate | Seed | Task | Controller | Episodes | Kick | Direction ≤10° |',
             '|---|---:|---|---|---:|---:|---:|']
fresh_results={}
hash_notes=[]
for tag in ['planner_flow_dagger_r1_fresh','planner_history_r2_fresh','planner_history_r3_contact_fresh','planner_history_r3_expert_target_fresh','planner_history_r4_pass_adapter_obs_expert_fresh','planner_history_r4_pass_adapter_obs_expert_fresh2']:
 status_path=runs/(tag+'_status.json')
 if not status_path.exists():continue
 status=json.loads(status_path.read_text())
 if status['status']!='completed':continue
 protocol=json.loads((runs/(tag+'_protocol.json')).read_text())
 fresh_results[tag]=dict(protocol=protocol,results={})
 for skill in ['pass','shoot']:
  pair={}
  for controller in ['planner']+(['baseline'] if protocol.get('baseline_checkpoint') else [])+['expert']:
   d=json.loads((runs/(tag+'_'+skill+'_'+controller)/'summary.json').read_text())
   assert d['completed']==d['requested']==protocol['num_envs']
   assert d['seed']==protocol['seed'] and not d['expert_assisted_oracle']
   pair[controller]=d
   fresh_lines.append(f"| {tag} | {d['seed']} | {skill} | {controller} | {d['completed']} | {d['quality_kick_success']:.1%} | {d['quality_ball_direction_success_10deg']:.1%} |")
  assert len({value['initial_state_sha256'] for value in pair.values()})==1
  fresh_results[tag]['results'][skill]=pair
 hash_notes.append(f"Candidate `{tag}` checkpoint SHA256: `{protocol['checkpoint_sha256']}`.")
 if protocol.get('baseline_checkpoint_sha256'):hash_notes.append(f"Baseline for `{tag}` checkpoint SHA256: `{protocol['baseline_checkpoint_sha256']}`.")
fresh_lines+=['']+hash_notes
fresh_lines+=['','Only completed paired evaluations are included. The seed was absent from the candidate training metadata. Each planner uses one frozen checkpoint for both skills; initial physical states match each expert comparison. Rates count the first episode per environment, not the number of ball events. These evaluations do not establish continuous skill switching.']
(root/'reports/planner_fresh_seed_results.md').write_text('\n'.join(fresh_lines)+'\n')
(runs/'planner_fresh_seed_results.json').write_text(json.dumps(fresh_results,indent=2))
