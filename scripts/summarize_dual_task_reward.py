"""Recompute evaluation counts from all recorded first episodes."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--tag',default='planner_dual_task_reward_r3');a=p.parse_args()
root=Path(__file__).resolve().parents[1];runs=root/'runs'
status=json.loads((runs/(a.tag+'_status.json')).read_text())
assert status['stage']=='completed', 'Do not report unfinished evaluation as final'
record=json.loads((runs/(a.tag+'_results.json')).read_text())
protocol=record['protocol']
source_changes=[]
for name,digest in protocol['source_hashes'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        snapshot=runs/a.tag/Path(name).name
        assert snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest()==digest, name
        source_changes.append(name)

def accurate(event):
    return event['preferred_foot'] and event['direction_error_deg']<=10 and event['relative_forward_speed_error']<=.3

counts={}
for model in ['baseline','trained']:
    total=dict(episodes=0,passes=0,accurate_passes=0,shots=0,accurate_shots=0,complete_success=0,survivors=0,wrong_foot_shots=0)
    for seed in protocol['validation_seeds']:
        path=runs/f'{a.tag}_{model}_{seed}/summary.json';d=json.loads(path.read_text())
        assert d==record['evaluations'][f'{a.tag}_{model}_{seed}']
        assert d['one_policy_instance'] and not d['expert_loaded']
        assert hashlib.sha256(Path(d['checkpoint'].replace('/workspace/fiber-soccer',str(root))).read_bytes()).hexdigest()==d['checkpoint_sha256']
        total['episodes']+=d['requested'];total['passes']+=len(d['pass_impacts']);total['shots']+=len(d['shoot_impacts'])
        total['accurate_passes']+=sum(accurate(v) for v in d['pass_impacts'].values())
        total['accurate_shots']+=sum(accurate(v) for v in d['shoot_impacts'].values())
        failed={str(v['environment']) for v in d['first_resets']}
        full=sum(accurate(v) and key in d['shoot_impacts'] and accurate(d['shoot_impacts'][key]) and key not in failed for key,v in d['pass_impacts'].items())
        assert full==d['both_direction_foot_speed_and_continuous_through_end']
        total['complete_success']+=full;total['survivors']+=d['continuous_survivors']
        total['wrong_foot_shots']+=sum(not v['preferred_foot'] for v in d['shoot_impacts'].values())
    counts[model]=total
for seed in protocol['validation_seeds']:
    assert record['evaluations'][f'{a.tag}_baseline_{seed}']['initial_state_sha256']==record['evaluations'][f'{a.tag}_trained_{seed}']['initial_state_sha256']
passed=counts['trained']['complete_success']>counts['baseline']['complete_success'] and counts['trained']['accurate_shots']>=counts['baseline']['accurate_shots'] and counts['trained']['accurate_passes']>=counts['baseline']['accurate_passes']
result=dict(counts=counts,screen_passed=passed,post_run_source_changes=source_changes,changed_runtime_sources_verified_from_training_snapshots=True,scope='Two held-out seeds, 64 episodes per controller; task rewards plus new pass expert auxiliary loss. No causal separation of those two changes.',protocol=protocol,training=record['training'])
(root/'reports'/f'{a.tag}_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(counts,indent=2));print('screen_passed',passed)
