"""Train a continuous-shooting adapter, audit it, and evaluate native and fresh sequences."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from audit_sequence_recovery import validate

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--tag',default='planner_sequence_shoot_adapter_r1')
p.add_argument('--updates',type=int,default=10000)
p.add_argument('--seeds',type=int,nargs='+',default=[815301,815302])
a=p.parse_args()
assert a.tag.replace('_','').isalnum() and a.updates>0
assert len(a.seeds)==len(set(a.seeds)) and not set(a.seeds)&{811314,813201,815101,815102,815201,815211}
root=Path(__file__).resolve().parents[1];run=root/'runs'/a.tag
status=root/'runs'/f'{a.tag}_status.json';assert not status.exists() and not run.exists()
base=root/'runs/planner_history_r3_contact_train/best.pt'
labels=root/'runs/planner_sequence_shoot_r1_training/shoot_labels_clamped.pt'
retention=[root/'runs'/f'planner_recovery_r{i}_sum_semantic/shoot_trajectory.pt' for i in [2,3]]
passing=root/'runs/planner_recovery_r2_sum_semantic/pass_trajectory.pt'
results={}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def container(path):return '/workspace/fiber-soccer/'+str(path.relative_to(root))
def state(stage,**details):
    status.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),seeds=a.seeds,
        results=results,**details),indent=2))
def execute(command,log,gpu):
    with log.open('x') as f:
        subprocess.run([str(root/'scripts/container_exec.sh')]+command,stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES=str(gpu)),check=True)
try:
    smoke=json.loads((root/'runs/planner_sequence_shoot_adapter_smoke/complete.json').read_text())
    assert smoke['smoke_test'] and smoke['deployable_roundtrip'] and smoke['pass_probe_outputs_unchanged']
    state('training',checkpoint_selection='final_iteration')
    execute(['/workspace/fiber-soccer/scripts/train_sequence_shoot_adapter.py','--base',container(base),
        '--labels',container(labels),'--retention-data',*[container(x) for x in retention],
        '--pass-probe-data',container(passing),'--output',container(run),'--updates',str(a.updates)],
        run.with_suffix('.log'),2)
    complete=json.loads((run/'complete.json').read_text());checkpoint=run/'final.pt';digest=sha(checkpoint)
    assert not complete['smoke_test'] and complete['updates']==a.updates
    assert complete['shared_tensors_unchanged'] and complete['pass_probe_outputs_unchanged'] and complete['deployable_roundtrip']
    assert complete['checkpoint_sha256']==digest
    state('auditing_protected_pass',checkpoint_sha256=digest)
    execute(['/workspace/fiber-soccer/scripts/audit_skill_adapter.py','--base',container(base),
        '--adapted',container(checkpoint),'--data',container(passing),container(retention[0]),
        '--device','cuda:0','--output',container(root/'runs'/f'{a.tag}_preservation.json')],
        root/'runs'/f'{a.tag}_preservation.log',2)
    results['native']={}
    for skill in ['pass','shoot']:
        destination=root/'runs'/f'{a.tag}_{skill}_eval'
        state('evaluating_native_'+skill,checkpoint_sha256=digest)
        entry='expert_baseline.py' if skill=='pass' else 'probe_shoot_expert.py'
        cmd=['/workspace/fiber-soccer/scripts/'+entry,'--headless','--device','cuda:0','--evaluate',
            '--seed','811314','--num-envs','16','--steps','2000',
            '--projector-checkpoint',container(checkpoint),'--output',container(destination)]
        if skill=='pass':cmd+=['--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
        execute(cmd,destination.with_suffix('.log'),3)
        s=json.loads((destination/'summary.json').read_text())
        reference=root/'runs'/f'planner_history_r3_contact_{skill}_eval'
        baseline=json.loads((reference/'summary.json').read_text())
        assert s['completed']==s['requested']==16 and not s['expert_assisted_oracle']
        assert s['initial_state_sha256']==baseline['initial_state_sha256']
        if skill=='pass':
            for name in ['episodes.json','first_kick_velocity.json']:
                assert json.loads((destination/name).read_text())==json.loads((reference/name).read_text()),name
        results['native'][skill]=dict(candidate=s,baseline=baseline)
    for seed in a.seeds:
        pair={};protocols=[];contracts=[]
        for label,model in [('baseline',base),('candidate',checkpoint)]:
            destination=root/'runs'/f'{a.tag}_{seed}_{label}'
            state('evaluating_sequence',seed=seed,model=label,checkpoint_sha256=digest)
            cmd=['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py','--headless','--device','cuda:0',
                '--checkpoint',container(model),'--output',container(destination),'--num-envs','32','--steps','1000',
                '--seed',str(seed),'--handoff-seed',str(seed),'--scene-base','shoot','--action-contract','shoot',
                '--reset-contract','shoot','--events-contract','shoot','--incoming-pass-command',
                '--continuous-after-kick','--transition-steps','0','--post-shoot-recovery-steps','10']
            execute(cmd,destination.with_suffix('.log'),3)
            s=json.loads((destination/'summary.json').read_text())
            assert s['one_policy_instance'] and not s['expert_loaded'] and s['requested']==32
            assert s['checkpoint_sha256']==sha(model)
            pair[label]=dict(summary=s,accounting=validate(s))
            protocols.append(json.loads((destination/'protocol.json').read_text()))
            contracts.append(json.loads((destination/'scene_contracts.json').read_text()))
        b,c=[pair[label]['summary'] for label in ['baseline','candidate']]
        assert b['initial_state_sha256']==c['initial_state_sha256']
        assert protocols[0]==protocols[1] and contracts[0]==contracts[1]
        first=min(row['tick'] for row in b['switches'])
        for key in ['pass_impacts','shoot_impacts']:
            assert {k:v for k,v in b[key].items() if v['tick']<=first}=={k:v for k,v in c[key].items() if v['tick']<=first},key
        for key in ['switches','first_resets','contact_observations']:
            assert [v for v in b[key] if v['tick']<=first]==[v for v in c[key] if v['tick']<=first],key
        pair['audit']=dict(initial_tensors_equal=True,protocol_and_scene_equal=True,
            observed_prefix_equal=True,first_handoff_tick=first)
        results[str(seed)]=pair
        assert sha(checkpoint)==digest
    state('completed',checkpoint_sha256=digest,
        scope='Final-iteration shooting adapter, native-task development checks and two preselected new continuous seeds; no automatic promotion.')
except Exception as error:
    state('failed',error=repr(error));raise
