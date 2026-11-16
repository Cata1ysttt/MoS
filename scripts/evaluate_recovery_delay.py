"""Evaluate a delayed recovery adapter without changing its weights or task scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from audit_sequence_recovery import validate

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--training-run',type=Path,required=True)
p.add_argument('--delay',type=int,default=10)
a=p.parse_args()
if a.delay<1:p.error('Positive adapter delay required')
root=Path(__file__).resolve().parents[1];run=a.training_run.resolve();tag=run.name+f'_delay{a.delay}'
status=root/'runs'/f'{tag}_status.json';assert not status.exists()
def state(stage,**kwargs):
    status.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**kwargs),indent=2))
def container(path):return '/workspace/fiber-soccer/'+str(path.relative_to(root))
try:
    config=json.loads((run/'config.json').read_text());complete=json.loads((run/'complete.json').read_text())
    assert config['training_task']=='shoot_recover' and not complete['smoke_test']
    assert complete['precontact_probe_outputs_unchanged']
    checkpoint=run/'final.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest==complete['checkpoint_sha256']
    state('evaluating_original_tasks',checkpoint_sha256=digest)
    subprocess.run([sys.executable,str(root/'scripts/evaluate_latent_ppo.py'),'--training-run',str(run),
        '--evaluation-tag',tag,'--post-kick-adapter-delay',str(a.delay)],check=True)
    results={};protocols=[];contracts=[]
    for label,delay in [('immediate',0),('delayed',a.delay)]:
        destination=root/'runs'/f'{tag}_sequence_{label}'
        state('evaluating_sequence_'+label,checkpoint_sha256=digest)
        cmd=[str(root/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/probe_soccer_sequence.py',
            '--headless','--device','cuda:0','--checkpoint',container(checkpoint),'--output',container(destination),
            '--num-envs','16','--steps','1000','--seed','813201','--handoff-seed','813201',
            '--scene-base','shoot','--action-contract','shoot','--reset-contract','shoot',
            '--events-contract','shoot','--incoming-pass-command','--continuous-after-kick',
            '--transition-steps','0','--post-shoot-recovery-steps',str(config['recovery_delay_steps']),
            '--post-kick-adapter-delay',str(delay)]
        with destination.with_suffix('.log').open('x') as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
        summary=json.loads((destination/'summary.json').read_text())
        assert summary['checkpoint_sha256']==digest and summary['post_kick_adapter_delay']==delay
        assert summary['one_policy_instance'] and not summary['expert_loaded']
        results[label]=dict(summary=summary,accounting=validate(summary))
        protocols.append(json.loads((destination/'protocol.json').read_text()))
        contracts.append(json.loads((destination/'scene_contracts.json').read_text()))
    base,new=[results[key]['summary'] for key in ['immediate','delayed']]
    assert base['initial_state_sha256']==new['initial_state_sha256']
    assert contracts[0]==contracts[1]
    assert {k:v for k,v in protocols[0].items() if k!='post_kick_adapter_delay'}=={k:v for k,v in protocols[1].items() if k!='post_kick_adapter_delay'}
    contacts=[row for row in base['contact_observations'] if row['stage']==0]
    assert contacts,'No passing contact to exercise the intervention'
    first=min(row['tick'] for row in contacts)
    assert [row for row in base['contact_observations'] if row['tick']<=first]==[row for row in new['contact_observations'] if row['tick']<=first]
    for key in ['pass_impacts','shoot_impacts']:
        assert {k:v for k,v in base[key].items() if v['tick']<first}=={k:v for k,v in new[key].items() if v['tick']<first}
    for key in ['switches','first_resets']:
        assert [v for v in base[key] if v['tick']<first]==[v for v in new[key] if v['tick']<first]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    state('completed',checkpoint_sha256=digest,first_intervention_tick=first,initial_tensors_equal=True,
        observed_prefix_equal=True,results=results,
        scope='Same weights and scoring, only adapter activation timing differs. Repeated development seeds; no independent generalization claim.')
except Exception as error:
    state('failed',error=repr(error));raise
