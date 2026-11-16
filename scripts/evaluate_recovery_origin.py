"""Check native-task preservation, then paired continuous recovery on new seeds."""
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
p.add_argument('--seeds',type=int,nargs='+',default=[815101,815102])
p.add_argument('--num-envs',type=int,default=32)
a=p.parse_args()
assert a.num_envs>0 and len(a.seeds)==len(set(a.seeds))
root=Path(__file__).resolve().parents[1]
run=a.training_run.resolve();tag=run.name+'_shoot_origin'
status=root/'runs'/f'{tag}_status.json'
assert not status.exists(),'Preserve prior evaluation'
results={}
def state(stage,**details):
    status.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),seeds=a.seeds,
        num_envs=a.num_envs,results=results,**details),indent=2))
def container(path):
    return '/workspace/fiber-soccer/'+str(path.relative_to(root))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
try:
    config=json.loads((run/'config.json').read_text())
    complete=json.loads((run/'complete.json').read_text())
    assert config['training_task']=='shoot_recover' and not complete['smoke_test']
    assert complete['precontact_probe_outputs_unchanged']
    checkpoint=run/'final.pt';digest=sha(checkpoint)
    assert digest==complete['checkpoint_sha256']
    state('evaluating_original_tasks',checkpoint_sha256=digest)
    subprocess.run([sys.executable,str(root/'scripts/evaluate_latent_ppo.py'),'--training-run',str(run),
        '--evaluation-tag',tag,'--post-kick-adapter-delay','10','--recovery-adapter-shoot-only'],check=True)
    # Entire episode records, including task quality and termination, must be
    # preserved for both original tasks, not just aggregate success counts.
    for skill in ['pass','shoot']:
        for filename in ['episodes.json','first_kick_velocity.json']:
            current=json.loads((root/'runs'/f'{tag}_{skill}_eval'/filename).read_text())
            baseline=json.loads((root/'runs'/f'planner_history_r3_contact_{skill}_eval'/filename).read_text())
            assert current==baseline,(skill,filename,'native-task preservation failed')
    for seed in a.seeds:
        pair={};protocols=[];contracts=[]
        for label,policy in [('baseline',config['checkpoint']),('candidate',container(checkpoint))]:
            destination=root/'runs'/f'{tag}_{seed}_{label}'
            state('evaluating_sequence',seed=seed,model=label,checkpoint_sha256=digest,native_episode_records_equal=True)
            cmd=[str(root/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/probe_soccer_sequence.py',
                '--headless','--device','cuda:0','--checkpoint',policy,'--output',container(destination),
                '--num-envs',str(a.num_envs),'--steps','1000','--seed',str(seed),'--handoff-seed',str(seed),
                '--scene-base','shoot','--action-contract','shoot','--reset-contract','shoot',
                '--events-contract','shoot','--incoming-pass-command','--continuous-after-kick',
                '--transition-steps','0','--post-shoot-recovery-steps',str(config['recovery_delay_steps']),
                '--post-kick-adapter-delay','10','--recovery-adapter-shoot-only']
            with destination.with_suffix('.log').open('x') as f:
                subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,
                    env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
            summary=json.loads((destination/'summary.json').read_text())
            assert summary['one_policy_instance'] and not summary['expert_loaded']
            assert summary['requested']==a.num_envs and summary['steps']==1000
            assert summary['checkpoint_sha256']==(config['parent_sha256'] if label=='baseline' else digest)
            assert summary['recovery_adapter_shoot_only'] and summary['post_kick_adapter_delay']==10
            pair[label]=dict(summary=summary,accounting=validate(summary))
            protocols.append(json.loads((destination/'protocol.json').read_text()))
            contracts.append(json.loads((destination/'scene_contracts.json').read_text()))
        baseline,candidate=[pair[key]['summary'] for key in ['baseline','candidate']]
        assert baseline['initial_state_sha256']==candidate['initial_state_sha256']
        assert contracts[0]==contracts[1] and protocols[0]==protocols[1]
        switches=baseline['recovery_switches']
        assert switches,'No shooting recovery exercised'
        first=min(row['tick'] for row in switches)
        for key in ['pass_impacts','shoot_impacts']:
            assert {k:v for k,v in baseline[key].items() if v['tick']<=first}=={k:v for k,v in candidate[key].items() if v['tick']<=first},key
        for key in ['switches','first_resets','contact_observations','recovery_switches']:
            assert [v for v in baseline[key] if v['tick']<=first]==[v for v in candidate[key] if v['tick']<=first],key
        pair['audit']=dict(initial_tensors_equal=True,protocol_and_scene_equal=True,
            observed_prefix_equal=True,first_recovery_tick=first)
        results[str(seed)]=pair
        assert sha(checkpoint)==digest
    state('completed',checkpoint_sha256=digest,native_episode_records_equal=True,
        scope='Two preselected new seeds, paired source-physics continuous trials; original-task development episodes also preserved. No automatic model promotion.')
except Exception as error:
    state('failed',error=repr(error));raise
