"""Independently audit exported adapter weights, optimizer steps and training scene."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(source,run,sequence):
    config=read(run/'config.json');done=read(run/'complete.json');assert read(run/'status.json')['stage']=='completed'
    assert config['training']=='history_recovery_post_pass_latent_ppo_v1'
    assert sha(source)==config['parent_sha256'] and sha(run/'final.pt')==done['checkpoint_sha256']
    original=torch.load(source,map_location='cpu',weights_only=False)
    trained=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
    for key in original['planner']:assert torch.equal(original['planner'][key],trained['planner'][key]),key
    for key in original:
        if key not in ('planner','config','updates','post_pass_latent_adapter'):assert original[key]==trained[key],key
    a=original['post_pass_latent_adapter'];b=trained['post_pass_latent_adapter']
    assert a['width']==b['width'] and a['version']==b['version']==1
    changed=[]
    for key,value in a['model'].items():
        assert torch.isfinite(b['model'][key]).all()
        if not torch.equal(value,b['model'][key]):changed.append(key)
    assert changed and all(key.startswith('net.') for key in changed)
    metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    assert [v['iteration'] for v in metrics]==list(range(1,done['iterations']+1))
    assert done['iterations']==config['iterations']
    for row in metrics:
        assert row['baseline_calls']==1+row['iteration']*config['rollout_steps']
        assert row['environment_steps']==row['iteration']*config['rollout_steps']*config['num_envs']
        if row['active_policy_steps']==0:assert row['actor_updates']==0
    for key in ('actor_updates','critic_updates'):assert sum(v[key] for v in metrics)==done[key]
    assert sum(v['active_policy_steps'] for v in metrics)==done['active_policy_steps']
    assert done['baseline_calls']==1+config['iterations']*config['rollout_steps']
    trainer=torch.load(run/'trainer.pt',map_location='cpu',weights_only=False)
    assert trainer['checkpoint_sha256']==done['checkpoint_sha256']
    steps={}
    for key,total in [('optimizer',done['actor_updates']),('critic_optimizer',done['critic_updates'])]:
        values=[int(v['step']) for v in trainer[key]['state'].values()]
        assert values and all(v==total for v in values)
        steps[key]=dict(parameters=len(values),steps_per_parameter=total)
    scenes=read(run/'scene_contracts.json');train=scenes['training'];shoot=scenes['shooting']
    reference=read(sequence/'scene_contracts.json')['shared']
    for key in ('sim_dt','decimation','robot','actions','events','ball_dynamics'):assert train[key]==shoot[key]==reference[key],key
    for key,ignored in [('scene',{'num_envs'}),('sim',{'device'})]:
        values=[{k:v for k,v in c[key].items() if k not in ignored} for c in [train,shoot,reference]]
        assert values[0]==values[1]==values[2],key
    assert train['terminations']['event_kick_elapsed'] is None
    for key,value in shoot['terminations'].items():
        if key!='event_kick_elapsed':assert value==train['terminations'][key]==reference['terminations'][key],key
    assert train['command']==reference['command'],'Training command/reset differs from continuous evaluation'
    assert train['episode_length_s']==(8. if done['smoke_test'] else 20.)
    root=Path(__file__).resolve().parents[1]
    assert sha(root/'AH-BFM/AH-BFM.pt')==config['actor_sha256']
    assert all(sha(root/path)==h for path,h in config['implementation_sha256'].items())
    return dict(checkpoint_sha256=done['checkpoint_sha256'],parent_sha256=config['parent_sha256'],
        only_adapter_network_weights_changed=True,changed_adapter_tensors=changed,optimizer_steps=steps,
        metrics_and_stateful_call_counts_consistent=True,source_and_sequence_physics_match=True,failure_terms_preserved=True,
        command_and_reset_match_sequence=True,horizon_seconds=train['episode_length_s'],smoke_test=done['smoke_test'],
        actor_updates=done['actor_updates'],active_policy_steps=done['active_policy_steps'],
        joint_successful_training_timeouts=done['joint_successful_timeouts'],
        scope='Training/export audit only. Training outcomes include exploration and are not independent evaluation success rates.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    result=audit(a.source,a.run,a.sequence);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
