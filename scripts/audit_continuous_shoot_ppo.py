"""Audit continuous PPO scene fidelity and full exported policy on real held-out states."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.planner import with_task_history,SCHEMA_V3,SCHEMA_V4
from fiber_soccer.planner_policy import SoccerPolicy
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--sequence-reference',type=Path,required=True)
p.add_argument('--data',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cuda:0')
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
complete=json.loads((a.run/'complete.json').read_text());config=json.loads((a.run/'config.json').read_text())
contracts=json.loads((a.run/'scene_contracts.json').read_text())
train,shoot,passing=[contracts[k] for k in ['training','shooting','passing']]
reference=json.loads((a.sequence_reference/'scene_contracts.json').read_text())['shared']
for key in ['sim_dt','decimation','robot','actions','events','ball_dynamics']:
    assert train[key]==shoot[key]==reference[key],key
for key,ignored in [('scene',{'num_envs'}),('sim',{'device'})]:
    cleaned=[{k:v for k,v in c[key].items() if k not in ignored} for c in [train,shoot,reference]]
    assert cleaned[0]==cleaned[1]==cleaned[2],key
assert train['terminations']==reference['terminations']
for key,value in shoot['terminations'].items():
    if key!='event_kick_elapsed':assert value==train['terminations'][key],key
assert train['rewards']==shoot['rewards'],'Original reward execution changed'
assert train['command']==reference['command'],'Ball/reset command mismatch'
assert train['episode_length_s']==(8. if config['smoke_test'] else 20.)
assert config['finite_horizon'] and not config['timeout_bootstrap']
assert complete['shooting_entries']==complete['handoff_history_checks']>0
assert complete['active_policy_steps']>0 and complete['actor_updates']>0
metrics=[json.loads(line) for line in (a.run/'metrics.jsonl').read_text().splitlines()]
assert len(metrics)==complete['iterations']
resume_step_audit=None
if config.get('resume_run'):
    source=Path(config['resume_run'])
    old_state=torch.load(source/'trainer.pt',map_location='cpu',weights_only=False)
    new_state=torch.load(a.run/'trainer.pt',map_location='cpu',weights_only=False)
    resume=config['resume_audit']
    assert resume==complete['resume_audit']==json.loads((a.run/'resume_audit.json').read_text())
    assert resume['checkpoint_sha256']==hashlib.sha256((source/'final.pt').read_bytes()).hexdigest()
    assert resume['trainer_sha256']==hashlib.sha256((source/'trainer.pt').read_bytes()).hexdigest()
    assert resume['exploration_rng_restored'] and not resume['simulator_restored']
    for field,count in [('optimizer',complete['actor_updates']),('critic_optimizer',complete['critic_updates'])]:
        old,new=old_state[field]['state'],new_state[field]['state']
        assert old.keys()==new.keys() and len(old)>0
        for key in old:assert int(new[key]['step'])==int(old[key]['step'])+count,(field,key)
    previous=resume['previous_training']
    assert complete['total_iterations']==previous['iterations']+complete['iterations']
    assert complete['total_environment_steps']==previous['environment_steps']+complete['environment_steps']
    resume_step_audit=dict(optimizer_steps_continue_exactly=True,previous_training=previous,
        total_iterations=complete['total_iterations'],total_environment_steps=complete['total_environment_steps'])
    del old_state,new_state
for row in metrics:
    if row['active_policy_steps']==0:assert row['actor_updates']==0
assert sum(row['active_policy_steps'] for row in metrics)==complete['active_policy_steps']
assert sum(row['actor_updates'] for row in metrics)==complete['actor_updates']
assert sum(row['critic_updates'] for row in metrics)==complete['critic_updates']
assert metrics[-1]['shots']==complete['shots'] and metrics[-1]['timeouts']==complete['timeouts']
assert complete['shot_accurate_timeouts']<=complete['accurate_shots']<=complete['shots']<=complete['shooting_entries']
reward_totals={name:sum(row['reward_sums'][name] for row in metrics) for name in config['reward_terms']}
assert abs(reward_totals['accurate_shot_timeout']-5*complete['shot_accurate_timeouts'])<1e-5
assert 0<=reward_totals['impact']<=17*complete['shots']+1e-3
assert reward_totals['action_rate']<=0 and reward_totals['recovery']>=0
handoff_failures=round(-reward_totals['failure']/10)
assert abs(reward_totals['failure']+10*handoff_failures)<1e-5
assert 0<=handoff_failures<=sum(row['resets'] for row in metrics)-complete['timeouts']
inputs=[]
for path in a.data:
    d=torch.load(path,map_location='cpu',weights_only=True);rows=d['rows'];start=len(rows[0]['x'])*3//4
    for tick in torch.linspace(0,len(rows)-1,min(8,len(rows))).long().tolist():
        row=rows[tick];x=row['x'][start:start+8]
        if d['metadata']['schema']==SCHEMA_V3:x=with_task_history(x,row['expert_observation'][start:start+8])
        else:assert d['metadata']['schema']==SCHEMA_V4
        inputs.append(x)
x=torch.cat(inputs).to(a.device);assert set(x[:,-2:].argmax(-1).tolist())=={0,1}
rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state(a.device) if torch.device(a.device).type=='cuda' else None
base=SoccerPolicy(a.base,a.device)
after_base=torch.random.get_rng_state();after_base_cuda=torch.cuda.get_rng_state(a.device) if cuda_rng is not None else None
torch.random.set_rng_state(rng)
if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,a.device)
candidate=SoccerPolicy(a.run/'final.pt',a.device)
assert torch.equal(after_base,torch.random.get_rng_state()),'Candidate changes RNG consumption relative to base loading'
if cuda_rng is not None:assert torch.equal(after_base_cuda,torch.cuda.get_rng_state(a.device))
assert candidate.shoot_adapter_after_pass_only
for key,value in base.planner.state_dict().items():assert torch.equal(value,candidate.planner.state_dict()[key]),key
keys=['latent','action_normalized','joint_targets']
with torch.no_grad():
    # Clear contact across all slots to test standalone execution without arming context.
    standalone=x.clone();standalone[:,940]=0
    old=base(standalone);new=candidate(standalone)
    for key in keys:assert torch.equal(old[key],new[key]),key
    candidate.reset()
    # Original passing states (including real contacts) are always protected.
    passing=x[x[:,-2:].argmax(-1)==0]
    old=base(passing);new=candidate(passing)
    for key in keys:assert torch.equal(old[key],new[key]),key
    candidate.reset()
    contact=x.clone();contact[:,-2:]=contact.new_tensor([1.,0.]);contact[:,940]=1
    candidate(contact)
    deployed=candidate(x);latent=candidate.planner.act(x)
    action=candidate.actor(x[:,:64],x[:,64:93],x[:,93:930],latent)
    assert torch.equal(deployed['latent'],latent)
    assert torch.equal(deployed['action_normalized'],action)
    assert torch.equal(deployed['joint_targets'],candidate.default+(6*action).clamp(-6,6)*candidate.scale)
    original=base(x);shoot_mask=x[:,-2:].argmax(-1)==1
    latent_rms=float((deployed['latent'][shoot_mask]-original['latent'][shoot_mask]).square().mean().sqrt())
    target_mse=float((deployed['joint_targets'][shoot_mask]-original['joint_targets'][shoot_mask]).square().mean())
    assert latent_rms>0,'Training produced no sampled shooting latent change'
result=dict(resume_step_audit=resume_step_audit,post_pass_shoot_latent_rms_change=latent_rms,post_pass_shoot_joint_target_mse=target_mse,training_reward_accounting=dict(reward_totals=reward_totals,handoff_failures=handoff_failures,shot_timeout_bonuses_exact=True),scene_matches_continuous_evaluation=True,source_failure_terms_preserved=True,
    original_reward_execution_preserved=True,finite_horizon=True,states=len(x),passing_states=len(passing),
    state_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),shared_tensors_exact=True,
    standalone_outputs_match_base=True,passing_outputs_match_base=True,post_pass_outputs_match_trained_core=True,
    loading_rng_matches_base=True,checkpoint_sha256=hashlib.sha256((a.run/'final.pt').read_bytes()).hexdigest(),
    scope='Scene contracts and full policy routing on held-out states; no performance claim.')
assert result['checkpoint_sha256']==complete['checkpoint_sha256']
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
