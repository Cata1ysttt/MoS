"""Task-reward PPO over a cached stateful history/recovery baseline.

Both pass and shoot pre-contact latent heads learn from task rewards.
The new passing expert supplies an auxiliary action loss during training only. Finite-horizon episodes keep
both passing and shooting outcomes in the critic's privileged event memory.
"""
import argparse
import hashlib
import json
import math
import time
import types
import os
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--pass-config',type=Path,default=Path('/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/params/env.pkl'))
p.add_argument('--seed',type=int,default=819101)
p.add_argument('--pass-expert',type=Path,default=Path('/workspace/fiber-soccer/assets/passing_stage2_20261116/model_655000.pt'))
p.add_argument('--expert-weight',type=float,default=.1)
p.add_argument('--num-envs',type=int,default=256);p.add_argument('--iterations',type=int,default=100)
p.add_argument('--rollout-steps',type=int,default=32);p.add_argument('--epochs',type=int,default=4);p.add_argument('--minibatches',type=int,default=4)
p.add_argument('--learning-rate',type=float,default=4e-5);p.add_argument('--critic-learning-rate',type=float,default=3e-4)
p.add_argument('--exploration-std',type=float,default=.2);p.add_argument('--target-kl',type=float,default=.03)
p.add_argument('--anchor-weight',type=float,default=.01)
p.add_argument('--joint-impact-bonus',type=float,default=10.);p.add_argument('--joint-timeout-bonus',type=float,default=20.)
p.add_argument('--smoke-test',action='store_true',help='Eight-second training interface check; never a deployable performance result')
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
if min(a.num_envs,a.iterations,a.rollout_steps,a.epochs,a.minibatches)<1:p.error('Counts must be positive')
if a.num_envs*a.rollout_steps%a.minibatches:p.error('Rollout batch must divide into minibatches')
if any(not math.isfinite(v) or v<=0 for v in [a.learning_rate,a.critic_learning_rate,a.exploration_std,a.target_kl]):p.error('Invalid rates/std/KL')
if any(not math.isfinite(v) or v<0 for v in [a.anchor_weight,a.joint_impact_bonus,a.joint_timeout_bonus]):p.error('Invalid penalty/bonus')
if a.expert_weight<0 or not math.isfinite(a.expert_weight):p.error('Invalid expert weight')
app=AppLauncher(a).app
try:
    import torch
    import gymnasium as gym
    import whole_body_tracking.tasks
    from fiber_soccer.pass_expert_compat import install
    install()
    from fiber_soccer.paths import ASSETS,ROOT
    from fiber_soccer.bridge import Bridge
    from fiber_soccer.planner import observation,SCHEMA_V4
    from fiber_soccer.dual_task_latent import load_dual_task_policy as load_soccer_policy, DualTaskPhase
    from fiber_soccer.lpp import project
    from fiber_soccer.latent_ppo import gaussian_log_prob,mean_kl,generalized_advantages,clipped_policy_loss
    from fiber_soccer.recovery_training import stability_score,normalize_active_advantages
    from fiber_soccer.continuous_shooting import SequenceCritic,sample_shooting_ball,continuous_training_scene
    from fiber_soccer.joint_quality import JointQualityPhase,MemorySequenceCritic
    from fiber_soccer.shared_scene import shared_scene_config,scene_contract
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def status(stage,**details):
        temp=a.output/'status.tmp';temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**details),indent=2)+'\n');temp.replace(a.output/'status.json')
    status('initializing');parent_hash=digest(a.checkpoint)
    header=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert header['dual_task_adapter']['version']==1
    assert header['predicted_target_refinement_scope']=='initial_pass' and header['new_ball_history_mode']=='initialize_shoot_ball_task'
    assert all(not header['dual_task_adapter']['model'][f'heads.{i}.net.2.weight'].any() for i in range(2)), 'Start from zero heads'
    passing,shooting=shared_scene_config(a.pass_config,a.num_envs,a.device,a.seed)
    cfg=continuous_training_scene(passing,shooting,8. if a.smoke_test else 20.)
    contracts={name:scene_contract(value) for name,value in [('training',cfg),('passing',passing),('shooting',shooting)]}
    for field in ['sim_dt','decimation','robot','actions','events','ball_dynamics']:assert contracts['training'][field]==contracts['shooting'][field]
    for name,term in vars(shooting.terminations).items():
        if name!='event_kick_elapsed':assert str(getattr(cfg.terminations,name))==str(term)
    (a.output/'scene_contracts.json').write_text(json.dumps(contracts,indent=2,default=str))
    env=gym.make('Tracking-Flat-G1-Rolling-Pass-Event-v1-stage2',cfg=cfg)
    core=env.unwrapped;obs,_=env.reset();command=core.command_manager.get_term('motion')
    policy=load_soccer_policy(a.checkpoint,a.device);actor=policy.actor;adapter=policy.task_adapter
    frozen={k:v.detach().cpu().clone() for k,v in policy.state_dict().items() if not k.startswith('task_adapter.')}
    adapter.requires_grad_(True)
    params=list(adapter.parameters());assert {id(v) for v in policy.parameters() if v.requires_grad}=={id(v) for v in params}
    with torch.random.fork_rng(devices=[]):critic=MemorySequenceCritic(SequenceCritic(policy.planner,core.max_episode_length).to(a.device))
    optimizer=torch.optim.Adam(params,lr=a.learning_rate);critic_optimizer=torch.optim.Adam(critic.parameters(),lr=a.critic_learning_rate)
    generator=torch.Generator(device=a.device).manual_seed(a.seed+100000)
    import pickle
    from whole_body_tracking.utils.actor_multi_critic import ActorMultiCritic
    with (a.pass_expert.parent/'params/agent.pkl').open('rb') as stream:expert_cfg=pickle.load(stream).to_dict()['policy']
    expert_cfg.pop('class_name')
    expert=ActorMultiCritic(obs['policy'].shape[-1],obs['critic'].shape[-1],obs['rma_obs'].shape[-1],29,**expert_cfg).to(a.device)
    expert.load_state_dict(torch.load(a.pass_expert,map_location=a.device,weights_only=False)['model_state_dict'],strict=True)
    expert.eval().requires_grad_(False)
    phase=DualTaskPhase(a.num_envs,a.device,joint_impact_bonus=a.joint_impact_bonus,joint_timeout_bonus=a.joint_timeout_bonus)
    bridge=Bridge(core,ASSETS)
    foot_ids,_=core.scene['robot'].find_bodies(cfg.terminations.wrong_foot_kick.params['preferred_foot_body_name']);assert len(foot_ids)==1
    def distance():return (core.scene['robot'].data.body_pos_w[:,foot_ids[0],:2]-core.scene['ball'].data.root_pos_w[:,:2]).norm(dim=-1)
    def retained():
        robot=core.scene['robot'].data;values=[robot.root_state_w,robot.joint_pos,robot.joint_vel,bridge.last,bridge.history]
        for group in core.observation_manager._group_obs_term_history_buffer.values():
            for buffer in group.values():values.extend([buffer._buffer,buffer._num_pushes])
        return [v.clone() for v in values if v is not None]
    handoff_checks=baseline_calls=0
    def prepare():
        global handoff_checks,baseline_calls
        ids=phase.ready().nonzero().flatten()
        if len(ids):
            before=retained();sample_shooting_ball(core,shooting,ids)
            assert all(torch.equal(v,w) for v,w in zip(before,retained()))
            handoff_checks+=len(ids);phase.start_shooting(ids,distance())
        phase.advance_recovery()
        passing_history=torch.cat((obs['policy'],obs['rma_obs']),-1)
        shooting_history=torch.nn.functional.pad(torch.cat((obs['shoot_policy'],obs['shoot_rma_obs']),-1),(0,20))
        history=torch.where((phase.stage==1)[:,None],shooting_history,passing_history)
        raw_x=observation(core,bridge,phase.skills(),SCHEMA_V4,actor.model,history)
        output=policy(raw_x);baseline_calls+=1
        x=policy.last_adapter_input.clone();base=policy.last_base_latent.clone();active=policy.last_adapter_active.clone()
        assert torch.equal(active,(phase.stage<2)&~command.ball_kicked)
        recomputed=torch.where(active[:,None],adapter(x,base),base)
        assert torch.equal(output['latent'],recomputed),'Training mean differs from deployed policy'
        return raw_x,x,base,active
    def impacts(eligible,age):
        phase.observe(core.scene['ball'].data.root_lin_vel_w[:,:2],command.ball_target_dir_w[:,:2],
            command.ball_target_speed,command.ball_kicked,command._preferred_foot_kicked_mask(),eligible,age)
    original_reset=core._reset_idx
    def capture_reset(self,ids):
        with torch.no_grad():impacts(torch.ones(a.num_envs,dtype=torch.bool,device=a.device),phase.age+1)
        return original_reset(ids)
    core._reset_idx=types.MethodType(capture_reset,core)
    files=[Path(__file__)]+[ROOT/'src/fiber_soccer'/name for name in ['pass_expert_compat.py','dual_task_latent.py','post_pass_latent_adapter.py','initial_goal_recovery.py',
        'planner_policy.py','predicted_target_solver.py','new_ball_rma.py','new_ball_task_history.py','continuous_shooting.py',
        'latent_ppo.py','joint_quality.py','bridge.py','shared_scene.py','refinement_scope.py','skill_transition.py']]
    hashes={str(path.relative_to(ROOT)):digest(path) for path in files}
    for path in files:(a.output/path.name).write_bytes(path.read_bytes())
    config=dict(header['config']);config.update({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    config.update(training='dual_task_reward_latent_ppo_v1',parent_sha256=parent_hash,implementation_sha256=hashes,
        actor_sha256=digest(ROOT/'AH-BFM/AH-BFM.pt'),environment_config_sha256=digest(a.pass_config),
        gamma=.995,gae_lambda=.97,finite_horizon=True,timeout_bootstrap=False,expert_actions_used=True,expert_actions_deployment=False,expert_sha256=digest(a.pass_expert),expert_auxiliary_weight=a.expert_weight,
        checkpoint_selection='final_iteration',pass_delay=50,shoot_delay=10,trainable_parameters=sum(v.numel() for v in params),
        reward_terms=['approach','impact','recovery','action_rate','failure','accurate_shot_timeout','joint_impact','joint_timeout','pass_approach','pass_impact','pass_stability','pass_failure','pass_smooth'],
        actor_scope='Both pass and shoot pre-contact heads; frozen shared planner, AH-BFM and recovery',
        handoff_sampling='Source shooting distribution with global training RNG; evaluation uses isolated handoff seeds',
        data=header['config']['data']+[dict(seed=a.seed,skill='pass_and_shoot',source='dual_task_reward_ppo')])
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    def export(iteration):
        payload=dict(header,config=config,updates=iteration,dual_task_adapter=dict(version=1,width=adapter.width,
            model={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}))
        torch.save(payload,a.output/'latest.pt');return payload
    phase.previous_distance=distance().clone()
    with torch.no_grad():raw_x,x,base,active=prepare()
    initial_probe=raw_x.clone();torch.save(dict(raw_observation=raw_x.cpu(),conditioned_observation=x.cpu(),base_latent=base.cpu()),a.output/'initial_state.pt')
    active_total=actor_updates_total=critic_updates_total=timeouts=0
    for iteration in range(1,a.iterations+1):
        keys=['x','base','active','stage','age','memory','raw','mean','logp','value','next_value','reward','done','teacher']
        store={key:[] for key in keys};reward_sums=torch.zeros(13,device=a.device)
        for step in range(a.rollout_steps):
            with torch.no_grad():
                teacher=bridge.expert_targets(expert.act_inference(obs['policy'],obs['rma_obs']))
                stage=phase.stage.clone();age=phase.age.clone();memory=phase.memory(core.max_episode_length);phase.begin_step()
                mean=torch.where(active[:,None],adapter(x,base),base)
                assert torch.equal(mean[~active],base[~active])
                value=critic(x,stage,age,memory);active_total+=int(active.sum())
                raw=mean+a.exploration_std*torch.randn(mean.shape,device=a.device,generator=generator)*active[:,None]
                z=torch.where(active[:,None],project(raw),mean);logp=gaussian_log_prob(raw,mean,a.exploration_std)
                executed=actor(x[:,:64],x[:,64:93],x[:,93:930],z)
                obs,_,terminated,truncated,_=env.step(bridge.env_action(bridge.targets(executed)))
                done=terminated|truncated;phase.age+=1;impacts(~done,phase.age)
                robot=core.scene['robot'].data
                stability=stability_score(robot.projected_gravity_b,robot.root_pos_w[:,2]-core.scene.env_origins[:,2],robot.root_lin_vel_w,robot.root_ang_vel_b)
                action_rate=((core.action_manager.action-core.action_manager.prev_action)/10).square().mean(-1)
                terms=phase.rewards(distance(),stability,action_rate,terminated,truncated,core.step_dt)
                reward=terms.sum(-1);assert torch.isfinite(reward).all();reward_sums+=terms.sum(0)
                timeouts+=int((truncated&~terminated).sum())
                phase.reset(done);phase.previous_distance[done]=distance()[done];bridge.advance(executed,done);policy.reset(done)
                assert torch.equal(phase.age,core.episode_length_buf)
                next_raw,next_x,next_base,next_active=prepare()
                next_value=critic(next_x,phase.stage,phase.age,phase.memory(core.max_episode_length))
            for key,val in zip(keys,[x,base,active,stage,age,memory,raw,mean,logp,value,next_value,reward,done,teacher]):store[key].append(val.detach().clone())
            raw_x,x,base,active=next_raw,next_x,next_base,next_active
        data={k:torch.stack(v) for k,v in store.items()}
        advantages,returns=generalized_advantages(data['reward'],data['value'],data['next_value'],data['done'],
            torch.zeros_like(data['done']),torch.zeros_like(data['value']),gamma=.995,lam=.97)
        flat={k:v.flatten(0,1) for k,v in data.items()};returns=returns.flatten()
        adv=advantages.flatten().clone()
        for skill in range(2):
            selected=flat['active']&(flat['x'][:,-2:].argmax(-1)==skill)
            if selected.any():adv[selected]=(adv[selected]-adv[selected].mean())/adv[selected].std(unbiased=False).clamp_min(1e-8)
        size=len(adv)//a.minibatches
        actor_updates=critic_updates=0;stop=False;last_policy=last_value=0.;calls_before=baseline_calls
        for epoch in range(a.epochs):
            for ids in torch.randperm(len(adv),device=a.device,generator=generator).split(size):
                selected=ids[flat['active'][ids]]
                if len(selected) and not stop:
                    means=adapter(flat['x'][selected],flat['base'][selected])
                    kl=mean_kl(flat['mean'][selected],means,a.exploration_std).mean()
                    if float(kl.detach())>1.5*a.target_kl:stop=True
                    else:
                        loss=clipped_policy_loss(gaussian_log_prob(flat['raw'][selected],means,a.exploration_std),flat['logp'][selected],adv[selected])
                        loss+=a.anchor_weight*mean_kl(flat['base'][selected],means,a.exploration_std).mean()
                        teacher_ids=selected[flat['x'][selected,-2]>.5][:32]
                        if len(teacher_ids) and a.expert_weight:
                            tx=flat['x'][teacher_ids];tz=adapter(tx,flat['base'][teacher_ids])
                            predicted=bridge.targets(actor(tx[:,:64],tx[:,64:93],tx[:,93:930],tz))
                            loss+=a.expert_weight*(predicted-flat['teacher'][teacher_ids]).square().mean()
                        assert torch.isfinite(loss);optimizer.zero_grad();loss.backward()
                        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                        actor_updates+=1;last_policy=float(loss.detach())
                value_loss=.5*(critic(flat['x'][ids],flat['stage'][ids],flat['age'][ids],flat['memory'][ids])-returns[ids]).square().mean()
                assert torch.isfinite(value_loss);critic_optimizer.zero_grad();value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);critic_optimizer.step()
                critic_updates+=1;last_value=float(value_loss.detach())
        assert baseline_calls==calls_before,'Minibatch updates advanced stateful history'
        actor_updates_total+=actor_updates;critic_updates_total+=critic_updates
        row=dict(iteration=iteration,environment_steps=iteration*a.rollout_steps*a.num_envs,active_policy_steps=int(flat['active'].sum()),
            total_active_policy_steps=active_total,baseline_calls=baseline_calls,shooting_entries=phase.entries,shots=phase.shots,
            accurate_shots=phase.accurate_shots,passes=phase.passes,accurate_passes=phase.accurate_passes,
            joint_accurate_shots=phase.joint_accurate_shots,joint_successful_timeouts=phase.joint_successful_timeouts,timeouts=timeouts,
            actor_updates=actor_updates,critic_updates=critic_updates,kl_early_stop=stop,policy_loss=last_policy,value_loss=last_value,
            reward_sums=dict(zip(config['reward_terms'],reward_sums.cpu().tolist())),resets=int(data['done'].sum()),elapsed=time.monotonic()-started)
        with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('POST_PASS_PPO',json.dumps(row),flush=True);status('training',**row)
        if iteration%10==0 or iteration==a.iterations:payload=export(iteration)
    assert baseline_calls==1+a.iterations*a.rollout_steps
    assert digest(a.checkpoint)==parent_hash and all(digest(ROOT/path)==h for path,h in hashes.items())
    for key,value in frozen.items():assert torch.equal(value,policy.state_dict()[key].cpu()),key
    assert all(v.grad is None for name,v in policy.named_parameters() if not name.startswith('task_adapter.'))
    assert phase.entries>0 and active_total>0 and actor_updates_total>0
    for skill in range(2):
        assert not torch.equal(adapter.heads[skill].net[-1].weight.cpu(),header['dual_task_adapter']['model'][f'heads.{skill}.net.2.weight']), 'Both heads must learn'
    if a.smoke_test:assert timeouts>0,'Smoke must exercise finite-horizon resets'
    torch.save(payload,a.output/'final.pt')
    torch.save(dict(critic=critic.state_dict(),optimizer=optimizer.state_dict(),critic_optimizer=critic_optimizer.state_dict(),
        generator_state=generator.get_state(),checkpoint_sha256=digest(a.output/'final.pt')),a.output/'trainer.pt')
    deployed=load_soccer_policy(a.output/'final.pt',a.device)
    probe=initial_probe.clone()
    with torch.no_grad():
        for skill in range(2):
            deployed.reset();probe[:,-2:]=0;probe[:,-2+skill]=1;probe[:,940]=0
            actual=deployed(probe);expected=adapter(deployed.last_adapter_input,deployed.last_base_latent)
            assert torch.equal(actual['latent'],expected)
            expected_action=actor(probe[:,:64],probe[:,64:93],probe[:,93:930],expected)
            assert torch.equal(actual['joint_targets'],deployed.default+(6*expected_action).clamp(-6,6)*deployed.scale)
    complete=dict(iterations=a.iterations,environment_steps=a.iterations*a.rollout_steps*a.num_envs,
        active_policy_steps=active_total,actor_updates=actor_updates_total,critic_updates=critic_updates_total,
        baseline_calls=baseline_calls,one_stateful_call_per_observation=True,minibatch_updates_stateless=True,
        shooting_entries=phase.entries,shots=phase.shots,accurate_shots=phase.accurate_shots,
        joint_accurate_shots=phase.joint_accurate_shots,joint_successful_timeouts=phase.joint_successful_timeouts,
        timeouts=timeouts,handoff_history_checks=handoff_checks,shared_tensors_unchanged=True,deployable_roundtrip=True,
        checkpoint_selection='final_iteration',smoke_test=a.smoke_test,checkpoint_sha256=digest(a.output/'final.pt'))
    (a.output/'complete.json').write_text(json.dumps(complete,indent=2)+'\n');status('completed',**complete)
    core._reset_idx=original_reset;env.close()
except Exception as error:
    import traceback,sys
    traceback.print_exc();sys.stderr.flush();sys.stdout.flush()
    if 'status' in globals():status('failed',error=repr(error))
    raise
finally:app.close()
