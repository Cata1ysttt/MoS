"""Experimental task-reward PPO for shooting after an actual frozen passing prefix.

One shared planner/BFM; only the shooting latent adapter is trained. No expert
executes or labels actions. Finite episodes end without timeout bootstrapping.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import time
import types
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--pass-config',type=Path,default=Path('/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/params/env.pkl'))
p.add_argument('--output',type=Path,required=True)
p.add_argument('--resume-run',type=Path,help='Restore planner, critic, optimizers and exploration RNG; start fresh simulator episodes with unchanged objective/options')
p.add_argument('--joint-quality-study',action='store_true',help='Warm-start a completed v1 run and add privileged critic event memory; study joint bonuses explicitly')
p.add_argument('--joint-impact-bonus',type=float,default=0.)
p.add_argument('--joint-timeout-bonus',type=float,default=0.)
p.add_argument('--seed',type=int,default=815401)
p.add_argument('--num-envs',type=int,default=256)
p.add_argument('--iterations',type=int,default=100)
p.add_argument('--rollout-steps',type=int,default=32)
p.add_argument('--epochs',type=int,default=4)
p.add_argument('--minibatches',type=int,default=4)
p.add_argument('--learning-rate',type=float,default=4e-5)
p.add_argument('--critic-learning-rate',type=float,default=3e-4)
p.add_argument('--exploration-std',type=float,default=.2)
p.add_argument('--target-kl',type=float,default=.03)
p.add_argument('--anchor-weight',type=float,default=.01)
p.add_argument('--smoke-test',action='store_true',help='8-second horizon; never promote this checkpoint')
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
if a.joint_quality_study and a.resume_run is None:p.error('Joint study requires a completed v1 source run')
if any(not math.isfinite(v) or v<0 for v in [a.joint_impact_bonus,a.joint_timeout_bonus]):p.error('Invalid joint bonuses')
if not a.joint_quality_study and (a.joint_impact_bonus or a.joint_timeout_bonus):p.error('Joint bonuses require --joint-quality-study')
if min(a.num_envs,a.iterations,a.rollout_steps,a.epochs,a.minibatches)<1:p.error('Counts must be positive')
if a.num_envs*a.rollout_steps%a.minibatches:p.error('Rollout batch must divide into minibatches')
if any(not math.isfinite(v) or v<=0 for v in [a.learning_rate,a.critic_learning_rate,a.exploration_std,a.target_kl]):p.error('Rates/std/KL must be positive and finite')
if not math.isfinite(a.anchor_weight) or a.anchor_weight<0:p.error('Invalid anchor weight')
if a.seed in {811314,812801,812901,812902,812903,812904,812905,813201,813202,815101,815102,815301,815302,815501,815502,815601,815602,815701,815702}:p.error('Reserved evaluation seed')
app=AppLauncher(a).app
try:
    import torch
    import gymnasium as gym
    import whole_body_tracking.tasks
    from fiber_soccer.bridge import Bridge
    from fiber_soccer.paths import ASSETS,ROOT
    from fiber_soccer.planner import observation,SCHEMA_V4
    from fiber_soccer.planner_policy import SoccerPolicy
    from fiber_soccer.lpp import project
    from fiber_soccer.latent_ppo import gaussian_log_prob,mean_kl,generalized_advantages,clipped_policy_loss
    from fiber_soccer.recovery_training import stability_score,normalize_active_advantages
    from fiber_soccer.continuous_shooting import ContinuousShootingPhase,SequenceCritic,sample_shooting_ball,continuous_training_scene
    from fiber_soccer.shared_scene import shared_scene_config,scene_contract
    from fiber_soccer.continuous_resume import validate_continuous_resume
    from fiber_soccer.ppo_resume import restore_learning_state
    from fiber_soccer.joint_quality import JointQualityPhase,MemorySequenceCritic
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def status(stage,**details):
        (a.output/'status.json').write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**details),indent=2))
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    status('initializing')
    header=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if header.get('planner_arch')!='flow_endpoint_v1' or header['observation_schema']!=SCHEMA_V4:
        raise ValueError('Requires unadapted v4 flow checkpoint')
    parent_hash=digest(a.checkpoint)
    previous=dict(iterations=0,environment_steps=0);resume_config=None;resume_audit=None
    resume_hash=resume_trainer_hash=None
    if a.resume_run is not None:
        resume_config=json.loads((a.resume_run/'config.json').read_text())
        resume_complete=json.loads((a.resume_run/'complete.json').read_text())
        previous=validate_continuous_resume(resume_config,resume_complete,vars(a),parent_hash)
        resume_hash=digest(a.resume_run/'final.pt');resume_trainer_hash=digest(a.resume_run/'trainer.pt')
        assert resume_hash==resume_complete['checkpoint_sha256']
        assert resume_config['actor_sha256']==digest(ROOT/'AH-BFM/AH-BFM.pt')
        assert resume_config['environment_config_sha256']==digest(a.pass_config)
        for name in ['continuous_shooting.py','latent_ppo.py','shared_scene.py','flow_planner.py','planner_policy.py']:
            assert resume_config['implementation_sha256'][name]==digest(ROOT/'src/fiber_soccer'/name),'Changed objective/implementation: '+name
    passing,shooting=shared_scene_config(a.pass_config,a.num_envs,a.device,a.seed)
    cfg=continuous_training_scene(passing,shooting,8. if a.smoke_test else 20.)
    contracts={name:scene_contract(value) for name,value in [('training',cfg),('passing',passing),('shooting',shooting)]}
    for field in ['sim_dt','decimation','robot','actions','events','ball_dynamics']:
        assert contracts['training'][field]==contracts['shooting'][field],field
    for name,term in vars(shooting.terminations).items():
        if name!='event_kick_elapsed':assert str(getattr(cfg.terminations,name))==str(term),name
    (a.output/'scene_contracts.json').write_text(json.dumps(contracts,indent=2,default=str))
    if a.resume_run is not None:
        assert json.loads(json.dumps(contracts,default=str))==json.loads((a.resume_run/'scene_contracts.json').read_text()),'Continuation scene changed'
    env=gym.make('Tracking-Flat-G1-Rolling-Pass-Event-v1-stage2',cfg=cfg)
    core=env.unwrapped;obs,_=env.reset();command=core.command_manager.get_term('motion')
    policy=SoccerPolicy(a.checkpoint,a.device);planner=policy.planner;actor=policy.actor
    anchor=copy.deepcopy(planner).requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        planner.enable_skill_adapter(1,128,observation_input=True,space='latent')
        critic=SequenceCritic(planner,core.max_episode_length).to(a.device)
    params=list(planner.adapter.parameters())
    assert all(v.requires_grad for v in params)
    optimizer=torch.optim.Adam(params,lr=a.learning_rate)
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=a.critic_learning_rate)
    generator=torch.Generator(device=a.device).manual_seed(a.seed+100000)
    if a.resume_run is not None:
        resumed=torch.load(a.resume_run/'final.pt',map_location='cpu',weights_only=False)
        state=torch.load(a.resume_run/'trainer.pt',map_location='cpu',weights_only=False)
        assert resumed['shoot_adapter_after_pass_only'] and resumed['adapter_skill']==1
        if 'checkpoint_sha256' in state:assert state['checkpoint_sha256']==resume_hash
        assert 'generator_state' in state,'Continuation requires saved exploration RNG'
        resume_audit=restore_learning_state(planner,critic,optimizer,critic_optimizer,generator,resumed,state,a.seed+100000)
        for key,value in header['planner'].items():assert torch.equal(value.cpu(),planner.state_dict()[key].cpu()),key
        assert all(group['lr']==a.learning_rate for group in optimizer.param_groups)
        assert all(group['lr']==a.critic_learning_rate for group in critic_optimizer.param_groups)
        resume_audit.update(checkpoint_sha256=resume_hash,trainer_sha256=resume_trainer_hash,previous_training=previous)
        (a.output/'resume_audit.json').write_text(json.dumps(resume_audit,indent=2))
        del resumed,state
    if a.joint_quality_study:
        critic=MemorySequenceCritic(critic)
        critic_optimizer.add_param_group({'params':critic.memory_projection.parameters()})
        resume_audit.update(critic_memory_zero_initialized=True,critic_memory_parameters=sum(v.numel() for v in critic.memory_projection.parameters()),
            joint_quality_study=True,joint_impact_bonus=a.joint_impact_bonus,joint_timeout_bonus=a.joint_timeout_bonus)
        (a.output/'resume_audit.json').write_text(json.dumps(resume_audit,indent=2))
    def critic_value(x,stage,age,memory):
        return critic(x,stage,age,memory) if a.joint_quality_study else critic(x,stage,age)
    bridge=Bridge(core,ASSETS)
    phase=JointQualityPhase(a.num_envs,a.device,joint_impact_bonus=a.joint_impact_bonus,joint_timeout_bonus=a.joint_timeout_bonus) if a.joint_quality_study else ContinuousShootingPhase(a.num_envs,a.device)
    def event_memory():
        return phase.memory(core.max_episode_length) if a.joint_quality_study else torch.empty((a.num_envs,0),device=a.device)
    def joint_metrics():
        return dict(passes=phase.passes,accurate_passes=phase.accurate_passes,joint_accurate_shots=phase.joint_accurate_shots,
            joint_successful_timeouts=phase.joint_successful_timeouts) if a.joint_quality_study else {}

    foot_name=cfg.terminations.wrong_foot_kick.params['preferred_foot_body_name']
    foot_ids,_=core.scene['robot'].find_bodies(foot_name);assert len(foot_ids)==1
    def distance():
        return (core.scene['robot'].data.body_pos_w[:,foot_ids[0],:2]-core.scene['ball'].data.root_pos_w[:,:2]).norm(dim=-1)
    def retained():
        robot=core.scene['robot'].data
        values=[robot.root_state_w,robot.joint_pos,robot.joint_vel,bridge.last,bridge.history]
        for group in core.observation_manager._group_obs_term_history_buffer.values():
            for buffer in group.values():values.extend([buffer._buffer,buffer._num_pushes])
        return [v.clone() for v in values if v is not None]
    handoff_checks=0
    def prepare():
        global handoff_checks
        ids=phase.ready().nonzero().flatten()
        if len(ids):
            before=retained();sample_shooting_ball(core,shooting,ids)
            assert all(torch.equal(v,w) for v,w in zip(before,retained())),'Handoff changed robot/history'
            handoff_checks+=len(ids);phase.start_shooting(ids,distance())
        phase.advance_recovery()
        passing_history=torch.cat((obs['policy'],obs['rma_obs']),-1)
        shooting_history=torch.nn.functional.pad(torch.cat((obs['shoot_policy'],obs['shoot_rma_obs']),-1),(0,20))
        history=torch.where((phase.stage==1)[:,None],shooting_history,passing_history)
        return observation(core,bridge,phase.skills(),SCHEMA_V4,actor.model,history)
    def impacts(eligible,age):
        phase.observe(core.scene['ball'].data.root_lin_vel_w[:,:2],command.ball_target_dir_w[:,:2],
            command.ball_target_speed,command.ball_kicked,command._preferred_foot_kicked_mask(),eligible,age)
    original_reset=core._reset_idx
    pre_reset_events=0
    def capture_reset(self,ids):
        global pre_reset_events
        before=phase.shots
        with torch.no_grad():impacts(torch.ones(a.num_envs,dtype=torch.bool,device=a.device),phase.age+1)
        pre_reset_events+=phase.shots-before
        return original_reset(ids)
    core._reset_idx=types.MethodType(capture_reset,core)
    hashes={}
    for path in [Path(__file__),ROOT/'src/fiber_soccer/continuous_shooting.py',ROOT/'src/fiber_soccer/latent_ppo.py',ROOT/'src/fiber_soccer/shared_scene.py',ROOT/'src/fiber_soccer/flow_planner.py',ROOT/'src/fiber_soccer/planner_policy.py',ROOT/'src/fiber_soccer/continuous_resume.py',ROOT/'src/fiber_soccer/ppo_resume.py',ROOT/'src/fiber_soccer/joint_quality.py']:
        (a.output/path.name).write_bytes(path.read_bytes());hashes[path.name]=digest(path)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(training='continuous_joint_quality_ppo_v1' if a.joint_quality_study else 'continuous_post_pass_shooting_ppo_v1',parent_sha256=parent_hash,
        implementation_sha256=hashes,previous_training=previous,resume_audit=resume_audit,resumed_checkpoint_sha256=resume_hash,gamma=.995,gae_lambda=.97,finite_horizon=True,timeout_bootstrap=False,
        expert_actions_used=False,checkpoint_selection='final_iteration',pass_delay=50,shoot_delay=10,
        trainable_parameters=sum(v.numel() for v in params),actor_sha256=digest(ROOT/'AH-BFM/AH-BFM.pt'),
        environment_config_sha256=digest(a.pass_config),reward_terms=['approach','impact','recovery','action_rate','failure','accurate_shot_timeout']+(['joint_impact','joint_timeout'] if a.joint_quality_study else []),
        critic_memory='pass_shot_event_10_zero_initialized_branch' if a.joint_quality_study else None,
        reward_scope='Original six shooting terms plus explicit joint impact/timeout terms; only joint_successful_timeouts count both skills' if a.joint_quality_study else 'shooting after pass; timeout counts are NOT full two-skill successes',
        data=(resume_config if resume_config is not None else json.loads((a.checkpoint.parent/'config.json').read_text()))['data']+[dict(seed=a.seed,skill='shoot',source='continuous_task_ppo',start_iteration=previous['iterations'])])
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    def export(iteration):
        payload=dict(header)
        payload.update(planner={k:v.detach().cpu() for k,v in planner.state_dict().items()},
            planner_arch='flow_latent_adapter_v1',adapter_space='latent',adapter_width=128,adapter_skill=1,
            adapter_observation_input=True,adapter_ball_distance_gate=None,adapter_post_kick_only=False,
            shoot_adapter_after_pass_only=True,latent_interval=1,latent_execution='hold',updates=previous['iterations']+iteration,config=config)
        torch.save(payload,a.output/'latest.pt');return payload
    with torch.no_grad():x=prepare()
    initial_probe=x.clone()
    if a.joint_quality_study:
        with torch.no_grad():
            assert not critic.memory_projection.weight.any()
            old_value=critic.base(x,phase.stage,phase.age)
            assert torch.equal(old_value,critic_value(x,phase.stage,phase.age,event_memory()))
            assert torch.equal(old_value,critic_value(x,phase.stage,phase.age,torch.ones((a.num_envs,10),device=a.device)))
        robot=core.scene['robot'].data
        initial=dict(observation=x.cpu(),robot_root=robot.root_state_w.cpu(),joint_pos=robot.joint_pos.cpu(),
            joint_vel=robot.joint_vel.cpu(),ball_root=core.scene['ball'].data.root_state_w.cpu(),
            shooting_history=torch.cat((obs['shoot_policy'],obs['shoot_rma_obs']),-1).cpu())
        torch.save(initial,a.output/'initial_state.pt')
        initialization=dict(initial_state_sha256=hashlib.sha256(b''.join(k.encode()+v.numpy().tobytes() for k,v in sorted(initial.items()))).hexdigest(),
            cpu_rng_sha256=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
            cuda_rng_sha256=hashlib.sha256(torch.cuda.get_rng_state(a.device).cpu().numpy().tobytes()).hexdigest() if torch.device(a.device).type=='cuda' else None,
            exploration_rng_sha256=hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest(),
            original_critic_values_exact=True,nonzero_memory_initial_values_exact=True,source_checkpoint_sha256=resume_hash)
        (a.output/'initialization.json').write_text(json.dumps(initialization,indent=2))
    active_total=actor_updates_total=critic_updates_total=timeouts=0
    for iteration in range(1,a.iterations+1):
        keys=['x','stage','age','memory','raw','mean','anchor','logp','value','next_value','reward','done','active']
        store={key:[] for key in keys};reward_sums=torch.zeros(len(config['reward_terms']),device=a.device);environment_sum=0.
        for step in range(a.rollout_steps):
            with torch.no_grad():
                stage=phase.stage.clone();age=phase.age.clone();memory=event_memory();active=stage==1
                active_total+=int(active.sum());phase.begin_step()
                mean=planner.act(x);anchor_mean=anchor.act(x);value=critic_value(x,stage,age,memory)
                assert torch.equal(mean[~active],anchor_mean[~active]),'Frozen pass/recovery changed'
                raw=mean+a.exploration_std*torch.randn(mean.shape,device=a.device,generator=generator)*active[:,None]
                z=torch.where(active[:,None],project(raw),mean);logp=gaussian_log_prob(raw,mean,a.exploration_std)
                executed=actor(x[:,:64],x[:,64:93],x[:,93:930],z)
                obs,env_reward,terminated,truncated,_=env.step(bridge.env_action(bridge.targets(executed)))
                done=terminated|truncated;phase.age+=1;impacts(~done,phase.age)
                robot=core.scene['robot'].data
                stability=stability_score(robot.projected_gravity_b,robot.root_pos_w[:,2]-core.scene.env_origins[:,2],robot.root_lin_vel_w,robot.root_ang_vel_b)
                action_rate=((core.action_manager.action-core.action_manager.prev_action)/10).square().mean(-1)
                terms=phase.rewards(distance(),stability,action_rate,terminated,truncated,core.step_dt)
                reward=terms.sum(-1);assert torch.isfinite(reward).all()
                reward_sums+=terms.sum(0);environment_sum+=float(env_reward.mean());timeouts+=int((truncated&~terminated).sum())
                phase.reset(done);bridge.advance(executed,done)
                assert torch.equal(phase.age,core.episode_length_buf),'Phase clock differs from simulator'
                next_x=prepare();next_value=critic_value(next_x,phase.stage,phase.age,event_memory())
            for key,val in zip(keys,[x,stage,age,memory,raw,mean,anchor_mean,logp,value,next_value,reward,done,active]):store[key].append(val.detach().clone())
            x=next_x
        data={k:torch.stack(v) for k,v in store.items()}
        advantages,returns=generalized_advantages(data['reward'],data['value'],data['next_value'],data['done'],torch.zeros_like(data['done']),torch.zeros_like(data['value']),gamma=.995,lam=.97)
        flat={k:v.flatten(0,1) for k,v in data.items()};returns=returns.flatten()
        adv=normalize_active_advantages(advantages.flatten(),flat['active']);size=len(adv)//a.minibatches
        actor_updates=critic_updates=0;stop=False;last_policy=last_value=0.
        for epoch in range(a.epochs):
            for ids in torch.randperm(len(adv),device=a.device,generator=generator).split(size):
                selected=ids[flat['active'][ids]]
                if len(selected) and not stop:
                    means=planner.act(flat['x'][selected]);kl=mean_kl(flat['mean'][selected],means,a.exploration_std).mean()
                    if float(kl.detach())>1.5*a.target_kl:stop=True
                    else:
                        loss=clipped_policy_loss(gaussian_log_prob(flat['raw'][selected],means,a.exploration_std),flat['logp'][selected],adv[selected])
                        loss=loss+a.anchor_weight*mean_kl(flat['anchor'][selected],means,a.exploration_std).mean()
                        assert torch.isfinite(loss)
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                        actor_updates+=1;last_policy=float(loss.detach())
                value_loss=.5*(critic_value(flat['x'][ids],flat['stage'][ids],flat['age'][ids],flat['memory'][ids])-returns[ids]).square().mean()
                assert torch.isfinite(value_loss)
                critic_optimizer.zero_grad();value_loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);critic_optimizer.step()
                critic_updates+=1;last_value=float(value_loss.detach())
        actor_updates_total+=actor_updates;critic_updates_total+=critic_updates
        row=dict(**joint_metrics(),iteration=iteration,total_iteration=previous['iterations']+iteration,total_environment_steps=previous['environment_steps']+iteration*a.rollout_steps*a.num_envs,environment_steps=iteration*a.rollout_steps*a.num_envs,active_policy_steps=int(flat['active'].sum()),
            total_active_policy_steps=active_total,shooting_entries=phase.entries,shots=phase.shots,accurate_shots=phase.accurate_shots,
            shot_accurate_timeouts=phase.successful_timeouts,timeouts=timeouts,pre_reset_shot_events=pre_reset_events,
            actor_updates=actor_updates,critic_updates=critic_updates,kl_early_stop=stop,policy_loss=last_policy,value_loss=last_value,
            reward_sums=dict(zip(config['reward_terms'],reward_sums.cpu().tolist())),environment_reward_mean=environment_sum/a.rollout_steps,
            resets=int(data['done'].sum()),elapsed=time.monotonic()-started)
        with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('CONTINUOUS_PPO',json.dumps(row),flush=True);status('training',**row)
        if iteration%10==0 or iteration==a.iterations:payload=export(iteration)
    assert digest(a.checkpoint)==parent_hash
    if a.resume_run is not None:
        assert digest(a.resume_run/'final.pt')==resume_hash and digest(a.resume_run/'trainer.pt')==resume_trainer_hash
    for key,value in header['planner'].items():assert torch.equal(value.cpu(),planner.state_dict()[key].cpu()),key
    assert all(v.grad is None for n,v in planner.named_parameters() if not n.startswith('adapter.'))
    assert all(v.grad is None for v in actor.parameters())
    assert phase.entries>0 and active_total>0 and actor_updates_total>0,'No shooting learning exercised'
    if a.smoke_test:assert timeouts>0,'Smoke did not exercise finite-horizon timeouts'
    torch.save(payload,a.output/'final.pt')
    torch.save(dict(critic=critic.state_dict(),optimizer=optimizer.state_dict(),critic_optimizer=critic_optimizer.state_dict(),generator_state=generator.get_state(),checkpoint_sha256=digest(a.output/'final.pt')),a.output/'trainer.pt')
    deployed=SoccerPolicy(a.output/'final.pt',a.device)
    with torch.no_grad():
        # Routing checks on actual observed states; synthetic commands only arm the context gate.
        probe=torch.cat((initial_probe,x));probe[:,-2:]=probe.new_tensor([0.,1.]);probe[:,940]=0
        assert torch.equal(deployed(probe)['latent'],anchor.act(probe)),'Standalone shooting changed'
        deployed.reset();prefix=probe.clone();prefix[:,-2:]=prefix.new_tensor([1.,0.]);prefix[:,940]=1
        assert torch.equal(deployed(prefix)['latent'],anchor.act(prefix)),'Passing changed'
        result=deployed(probe);expected=planner.act(probe)
        assert torch.equal(result['latent'],expected),'Post-pass export mismatch'
        action=actor(probe[:,:64],probe[:,64:93],probe[:,93:930],expected)
        assert torch.equal(result['joint_targets'],deployed.default+(6*action).clamp(-6,6)*deployed.scale)
    complete=dict(**joint_metrics(),iterations=a.iterations,total_iterations=previous['iterations']+a.iterations,total_environment_steps=previous['environment_steps']+a.iterations*a.rollout_steps*a.num_envs,resume_audit=resume_audit,environment_steps=a.iterations*a.rollout_steps*a.num_envs,
        active_policy_steps=active_total,actor_updates=actor_updates_total,critic_updates=critic_updates_total,
        shooting_entries=phase.entries,shots=phase.shots,accurate_shots=phase.accurate_shots,shot_accurate_timeouts=phase.successful_timeouts,
        timeouts=timeouts,handoff_history_checks=handoff_checks,shared_tensors_unchanged=True,deployable_roundtrip=True,
        checkpoint_selection='final_iteration',smoke_test=a.smoke_test,checkpoint_sha256=digest(a.output/'final.pt'))
    (a.output/'complete.json').write_text(json.dumps(complete,indent=2));status('completed',**complete)
    env.close()
except Exception as error:
    if 'status' in globals():status('failed',error=repr(error))
    raise
finally:app.close()
