"""Experimental passing PPO: frozen shared flow/BFM, trainable latent adapter.

Uses the saved passing environment reward, without an AMP discriminator or
expert actions. Exported deterministic checkpoints load with SoccerPolicy.
Final-iteration selection is explicit; training reward is not an evaluation.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import time
import types
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--pass-config',type=Path,default=Path('/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/params/env.pkl'))
p.add_argument('--training-scene',choices=['pass','shoot'],default='pass',help='Physical scene; shoot retains incoming passing commands/rewards with source shooting physics and failure terms')
p.add_argument('--resume-run',type=Path,help='Resume learning state from a completed PPO run; keep --checkpoint as the original frozen base and match optimizer/task options')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,default=814101)
p.add_argument('--num-envs',type=int,default=256)
p.add_argument('--iterations',type=int,default=100)
p.add_argument('--rollout-steps',type=int,default=32)
p.add_argument('--epochs',type=int,default=4)
p.add_argument('--minibatches',type=int,default=4)
p.add_argument('--learning-rate',type=float,default=1e-5)
p.add_argument('--critic-learning-rate',type=float,default=3e-4)
p.add_argument('--exploration-std',type=float,default=.05)
p.add_argument('--target-kl',type=float,default=.03)
p.add_argument('--reference-reward-scale',type=float,default=1.,help='Scale only the learner contribution of expert_kick_tracking; leave environment reward execution intact')
p.add_argument('--anchor-weight',type=float,default=.01)
p.add_argument('--smoke-test',action='store_true',help='Force short timeouts to verify terminal bootstrapping; not an evaluable candidate')
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
if min(a.num_envs,a.iterations,a.rollout_steps,a.epochs,a.minibatches)<1:p.error('Counts must be positive')
if a.num_envs*a.rollout_steps%a.minibatches:p.error('Rollout batch must divide into minibatches')
if any(not math.isfinite(v) or v<=0 for v in [a.learning_rate,a.critic_learning_rate,a.exploration_std,a.target_kl]):p.error('Rates, std and KL must be finite and positive')
if not math.isfinite(a.anchor_weight) or a.anchor_weight<0:p.error('Anchor weight must be finite and nonnegative')
if a.seed in {811314,812801,812901,812902,812903,812904,812905,813201}:p.error('Reserved evaluation seed')
if not math.isfinite(a.reference_reward_scale) or a.reference_reward_scale<0:p.error('Reference reward scale must be finite and nonnegative')
app=AppLauncher(a).app
try:
    import torch
    import gymnasium as gym
    import whole_body_tracking.tasks
    from fiber_soccer.bridge import Bridge
    from fiber_soccer.paths import ASSETS,ROOT
    from fiber_soccer.planner import load_planner,observation,SCHEMA_V2,SCHEMA_V4
    from fiber_soccer.native_bfm import NativeActor,load_native
    from fiber_soccer.lpp import project
    from fiber_soccer.latent_ppo import (LatentCritic,gaussian_log_prob,mean_kl,
        generalized_advantages,clipped_policy_loss)
    from fiber_soccer.ppo_resume import validate_resume,restore_learning_state
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def status(stage,**details):
        (a.output/'status.json').write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**details),indent=2))
    status('initializing')
    implementation_hashes={}
    for source in [Path(__file__),ROOT/'src/fiber_soccer/latent_ppo.py',ROOT/'src/fiber_soccer/ppo_resume.py',ROOT/'src/fiber_soccer/shared_scene.py']:
        content=source.read_bytes()
        (a.output/source.name).write_bytes(content)
        implementation_hashes[source.name]=hashlib.sha256(content).hexdigest()
    header=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if header.get('planner_arch')!='flow_endpoint_v1' or header['observation_schema']!=SCHEMA_V4:
        raise ValueError('Start from an unadapted v4 flow; existing adapters are not overwritten')
    parent_config=json.loads((a.checkpoint.parent/'config.json').read_text())
    assert (a.checkpoint.parent/'complete.json').exists()
    parent_hash=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    resume_config=resume_complete=None;previous=dict(iterations=0,environment_steps=0);resume_hash=None
    if a.resume_run is not None:
        resume_complete=json.loads((a.resume_run/'complete.json').read_text())
        resume_config=json.loads((a.resume_run/'config.json').read_text())
        previous=validate_resume(resume_config,resume_complete,vars(a),parent_hash)
        resume_hash=hashlib.sha256((a.resume_run/'final.pt').read_bytes()).hexdigest()
        assert resume_hash==resume_complete['checkpoint_sha256']
        assert resume_config['environment_config_sha256']==hashlib.sha256(a.pass_config.read_bytes()).hexdigest()
        assert resume_config['actor_sha256']==hashlib.sha256((ROOT/'AH-BFM/AH-BFM.pt').read_bytes()).hexdigest()
    from fiber_soccer.shared_scene import shared_scene_config,passing_in_shooting_scene,scene_contract
    if a.training_scene=='shoot':
        passing,shooting=shared_scene_config(a.pass_config,a.num_envs,a.device,a.seed)
        cfg=passing_in_shooting_scene(passing,shooting)
        (a.output/'scene_contracts.json').write_text(json.dumps(dict(
            training=scene_contract(cfg),passing=scene_contract(passing),
            shooting=scene_contract(shooting)),indent=2,default=str))
    else:
        with a.pass_config.open('rb') as f:cfg=pickle.load(f)
        for name in ['mpc_ik_reference_based','mpc_ik_use_pelvis_velocity']:
            if name not in vars(cfg):setattr(cfg,name,False)
        import whole_body_tracking.tasks.tracking.mdp.rewards as rewards
        exec(compile((ASSETS/'expert_rewards.py').read_text(),'expert_rewards.py','exec'),rewards.__dict__)
        cfg.rewards.ready_pose.func=rewards.rolling_pass_ready_pose_l2
    if cfg.is_finite_horizon:raise ValueError('This trainer bootstraps external time limits and requires is_finite_horizon=False')
    cfg.commands.motion.debug_vis=False
    cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device;cfg.seed=a.seed
    if a.smoke_test:cfg.episode_length_s=6*cfg.sim.dt*cfg.decimation
    (a.output/'environment.json').write_text(json.dumps(dict(rewards=cfg.rewards.to_dict(),terminations=cfg.terminations.to_dict(),episode_length_s=cfg.episode_length_s,is_finite_horizon=cfg.is_finite_horizon),indent=2,default=str))
    physical_contract=scene_contract(cfg)
    physical_hash=hashlib.sha256(json.dumps(physical_contract,sort_keys=True,default=str).encode()).hexdigest()
    (a.output/'physical_contract.json').write_text(json.dumps(physical_contract,indent=2,default=str))
    if resume_config is not None and 'physical_contract_sha256' in resume_config:
        assert resume_config['physical_contract_sha256']==physical_hash,'Physical training configuration changed'
    env=gym.make('Tracking-Flat-G1-Rolling-Pass-Event-v1-stage2',cfg=cfg)
    core=env.unwrapped;obs,_=env.reset()
    reward_names=list(core.reward_manager.active_terms)
    reference_index=reward_names.index('expert_kick_tracking') if 'expert_kick_tracking' in reward_names else None
    if reference_index is None and a.reference_reward_scale!=1:raise ValueError('Requested reward component is absent')
    reward_multipliers=torch.ones(len(reward_names),device=a.device)
    if reference_index is not None:reward_multipliers[reference_index]=a.reference_reward_scale
    planner=load_planner(header,a.device)
    planner.enable_skill_adapter(0,128,observation_input=True,space='latent')
    anchor=copy.deepcopy(planner).requires_grad_(False)
    actor=NativeActor(load_native(a.device)).eval().requires_grad_(False)
    bridge=Bridge(core,ASSETS)
    critic=LatentCritic(planner).to(a.device)
    params=list(planner.adapter.parameters())
    optimizer=torch.optim.Adam(params,lr=a.learning_rate)
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=a.critic_learning_rate)
    # Isolate exploration and minibatch randomness from simulator randomization.
    generator=torch.Generator(device=a.device).manual_seed(a.seed+100000)
    resume_audit=None
    if a.resume_run is not None:
        resumed=torch.load(a.resume_run/'final.pt',map_location='cpu',weights_only=False)
        required=dict(planner_arch='flow_latent_adapter_v1',adapter_space='latent',adapter_width=128,adapter_skill=0,adapter_observation_input=True,adapter_ball_distance_gate=None,observation_schema=SCHEMA_V4)
        for key,value in required.items():assert resumed.get(key)==value,key
        for key in ['horizon','width','flow_steps','noise_seed','command_conditioning']:assert resumed.get(key)==header.get(key),key
        for key,value in header['planner'].items():assert torch.equal(value,resumed['planner'][key]),key
        learning_state=torch.load(a.resume_run/'trainer.pt',map_location='cpu',weights_only=False)
        if 'checkpoint_sha256' in learning_state:assert learning_state['checkpoint_sha256']==resume_hash
        resume_audit=restore_learning_state(planner,critic,optimizer,critic_optimizer,generator,resumed,learning_state,a.seed+100000+previous['environment_steps'])
        assert all(group['lr']==a.learning_rate for group in optimizer.param_groups)
        assert all(group['lr']==a.critic_learning_rate for group in critic_optimizer.param_groups)
        resume_audit.update(checkpoint_sha256=resume_hash,trainer_sha256=hashlib.sha256((a.resume_run/'trainer.pt').read_bytes()).hexdigest())
        (a.output/'resume_audit.json').write_text(json.dumps(resume_audit,indent=2))
        del resumed,learning_state
    def inputs():
        return observation(core,bridge,'pass',SCHEMA_V4,actor.model,torch.cat((obs['policy'],obs['rma_obs']),-1))
    with torch.no_grad():x=inputs()
    probe=x[:min(len(x),64)].clone();probe[:,-2:]=probe.new_tensor([0.,1.])
    with torch.no_grad():protected_z=anchor.act(probe);protected_action=actor(probe[:,:64],probe[:,64:93],probe[:,93:930],protected_z)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(training='passing_latent_ppo_v1',parent_sha256=parent_hash,implementation_sha256=implementation_hashes,
        gamma=.99,gae_lambda=.95,ppo_clip=.2,max_gradient_norm=1.,critic_observation='instant_113',
        is_finite_horizon=False,timeout_bootstrap='pre_reset_timeout_only',
        data=(resume_config if resume_config is not None else parent_config)['data']+[dict(seed=a.seed,skill='pass',source='online_ppo')],
        resumed_checkpoint_sha256=resume_hash,previous_training=previous,resume_audit=resume_audit,
        checkpoint_selection='final_iteration',expert_actions_used=False,amp_discriminator_used=False,
        reward='saved_environment_total_reward' if a.reference_reward_scale==1 else 'saved_environment_reward_with_reference_component_rescaled',latent_action='Gaussian raw vector then sphere projection',
        trainable_parameters=sum(v.numel() for v in params),frozen_shared_parameters=sum(v.numel() for n,v in planner.named_parameters() if not n.startswith('adapter.')),
        physical_contract_sha256=physical_hash,environment_config_sha256=hashlib.sha256(a.pass_config.read_bytes()).hexdigest(),
        actor_sha256=hashlib.sha256((ROOT/'AH-BFM/AH-BFM.pt').read_bytes()).hexdigest())
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    terminal_values=torch.zeros(a.num_envs,device=a.device)
    terminal_seen=torch.zeros(a.num_envs,dtype=torch.bool,device=a.device)
    executed=None;timeout_count=0
    original_reset=core._reset_idx
    def capture_reset(self,ids):
        timeout=self.reset_time_outs[ids]&~self.reset_terminated[ids]
        timeout_ids=ids[timeout]
        if len(timeout_ids):
            previous=bridge.last
            bridge.last=(executed*6).clamp(-6,6)
            try:
                with torch.no_grad():
                    terminal=observation(core,bridge,'pass',SCHEMA_V2)
                    terminal_values[timeout_ids]=critic(terminal)[timeout_ids]
                    terminal_seen[timeout_ids]=True
            finally:bridge.last=previous
        return original_reset(ids)
    core._reset_idx=types.MethodType(capture_reset,core)
    def export(iteration):
        payload=dict(header)
        payload.update(planner={k:v.detach().cpu() for k,v in planner.state_dict().items()},
            planner_arch='flow_latent_adapter_v1',adapter_space='latent',adapter_width=128,
            adapter_skill=0,adapter_observation_input=True,adapter_ball_distance_gate=None,
            latent_interval=1,latent_execution='hold',updates=previous['iterations']+iteration,config=config)
        torch.save(payload,a.output/'latest.pt')
        return payload
    for iteration in range(1,a.iterations+1):
        store={k:[] for k in ['x','raw','mean','anchor','logp','value','next_value','reward','terminated','truncated','terminal_value']}
        reward_terms=torch.zeros(len(core.reward_manager.active_terms),device=a.device)
        exploration_target_mse=0.
        environment_reward_sum=reference_adjustment_sum=0.
        for step in range(a.rollout_steps):
            with torch.no_grad():
                mean=planner.act(x);anchor_mean=anchor.act(x);value=critic(x)
                raw=mean+a.exploration_std*torch.randn(mean.shape,device=a.device,generator=generator)
                logp=gaussian_log_prob(raw,mean,a.exploration_std)
                z=project(raw)
                executed=actor(x[:,:64],x[:,64:93],x[:,93:930],z)
                targets=bridge.targets(executed)
                deterministic_targets=bridge.targets(actor(x[:,:64],x[:,64:93],x[:,93:930],mean))
                exploration_target_mse+=float((targets-deterministic_targets).square().mean())
                terminal_values.zero_();terminal_seen.zero_()
                obs,reward,terminated,truncated,_=env.step(bridge.env_action(targets))
                environment_reward_sum+=float(reward.mean())
                if a.reference_reward_scale!=1:
                    contribution=core.reward_manager._step_reward[:,reference_index]*core.step_dt
                    adjustment=(1-a.reference_reward_scale)*contribution
                    reward=reward-adjustment
                    reference_adjustment_sum+=float(adjustment.mean())
                timeout=truncated&~terminated
                assert torch.equal(timeout,terminal_seen),'Missing pre-reset timeout bootstrap'
                timeout_count+=int(timeout.sum())
                bridge.advance(executed,terminated|truncated)
                next_x=inputs();next_value=critic(next_x)
                reward_terms+=core.reward_manager._step_reward.mean(0)*core.step_dt*reward_multipliers
            values=(x,raw,mean,anchor_mean,logp,value,next_value,reward,terminated,truncated,terminal_values)
            for key,val in zip(store,values):store[key].append(val.detach().clone())
            x=next_x
        data={k:torch.stack(v) for k,v in store.items()}
        advantages,returns=generalized_advantages(data['reward'],data['value'],data['next_value'],data['terminated'],data['truncated'],data['terminal_value'])
        flat={k:v.flatten(0,1) for k,v in data.items()}
        adv=advantages.flatten();adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
        returns=returns.flatten();total=len(adv);size=total//a.minibatches
        return_variance=returns.var(unbiased=False)
        explained_variance_before=float(1-(returns-flat['value']).var(unbiased=False)/return_variance) if float(return_variance)>1e-8 else None
        updates=0;stop=False;last_policy=last_value=last_anchor=0.
        for epoch in range(a.epochs):
            order=torch.randperm(total,device=a.device,generator=generator)
            for ids in order.split(size):
                new_mean=planner.act(flat['x'][ids])
                kl=mean_kl(flat['mean'][ids],new_mean,a.exploration_std).mean()
                if float(kl.detach())>1.5*a.target_kl:stop=True;break
                loss=clipped_policy_loss(gaussian_log_prob(flat['raw'][ids],new_mean,a.exploration_std),flat['logp'][ids],adv[ids])
                anchoring=mean_kl(flat['anchor'][ids],new_mean,a.exploration_std).mean()
                actor_loss=loss+a.anchor_weight*anchoring
                value_loss=.5*(critic(flat['x'][ids])-returns[ids]).square().mean()
                assert torch.isfinite(actor_loss)&torch.isfinite(value_loss)
                optimizer.zero_grad();actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                critic_optimizer.zero_grad();value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);critic_optimizer.step()
                updates+=1;last_policy=float(loss.detach());last_value=float(value_loss.detach());last_anchor=float(anchoring.detach())
            if stop:break
        with torch.no_grad():
            kl_sum=anchor_sum=0.
            for ids in torch.arange(total,device=a.device).split(size):
                means=planner.act(flat['x'][ids])
                kl_sum+=float(mean_kl(flat['mean'][ids],means,a.exploration_std).sum())
                anchor_sum+=float(mean_kl(flat['anchor'][ids],means,a.exploration_std).sum())
        row=dict(iteration=iteration,total_iteration=previous['iterations']+iteration,environment_steps=iteration*a.rollout_steps*a.num_envs,total_environment_steps=previous['environment_steps']+iteration*a.rollout_steps*a.num_envs,
            environment_reward_mean=environment_reward_sum/a.rollout_steps,
            reference_reward_removed_mean=reference_adjustment_sum/a.rollout_steps,
            exploration_joint_target_mse=exploration_target_mse/a.rollout_steps,
            critic_training_explained_variance_before=explained_variance_before,
            mean_training_reward=float(data['reward'].mean()),policy_loss=last_policy,value_loss=last_value,
            old_policy_kl=kl_sum/total,anchor_kl=anchor_sum/total,gradient_updates=updates,kl_early_stop=stop,
            terminated=int(data['terminated'].sum()),truncated=int(data['truncated'].sum()),
            timeout_bootstraps=timeout_count,reward_terms=dict(zip(core.reward_manager.active_terms,(reward_terms/a.rollout_steps).cpu().tolist())),elapsed=time.monotonic()-started)
        with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('PPO_PROGRESS',json.dumps({k:v for k,v in row.items() if k!='reward_terms'}),flush=True)
        if iteration%10==0 or iteration==a.iterations:payload=export(iteration)
        status('training',iteration=iteration,iterations=a.iterations)
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==parent_hash
    if a.resume_run is not None:
        assert hashlib.sha256((a.resume_run/'final.pt').read_bytes()).hexdigest()==resume_hash
        assert hashlib.sha256((a.resume_run/'trainer.pt').read_bytes()).hexdigest()==resume_audit['trainer_sha256']
    for key,value in header['planner'].items():assert torch.equal(value.cpu(),planner.state_dict()[key].cpu()),key
    with torch.no_grad():
        z=planner.act(probe);action=actor(probe[:,:64],probe[:,64:93],probe[:,93:930],z)
        assert torch.equal(z,protected_z) and torch.equal(action,protected_action)
    assert all(v.grad is None for n,v in planner.named_parameters() if not n.startswith('adapter.'))
    assert all(v.grad is None for v in actor.parameters())
    if a.smoke_test:assert timeout_count>0,'Smoke failed to exercise terminal bootstrapping'
    torch.save(payload,a.output/'final.pt')
    torch.save(dict(critic=critic.state_dict(),optimizer=optimizer.state_dict(),critic_optimizer=critic_optimizer.state_dict(),generator_state=generator.get_state(),checkpoint_sha256=hashlib.sha256((a.output/'final.pt').read_bytes()).hexdigest()),a.output/'trainer.pt')
    from fiber_soccer.planner_policy import SoccerPolicy
    deployed=SoccerPolicy(a.output/'final.pt',a.device)
    with torch.no_grad():assert torch.equal(deployed(x)['latent'],planner.act(x))
    complete=dict(iterations=a.iterations,total_iterations=previous['iterations']+a.iterations,total_environment_steps=previous['environment_steps']+a.iterations*a.rollout_steps*a.num_envs,checkpoint_selection='final_iteration',smoke_test=a.smoke_test,
        shared_tensors_unchanged=True,shoot_probe_outputs_unchanged=True,deployable_roundtrip=True,
        timeout_bootstraps=timeout_count,checkpoint_sha256=hashlib.sha256((a.output/'final.pt').read_bytes()).hexdigest())
    (a.output/'complete.json').write_text(json.dumps(complete,indent=2));status('completed',**complete)
    env.close()
except Exception as error:
    if 'status' in globals():status('failed',error=repr(error))
    raise
finally:
    app.close()
