"""Pass then static-ball shoot with one policy and no robot/history resets.

This is a shared passing-scene prototype. It retains that scene's dynamics and
termination rules; it is not identical to two separate original expert tasks.
"""
import argparse
import copy
import hashlib
import json
import types
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--pass-config',type=Path,default=Path('/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/params/env.pkl'))
p.add_argument('--num-envs',type=int,default=16)
p.add_argument('--steps',type=int,default=1000)
p.add_argument('--seed',type=int,default=813201)
p.add_argument('--recovery-steps',type=int,default=50)
p.add_argument('--transition-steps',type=int,default=10)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
assert a.steps>0 and a.recovery_steps>=1
app=AppLauncher(a).app
try:
    import torch
    import gymnasium as gym
    import whole_body_tracking.tasks
    from fiber_soccer.shared_scene import shared_scene_config
    from fiber_soccer.planner_policy import SoccerPolicy
    from fiber_soccer.bridge import Bridge
    from fiber_soccer.paths import ASSETS
    from fiber_soccer.planner import observation,SCHEMA_V4
    from fiber_soccer.pass_measurement import first_kick_samples
    cfg,shoot_cfg=shared_scene_config(a.pass_config,a.num_envs,a.device,a.seed)
    cfg.commands.motion.ball_resample_after_kick_enabled=False
    cfg.commands.motion.ball_reset_after_kick_enabled=False
    a.output.mkdir(parents=True,exist_ok=False)
    env=gym.make('Tracking-Flat-G1-Rolling-Pass-Event-v1-stage2',cfg=cfg)
    core=env.unwrapped;obs,_=env.reset();device=core.device
    assert [obs[k].shape[-1] for k in ['policy','rma_obs','shoot_policy','shoot_rma_obs']]==[1100,100,1030,150]
    policy=SoccerPolicy(a.checkpoint,device,transition_steps=a.transition_steps)
    assert policy.planner.schema==SCHEMA_V4
    bridge=Bridge(core,ASSETS);command=core.command_manager.get_term('motion')
    stage=torch.zeros(a.num_envs,dtype=torch.long,device=device)
    alive=torch.ones(a.num_envs,dtype=torch.bool,device=device)
    first_pass_tick=torch.full_like(stage,-1)
    pass_records={};shoot_records={};switches=[];resets=[]
    original_reset=core._reset_idx
    def capture_reset(self,ids):
        record_impacts()
        for i in ids.tolist():
            if alive[i]:
                resets.append(dict(tick=tick,environment=i,stage=int(stage[i]),terms={name:bool(self.termination_manager.get_term(name)[i]) for name in self.termination_manager.active_terms}))
        return original_reset(ids)
    core._reset_idx=types.MethodType(capture_reset,core)
    def retained():
        robot=core.scene['robot'].data
        values=[robot.root_state_w,robot.joint_pos,robot.joint_vel,bridge.last,bridge.history]
        for group in core.observation_manager._group_obs_term_history_buffer.values():
            for buffer in group.values():
                values.extend([buffer._buffer,buffer._num_pushes])
        return [v.clone() for v in values if v is not None]
    # Temporarily use the source shooting task's ball sampler only for new balls.
    sampling_keys=[name for name in vars(shoot_cfg.commands.motion) if name.startswith(('ball_init','ball_target','ball_launch','ball_sample_point'))]
    protocol=dict(seed=a.seed,num_envs=a.num_envs,steps=a.steps,recovery_steps=a.recovery_steps,transition_steps=a.transition_steps,shooting_ball_settings={name:getattr(shoot_cfg.commands.motion,name) for name in sampling_keys},scene='retained passing scene',ball_resample_after_kick_disabled=True)
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def new_shooting_ball(ids):
        before=retained();robot=core.scene['robot'].data
        saved={key:copy.deepcopy(getattr(command.cfg,key)) for key in sampling_keys}
        command._freeze_first_ball_quality(ids)
        try:
            for key in sampling_keys:setattr(command.cfg,key,copy.deepcopy(getattr(shoot_cfg.commands.motion,key)))
            command._reset_event_tracking_buffers(ids,reset_pass_quality=False)
            command._sample_ball_episode_state(ids,robot.root_pos_w[ids].clone(),robot.root_quat_w[ids].clone(),robot.root_pos_w[ids].clone(),robot.root_quat_w[ids].clone(),launch_from_current_step=True)
        finally:
            for key,value in saved.items():setattr(command.cfg,key,value)
        core.scene.write_data_to_sim();core.sim.forward()
        after=retained()
        assert all(torch.equal(x,y) for x,y in zip(before,after)),'Ball handoff reset physical or history state'
        ball=core.scene['ball'].data
        assert float(ball.root_lin_vel_w[ids].abs().max())<1e-5
        switches.append(dict(tick=tick,environments=ids.cpu().tolist(),robot_and_history_preserved=True,ball_speed=ball.root_lin_vel_w[ids].norm(dim=-1).cpu().tolist(),distance_to_robot=(ball.root_pos_w[ids,:2]-robot.root_pos_w[ids,:2]).norm(dim=-1).cpu().tolist()))
    def record_impacts():
        ball=core.scene['ball'].data
        eligible=alive.clone()
        for i in range(a.num_envs):
            if i in (pass_records if stage[i]==0 else shoot_records):eligible[i]=False
        valid,values=first_kick_samples(ball.root_lin_vel_w[:,:2],command.ball_target_dir_w[:,:2],command.ball_target_speed,command.ball_kicked,eligible)
        for i in valid.nonzero().flatten().tolist():
            record=dict(tick=tick,preferred_foot=bool(command._preferred_foot_kicked_mask()[i]),**dict(zip(['speed_mps','forward_speed_mps','direction_error_deg','target_speed_mps','relative_forward_speed_error'],values[i].tolist())))
            if stage[i]==0:
                pass_records[i]=record;first_pass_tick[i]=tick
            else:shoot_records[i]=record
    with torch.inference_mode():
        for tick in range(a.steps):
            record_impacts()
            ready=alive&(stage==0)&(first_pass_tick>=0)&((tick-first_pass_tick)>=a.recovery_steps)
            if ready.any():
                ids=ready.nonzero().flatten();new_shooting_ball(ids);stage[ids]=1
                # Histories stay causal and are advanced by the next physics step.
                # The new ball is immediately visible in the planner's true state.
            passing=torch.cat((obs['policy'],obs['rma_obs']),-1)
            shooting=torch.nn.functional.pad(torch.cat((obs['shoot_policy'],obs['shoot_rma_obs']),-1),(0,20))
            history=torch.where((stage==1)[:,None],shooting,passing)
            x=observation(core,bridge,stage,SCHEMA_V4,policy.actor.model,history)
            output=policy(x)
            obs,_,terminated,truncated,_=env.step(bridge.env_action(output['joint_targets']))
            done=terminated|truncated;alive &= ~done
            bridge.advance(output['action_normalized'],done);policy.reset(done)
            if tick%100==0:print('SEQUENCE_PROGRESS',tick,int(alive.sum()),int(stage.sum()),len(pass_records),len(shoot_records),flush=True)
    tick=a.steps
    with torch.inference_mode():record_impacts()
    both=[i for i in range(a.num_envs) if i in pass_records and i in shoot_records]
    accurate=[i for i in both if pass_records[i]['direction_error_deg']<=10 and shoot_records[i]['direction_error_deg']<=10]
    usable=[i for i in accurate if pass_records[i]['preferred_foot'] and shoot_records[i]['preferred_foot']]
    speed_usable=[i for i in usable if pass_records[i]['relative_forward_speed_error']<=.3 and shoot_records[i]['relative_forward_speed_error']<=.3]
    result=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),seed=a.seed,requested=a.num_envs,steps=a.steps,transition_steps=a.transition_steps,recovery_steps=a.recovery_steps,pass_impacts=pass_records,shoot_impacts=shoot_records,switched=int(stage.sum()),both_observed_kicks=len(both),both_direction_le_10deg=len(accurate),both_direction_and_preferred_foot=len(usable),both_direction_preferred_foot_and_speed_30pct=len(speed_usable),both_accurate_and_continuous_through_end=sum(bool(alive[i]) for i in usable),both_kicks_and_no_reset_through_end=sum(bool(alive[i]) for i in both),continuous_survivors=int(alive.sum()),switches=switches,first_resets=resets,one_policy_instance=True,expert_loaded=False,scope='One shared passing scene, static shooting ball sampled after observed pass plus recovery delay. First-observed impact diagnostics; no robot/history reset at handoff. Failed lanes never re-enter counting; original per-task quality metrics are not reused for second ball.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print('SEQUENCE_RESULT',result['both_observed_kicks'],result['both_direction_le_10deg'],flush=True)
    env.close()
finally:
    app.close()
