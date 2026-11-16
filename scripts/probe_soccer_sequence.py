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
p.add_argument('--post-shoot-recovery-steps',type=int,help='After observed shooting impact plus this delay, use the same policy passing branch without resampling or resetting')
p.add_argument('--minimum-root-height',type=float,choices=[.4,.45],help='Diagnostic comparison of the two original task height thresholds')
p.add_argument('--events-contract',choices=['pass','shoot'],default='pass',help='Fixed startup and interval randomization configuration')
p.add_argument('--scene-base',choices=['pass','shoot'],default='pass',help='Full source scene control; shoot requires source shooting action and reset contracts')
p.add_argument('--reset-contract',choices=['pass','shoot'],default='pass',help='Fixed initial/reset pose configuration; never resets robot at handoff')
p.add_argument('--action-contract',choices=['pass','shoot'],default='pass',help='Fixed action configuration for the whole run; never changes at a skill boundary')
p.add_argument('--initial-skill',choices=['pass','shoot'],default='pass',help='Shoot starts a static-ball scene diagnostic; it is not a pass-then-shoot sequence')
p.add_argument('--shoot-expert-diagnostic',action='store_true',help='Use the original ONNX only after handoff; never count as autonomous shared-policy performance')
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
assert a.steps>0 and a.recovery_steps>=1
if a.post_shoot_recovery_steps is not None and a.post_shoot_recovery_steps<0:
    p.error('Post-shoot recovery delay must be nonnegative')
if a.scene_base=='shoot' and (a.action_contract!='shoot' or a.reset_contract!='shoot' or a.events_contract!='shoot'):
    p.error('Shooting scene control requires --action-contract shoot --reset-contract shoot --events-contract shoot')
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
    if a.scene_base=='shoot':
        combined_observations=cfg.observations
        cfg=copy.deepcopy(shoot_cfg)
        cfg.observations=combined_observations
        cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device;cfg.seed=a.seed
        cfg.commands.motion.debug_vis=False
    if a.events_contract=='shoot':
        cfg.events=copy.deepcopy(shoot_cfg.events)
    if a.action_contract=='shoot':
        cfg.actions=copy.deepcopy(shoot_cfg.actions)
    if a.reset_contract=='shoot':
        cfg.scene.robot.init_state=copy.deepcopy(shoot_cfg.scene.robot.init_state)
        reset_keys=['pose_range','velocity_range','joint_position_range','randomize_time_on_reset','event_reset_to_reference_pose','prelaunch_training_probability']
        reset_keys += [key for key in vars(shoot_cfg.commands.motion) if key.startswith('reset_joint_')]
        for key in reset_keys:
            setattr(cfg.commands.motion,key,copy.deepcopy(getattr(shoot_cfg.commands.motion,key)))
    if a.minimum_root_height is not None:
        assert 'minimum_height' in cfg.terminations.anchor_pos.params
        cfg.terminations.anchor_pos.params['minimum_height']=a.minimum_root_height
    cfg.commands.motion.ball_resample_after_kick_enabled=False
    cfg.commands.motion.ball_reset_after_kick_enabled=False
    a.output.mkdir(parents=True,exist_ok=False)
    def config_contract(scene_cfg):
        return dict(sim_dt=scene_cfg.sim.dt,decimation=scene_cfg.decimation,
            robot=scene_cfg.scene.robot.to_dict(),actions=scene_cfg.actions.to_dict(),
            events=scene_cfg.events.to_dict(),terminations=scene_cfg.terminations.to_dict(),
            command=scene_cfg.commands.motion.to_dict(),
            ball_dynamics={key:value for key,value in vars(scene_cfg).items() if key.startswith(('ball_','booster_'))})
    (a.output/'scene_contracts.json').write_text(json.dumps(
        {'shared':config_contract(cfg),'original_shoot':config_contract(shoot_cfg)},indent=2,default=str))
    env=gym.make('Tracking-Flat-G1-Rolling-Pass-Event-v1-stage2',cfg=cfg)
    core=env.unwrapped;obs,_=env.reset();device=core.device
    assert [obs[k].shape[-1] for k in ['policy','rma_obs','shoot_policy','shoot_rma_obs']]==[1100,100,1030,150]
    policy=SoccerPolicy(a.checkpoint,device,transition_steps=a.transition_steps)
    assert policy.planner.schema==SCHEMA_V4
    bridge=Bridge(core,ASSETS);command=core.command_manager.get_term('motion')
    stage=torch.zeros(a.num_envs,dtype=torch.long,device=device)
    alive=torch.ones(a.num_envs,dtype=torch.bool,device=device)
    first_pass_tick=torch.full_like(stage,-1)
    first_shoot_tick=torch.full_like(stage,-1)
    recovering=torch.zeros_like(alive)
    recovery_switches=[]
    shoot_expert = None
    if a.shoot_expert_diagnostic:
        import numpy as np
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        shoot_expert = ort.InferenceSession('/workspace/fiber-soccer/body_yaw_49999.onnx',sess_options=options,providers=['CPUExecutionProvider'])
    pass_records={};shoot_records={};switches=[];resets=[]
    original_reset=core._reset_idx
    def capture_reset(self,ids):
        record_impacts()
        for i in ids.tolist():
            if alive[i]:
                robot=self.scene['robot'].data
                resets.append(dict(tick=tick,environment=i,stage=int(stage[i]),root_height=float(robot.root_pos_w[i,2]-self.scene.env_origins[i,2]),root_tilt_rad=float(torch.acos((-robot.projected_gravity_b[i,2]).clamp(-1,1))),terms={name:bool(self.termination_manager.get_term(name)[i]) for name in self.termination_manager.active_terms}))
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
    protocol=dict(post_shoot_recovery_steps=a.post_shoot_recovery_steps,post_shoot_recovery_skill='pass' if a.post_shoot_recovery_steps is not None else None,post_shoot_command_preserved=True,seed=a.seed,num_envs=a.num_envs,steps=a.steps,recovery_steps=a.recovery_steps,transition_steps=a.transition_steps,shooting_ball_settings={name:getattr(shoot_cfg.commands.motion,name) for name in sampling_keys},scene=a.scene_base,ball_resample_after_kick_disabled=True,event_reset_protocol='native_ball_resample_v2',action_contract=a.action_contract,events_contract=a.events_contract,minimum_root_height=cfg.terminations.anchor_pos.params.get('minimum_height'),reset_contract=a.reset_contract,initial_skill=a.initial_skill,expert_action_execution='raw_env_action',shoot_controller='original_expert' if a.shoot_expert_diagnostic else 'unified_planner')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def new_shooting_ball(ids):
        before=retained();robot=core.scene['robot'].data
        saved={key:copy.deepcopy(getattr(command.cfg,key)) for key in sampling_keys}
        command._freeze_first_ball_quality(ids)
        # Match MotionCommand.resample_ball_episode: a new ball also starts a
        # new reference event. The previous kick's reference latch/yaw must not
        # leak into the new task, while robot and actor histories stay intact.
        if command._event_aligned_reference_enabled():
            command._event_reference_started[ids] = False
            command._event_anchor_xy_offset_w[ids] = 0.0
            command._event_reference_yaw_offset_w[ids] = 0.0
        try:
            for key in sampling_keys:setattr(command.cfg,key,copy.deepcopy(getattr(shoot_cfg.commands.motion,key)))
            initial_velocity = command._sample_ball_episode_state(ids,robot.root_pos_w[ids].clone(),robot.root_quat_w[ids].clone(),robot.root_pos_w[ids].clone(),robot.root_quat_w[ids].clone(),launch_from_current_step=True)
            command._reset_event_tracking_buffers(ids,reset_pass_quality=False)
            command._ball_last_vel_xy[ids] = initial_velocity
            assert not command._event_reference_started[ids].any()
            assert not command.ball_kicked[ids].any()
        finally:
            for key,value in saved.items():setattr(command.cfg,key,value)
        core.scene.write_data_to_sim();core.sim.forward()
        after=retained()
        assert all(torch.equal(x,y) for x,y in zip(before,after)),'Ball handoff reset physical or history state'
        ball=core.scene['ball'].data
        assert float(ball.root_lin_vel_w[ids].abs().max())<1e-5
        switches.append(dict(tick=tick,environments=ids.cpu().tolist(),robot_and_history_preserved=True,
            root_height=(robot.root_pos_w[ids,2]-core.scene.env_origins[ids,2]).cpu().tolist(),
            root_speed_xy=robot.root_lin_vel_w[ids,:2].norm(dim=-1).cpu().tolist(),
            root_angular_speed=robot.root_ang_vel_w[ids].norm(dim=-1).cpu().tolist(),
            root_tilt_rad=torch.acos((-robot.projected_gravity_b[ids,2]).clamp(-1,1)).cpu().tolist(),
            ball_speed=ball.root_lin_vel_w[ids].norm(dim=-1).cpu().tolist(),distance_to_robot=(ball.root_pos_w[ids,:2]-robot.root_pos_w[ids,:2]).norm(dim=-1).cpu().tolist()))
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
            else:
                shoot_records[i]=record;first_shoot_tick[i]=tick
    with torch.inference_mode():
        for tick in range(a.steps):
            if tick==0 and a.initial_skill=='shoot':
                ids=torch.arange(a.num_envs,device=device)
                new_shooting_ball(ids);stage[ids]=1
            record_impacts()
            ready=alive&(stage==0)&(first_pass_tick>=0)&((tick-first_pass_tick)>=a.recovery_steps)
            if ready.any():
                ids=ready.nonzero().flatten();new_shooting_ball(ids);stage[ids]=1
                # Histories stay causal and are advanced by the next physics step.
                # The new ball is immediately visible in the planner's true state.
            if a.post_shoot_recovery_steps is not None:
                begin=alive&(stage==1)&~recovering&(first_shoot_tick>=0)&((tick-first_shoot_tick)>=a.post_shoot_recovery_steps)
                if begin.any():
                    before=retained()
                    recovering |= begin
                    recovery_switches.append(dict(tick=tick,environments=begin.nonzero().flatten().cpu().tolist(),robot_and_history_preserved=True))
                    assert all(torch.equal(x,y) for x,y in zip(before,retained()))
            # Sequence stage remains 'shoot' for outcome accounting. Only the
            # selected skill changes after impact; no new ball or command reset.
            skill=torch.where(recovering,torch.zeros_like(stage),stage)
            active_shoot=skill==1
            passing=torch.cat((obs['policy'],obs['rma_obs']),-1)
            shooting=torch.nn.functional.pad(torch.cat((obs['shoot_policy'],obs['shoot_rma_obs']),-1),(0,20))
            history=torch.where(active_shoot[:,None],shooting,passing)
            x=observation(core,bridge,skill,SCHEMA_V4,policy.actor.model,history)
            output=policy(x)
            targets=output['joint_targets']
            executed_action=output['action_normalized']
            if shoot_expert is not None and active_shoot.any():
                ids=active_shoot.nonzero().flatten()
                actions=np.concatenate([shoot_expert.run(['actions'],{'obs':obs['shoot_policy'][i:i+1].cpu().numpy(),'rma_obs':obs['shoot_rma_obs'][i:i+1].cpu().numpy(),'time_step':np.zeros((1,1),np.float32)})[0] for i in ids.tolist()])
                full_action=torch.zeros(a.num_envs,actions.shape[-1],device=device)
                full_action[ids]=torch.as_tensor(actions,device=device)
                expert_targets=bridge.expert_targets(full_action)
                targets=targets.clone();targets[ids]=expert_targets[ids]
                executed_action=executed_action.clone()
                executed_action[ids]=(expert_targets[ids]-bridge.default)/(bridge.scale*6)
            environment_action=bridge.env_action(targets)
            if shoot_expert is not None and active_shoot.any():
                # Preserve the expert's raw action in observation-manager history.
                # Converting clipped physical targets back into actions changes
                # last_action, even when the physical command is equivalent.
                environment_action[ids]=full_action[ids]
            obs,_,terminated,truncated,_=env.step(environment_action)
            done=terminated|truncated;alive &= ~done
            bridge.advance(executed_action,done);policy.reset(done)
            if tick%100==0:print('SEQUENCE_PROGRESS',tick,int(alive.sum()),int(stage.sum()),len(pass_records),len(shoot_records),flush=True)
    tick=a.steps
    with torch.inference_mode():record_impacts()
    both=[i for i in range(a.num_envs) if i in pass_records and i in shoot_records]
    accurate=[i for i in both if pass_records[i]['direction_error_deg']<=10 and shoot_records[i]['direction_error_deg']<=10]
    usable=[i for i in accurate if pass_records[i]['preferred_foot'] and shoot_records[i]['preferred_foot']]
    speed_usable=[i for i in usable if pass_records[i]['relative_forward_speed_error']<=.3 and shoot_records[i]['relative_forward_speed_error']<=.3]
    result=dict(post_shoot_recovery_steps=a.post_shoot_recovery_steps,recovery_switches=recovery_switches,recovery_started=int(recovering.sum()),checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),seed=a.seed,requested=a.num_envs,steps=a.steps,scene_base=a.scene_base,initial_skill=a.initial_skill,transition_steps=a.transition_steps,recovery_steps=a.recovery_steps,pass_impacts=pass_records,shoot_impacts=shoot_records,switched=int(stage.sum()),both_observed_kicks=len(both),both_direction_le_10deg=len(accurate),both_direction_and_preferred_foot=len(usable),both_direction_preferred_foot_and_speed_30pct=len(speed_usable),both_accurate_and_continuous_through_end=sum(bool(alive[i]) for i in usable),both_kicks_and_no_reset_through_end=sum(bool(alive[i]) for i in both),continuous_survivors=int(alive.sum()),switches=switches,first_resets=resets,one_policy_instance=not a.shoot_expert_diagnostic,expert_loaded=a.shoot_expert_diagnostic,shoot_controller='original_expert' if a.shoot_expert_diagnostic else 'unified_planner',scope='Scene selected by scene_base with the recorded initial_skill. With pass initialization, a static shooting ball is sampled after observed pass plus recovery delay; shoot initialization is only a scene diagnostic. First-observed impact diagnostics; no robot/history reset at handoff. Failed lanes never re-enter counting; original per-task quality metrics are not reused for second ball.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print('SEQUENCE_RESULT',result['both_observed_kicks'],result['both_direction_le_10deg'],flush=True)
    env.close()
finally:
    app.close()
