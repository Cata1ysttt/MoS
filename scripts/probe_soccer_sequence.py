"""Pass then static-ball shoot with one policy and no robot/history resets.

The default uses passing-scene dynamics and terminations. An explicit shooting
scene diagnostic can retain shooting physics while importing the incoming-ball
command. Continuous mode removes normal post-kick termination and extends the
timeout; all failure terms stay active. Neither is the two original tasks.
"""
import argparse
import copy
import hashlib
import json
import math
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
p.add_argument('--handoff-seed',type=int,help='Isolate each environment’s second-ball Torch sampling from other environments and call order')
p.add_argument('--recovery-steps',type=int,default=50)
p.add_argument('--transition-steps',type=int,default=10)
p.add_argument('--post-shoot-recovery-steps',type=int,help='After observed shooting impact plus this delay, use the same policy passing branch without resampling or resetting')
p.add_argument('--post-shoot-recovery-speed',type=float,help='Planner-only speed override after recovery starts; physical task command and shot scoring remain unchanged')
p.add_argument('--minimum-root-height',type=float,choices=[.4,.45],help='Diagnostic comparison of the two original task height thresholds')
p.add_argument('--events-contract',choices=['pass','shoot'],default='pass',help='Fixed startup and interval randomization configuration')
p.add_argument('--scene-base',choices=['pass','shoot'],default='pass',help='Full source scene control; shoot requires source shooting action and reset contracts')
p.add_argument('--incoming-pass-command',action='store_true',help='Use the complete passing motion command in a shooting physics scene; initial ball is incoming')
p.add_argument('--continuous-after-kick',action='store_true',help='Disable normal post-kick termination and extend timeout through the requested horizon; retain all failure terms')
p.add_argument('--reset-contract',choices=['pass','shoot'],default='pass',help='Fixed initial/reset pose configuration; never resets robot at handoff')
p.add_argument('--action-contract',choices=['pass','shoot'],default='pass',help='Fixed action configuration for the whole run; never changes at a skill boundary')
p.add_argument('--initial-skill',choices=['pass','shoot'],default='pass',help='Shoot starts a static-ball scene diagnostic; it is not a pass-then-shoot sequence')
p.add_argument('--shoot-expert-diagnostic',action='store_true',help='Use the original ONNX only after handoff; never count as autonomous shared-policy performance')
p.add_argument('--post-kick-adapter-delay',type=int,default=0,help='Control steps after observed contact before enabling a post-kick-only adapter')
p.add_argument('--failed-lane-control',choices=['policy','zero'],default='policy',help='Use identical raw zero actions after a lane first fails; failed lanes remain excluded from success counts')
p.add_argument('--audit-post-pass-gate',action='store_true',help='Passively compare gated and ungated outputs on every alive state')
p.add_argument('--record-shoot-states',action='store_true',help='Passively save alive post-handoff shooting inputs for offline expert labeling')
p.add_argument('--collection-purpose',choices=['diagnostic','training'],default='diagnostic')
p.add_argument('--recovery-adapter-shoot-only',action='store_true',help='Enable recovery correction only after a contact first observed under the shooting skill')
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
assert a.steps>0 and a.recovery_steps>=1
if a.handoff_seed is not None and a.handoff_seed<0:p.error('Handoff seed must be nonnegative')
if a.post_shoot_recovery_steps is not None and a.post_shoot_recovery_steps<0:
    p.error('Post-shoot recovery delay must be nonnegative')
if a.post_shoot_recovery_speed is not None and (a.post_shoot_recovery_steps is None or not math.isfinite(a.post_shoot_recovery_speed) or a.post_shoot_recovery_speed<=0):
    p.error('Recovery speed requires a recovery delay and a finite positive value')
if a.scene_base=='shoot' and (a.action_contract!='shoot' or a.reset_contract!='shoot' or a.events_contract!='shoot'):
    p.error('Shooting scene control requires --action-contract shoot --reset-contract shoot --events-contract shoot')
if a.incoming_pass_command and (a.scene_base!='shoot' or a.initial_skill!='pass'):
    p.error('Incoming passing command requires shooting scene and pass initialization')
if a.scene_base=='shoot' and a.initial_skill=='pass' and not a.incoming_pass_command:
    p.error('Pass initialization in a shooting scene requires --incoming-pass-command')
if a.record_shoot_states and (a.shoot_expert_diagnostic or a.scene_base!='shoot' or a.initial_skill!='pass'):
    p.error('State recording requires autonomous pass-to-shoot execution in the source shooting scene')
app=AppLauncher(a).app
try:
    import torch
    import gymnasium as gym
    import whole_body_tracking.tasks
    from fiber_soccer.shared_scene import shared_scene_config,apply_shooting_reset
    from fiber_soccer.planner_policy import load_soccer_policy,recovery_controller_speed
    from fiber_soccer.bridge import Bridge
    from fiber_soccer.paths import ASSETS
    from fiber_soccer.planner import observation,SCHEMA_V4
    from fiber_soccer.pass_measurement import first_kick_samples
    from fiber_soccer.isolated_sampling import event_seed,isolated_torch_rng
    from isaaclab.utils.math import quat_apply_inverse,yaw_quat
    cfg,shoot_cfg=shared_scene_config(a.pass_config,a.num_envs,a.device,a.seed)
    pass_motion=copy.deepcopy(cfg.commands.motion)
    if a.scene_base=='shoot':
        combined_observations=cfg.observations
        cfg=copy.deepcopy(shoot_cfg)
        cfg.observations=combined_observations
        cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device;cfg.seed=a.seed
        cfg.commands.motion.debug_vis=False
    if a.incoming_pass_command:
        # Copy the full command, including receive-point, TTC, launch delay and
        # prelaunch fields. Copying only ball_init* produces a different task.
        cfg.commands.motion=pass_motion
    removed_normal_termination=None
    original_episode_length_s=cfg.episode_length_s
    if a.continuous_after_kick:
        cfg.episode_length_s=max(cfg.episode_length_s,(a.steps+1)*cfg.sim.dt*cfg.decimation)
        term=getattr(cfg.terminations,'event_kick_elapsed',None)
        if term is not None:
            assert term.func.__name__=='event_kick_elapsed'
            removed_normal_termination=term.to_dict()
            cfg.terminations.event_kick_elapsed=None
    if a.events_contract=='shoot':
        cfg.events=copy.deepcopy(shoot_cfg.events)
    if a.action_contract=='shoot':
        cfg.actions=copy.deepcopy(shoot_cfg.actions)
    if a.reset_contract=='shoot':
        apply_shooting_reset(cfg,shoot_cfg)
    if a.minimum_root_height is not None:
        assert 'minimum_height' in cfg.terminations.anchor_pos.params
        cfg.terminations.anchor_pos.params['minimum_height']=a.minimum_root_height
    cfg.commands.motion.ball_resample_after_kick_enabled=False
    cfg.commands.motion.ball_reset_after_kick_enabled=False
    a.output.mkdir(parents=True,exist_ok=False)
    def config_contract(scene_cfg):
        return dict(sim_dt=scene_cfg.sim.dt,decimation=scene_cfg.decimation,episode_length_s=scene_cfg.episode_length_s,
            scene=scene_cfg.scene.to_dict(),sim=scene_cfg.sim.to_dict(),
            robot=scene_cfg.scene.robot.to_dict(),actions=scene_cfg.actions.to_dict(),
            events=scene_cfg.events.to_dict(),terminations=scene_cfg.terminations.to_dict(),
            command=scene_cfg.commands.motion.to_dict(),
            ball_dynamics={key:value for key,value in vars(scene_cfg).items() if key.startswith(('ball_','booster_'))})
    (a.output/'scene_contracts.json').write_text(json.dumps(
        {'shared':config_contract(cfg),'original_shoot':config_contract(shoot_cfg)},indent=2,default=str))
    env=gym.make('Tracking-Flat-G1-Rolling-Pass-Event-v1-stage2',cfg=cfg)
    core=env.unwrapped;obs,_=env.reset();device=core.device
    assert [obs[k].shape[-1] for k in ['policy','rma_obs','shoot_policy','shoot_rma_obs']]==[1100,100,1030,150]
    initial_robot=core.scene['robot'].data
    initial_state=dict(robot_root=initial_robot.root_state_w.cpu().clone(),
        joint_pos=initial_robot.joint_pos.cpu().clone(),joint_vel=initial_robot.joint_vel.cpu().clone(),
        ball_root=core.scene['ball'].data.root_state_w.cpu().clone(),
        **{key:obs[key].cpu().clone() for key in ['policy','rma_obs','shoot_policy','shoot_rma_obs']})
    initial_state_sha256=hashlib.sha256(b''.join(
        key.encode()+value.numpy().tobytes() for key,value in sorted(initial_state.items()))).hexdigest()
    torch.save(initial_state,a.output/'initial_state.pt')
    policy=load_soccer_policy(a.checkpoint,device,transition_steps=a.transition_steps,recovery_adapter_shoot_only=a.recovery_adapter_shoot_only,post_kick_adapter_delay=a.post_kick_adapter_delay)
    assert policy.planner.schema==SCHEMA_V4
    if hasattr(policy,'solver_records') and a.shoot_expert_diagnostic:
        raise ValueError('Internal target refinement cannot be combined with expert takeover')
    if a.audit_post_pass_gate:
        assert policy.shoot_adapter_after_pass_only and a.transition_steps==0
        assert not a.shoot_expert_diagnostic and a.post_kick_adapter_delay==0
        assert a.post_shoot_recovery_speed is None,'Gate audit requires the unmodified observation command'
    gate_audit=dict(alive_states_checked=0,alive_latent_outputs_exact=True,alive_joint_targets_exact=True,
        failed_states_with_different_latents=0,first_failed_difference_tick=None,comparison_rng_preserved=True)
    bridge=Bridge(core,ASSETS);command=core.command_manager.get_term('motion')
    (a.output/'reset_ball_state.json').write_text(json.dumps(dict(scope='Environment reset before any initial-skill override or policy action',ball_velocity_w=core.scene['ball'].data.root_lin_vel_w.cpu().tolist(),ball_relative_position_w=(core.scene['ball'].data.root_pos_w-core.scene['robot'].data.root_pos_w).cpu().tolist()),indent=2))
    stage=torch.zeros(a.num_envs,dtype=torch.long,device=device)
    alive=torch.ones(a.num_envs,dtype=torch.bool,device=device)
    first_pass_tick=torch.full_like(stage,-1)
    first_shoot_tick=torch.full_like(stage,-1)
    recovering=torch.zeros_like(alive)
    previous_contact=torch.zeros_like(alive);contact_observations=[]
    recovery_switches=[]
    shoot_expert = None
    if a.shoot_expert_diagnostic:
        import numpy as np
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        shoot_expert = ort.InferenceSession('/workspace/fiber-soccer/body_yaw_49999.onnx',sess_options=options,providers=['CPUExecutionProvider'])
    recorder=None
    if a.record_shoot_states:
        from fiber_soccer.sequence_collection import SequenceShootingRecorder
        assert bridge.term.cfg.clip is None,'Offline labels require no post-affine physical clipping'
        assert type(core).__name__=='TrackingKickEnv','Raw action clamp contract requires TrackingKickEnv'
        recorder=SequenceShootingRecorder(a.num_envs,bridge.order)
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
    protocol=dict(recovery_adapter_shoot_only=a.recovery_adapter_shoot_only,post_kick_adapter_delay=a.post_kick_adapter_delay,post_shoot_recovery_speed=a.post_shoot_recovery_speed,handoff_seed=a.handoff_seed,shooting_ball_rng='isolated_per_environment_v1' if a.handoff_seed is not None else 'shared_global_rng',original_episode_length_s=original_episode_length_s,episode_length_s=cfg.episode_length_s,incoming_pass_command=a.incoming_pass_command,continuous_after_kick=a.continuous_after_kick,removed_normal_termination=removed_normal_termination,post_shoot_recovery_steps=a.post_shoot_recovery_steps,post_shoot_recovery_skill='pass' if a.post_shoot_recovery_steps is not None else None,post_shoot_command_preserved=True,seed=a.seed,num_envs=a.num_envs,steps=a.steps,recovery_steps=a.recovery_steps,transition_steps=a.transition_steps,shooting_ball_settings={name:getattr(shoot_cfg.commands.motion,name) for name in sampling_keys},scene=a.scene_base,ball_resample_after_kick_disabled=True,event_reset_protocol='native_ball_resample_v2',action_contract=a.action_contract,events_contract=a.events_contract,minimum_root_height=cfg.terminations.anchor_pos.params.get('minimum_height'),reset_contract=a.reset_contract,initial_skill=a.initial_skill,expert_action_execution='raw_env_action',shoot_controller='original_expert' if a.shoot_expert_diagnostic else 'unified_planner')
    if hasattr(policy,'solver_records'):
        protocol.update(shoot_controller='predicted_target_latent_solver',
            predicted_target_solver_steps=policy.solver_steps,predicted_target_solver_backend=policy.solver_backend,
            predicted_target_solver_expert_targets_used=False)
    if getattr(policy,'refinement_scope','all')!='all':protocol['predicted_target_refinement_scope']=policy.refinement_scope
    if a.failed_lane_control!='policy':protocol['failed_lane_control']=a.failed_lane_control
    if policy.shoot_adapter_after_pass_only:protocol['shoot_adapter_after_pass_only']=True
    if hasattr(policy,'standing_recovery_mode'):
        protocol['standing_recovery_mode']=policy.standing_recovery_mode
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2,default=str))
    def new_shooting_ball(ids):
        before=retained();robot=core.scene['robot'].data
        rng_before=(torch.random.get_rng_state(),torch.cuda.get_rng_state(device) if torch.device(device).type=='cuda' else None) if a.handoff_seed is not None else None
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
            def sample(selected):
                velocity=command._sample_ball_episode_state(selected,robot.root_pos_w[selected].clone(),robot.root_quat_w[selected].clone(),robot.root_pos_w[selected].clone(),robot.root_quat_w[selected].clone(),launch_from_current_step=True)
                # This reset also samples observation/relaunch durations.
                command._reset_event_tracking_buffers(selected,reset_pass_quality=False)
                command._ball_last_vel_xy[selected]=velocity
            if a.handoff_seed is None:
                sample(ids)
            else:
                for environment in ids.tolist():
                    with isolated_torch_rng(event_seed(a.handoff_seed,environment),device):
                        sample(ids.new_tensor([environment]))
            assert not command._event_reference_started[ids].any()
            assert not command.ball_kicked[ids].any()
        finally:
            for key,value in saved.items():setattr(command.cfg,key,value)
        core.scene.write_data_to_sim();core.sim.forward()
        after=retained()
        assert all(torch.equal(x,y) for x,y in zip(before,after)),'Ball handoff reset physical or history state'
        ball=core.scene['ball'].data
        assert float(ball.root_lin_vel_w[ids].abs().max())<1e-5
        if rng_before is not None:
            assert torch.equal(rng_before[0],torch.random.get_rng_state())
            if rng_before[1] is not None:assert torch.equal(rng_before[1],torch.cuda.get_rng_state(device))
        switches.append(dict(tick=tick,environments=ids.cpu().tolist(),robot_and_history_preserved=True,
            handoff_seeds=[event_seed(a.handoff_seed,i) for i in ids.tolist()] if a.handoff_seed is not None else None,
            torch_rng_restored=True if rng_before is not None else None,
            ball_offset_yaw=quat_apply_inverse(yaw_quat(robot.root_quat_w[ids]),ball.root_pos_w[ids]-robot.root_pos_w[ids]).cpu().tolist(),
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
                    recovery_switches.append(dict(tick=tick,environments=begin.nonzero().flatten().cpu().tolist(),robot_and_history_preserved=True,task_target_speed=command.ball_target_speed[begin].cpu().tolist(),planner_target_speed=(torch.full_like(command.ball_target_speed[begin],a.post_shoot_recovery_speed) if a.post_shoot_recovery_speed is not None else command.ball_target_speed[begin]).cpu().tolist()))
                    assert all(torch.equal(x,y) for x,y in zip(before,retained()))
            # Sequence stage remains 'shoot' for outcome accounting. Only the
            # selected skill changes after impact; no new ball or command reset.
            skill=torch.where(recovering,torch.zeros_like(stage),stage)
            active_shoot=skill==1
            passing=torch.cat((obs['policy'],obs['rma_obs']),-1)
            shooting=torch.nn.functional.pad(torch.cat((obs['shoot_policy'],obs['shoot_rma_obs']),-1),(0,20))
            history=torch.where(active_shoot[:,None],shooting,passing)
            x=observation(core,bridge,skill,SCHEMA_V4,policy.actor.model,history)
            contact=command.ball_kicked.clone()
            observed=alive&contact&~previous_contact
            for i in observed.nonzero().flatten().tolist():
                contact_observations.append(dict(tick=tick,environment=i,stage=int(stage[i])))
            previous_contact=contact
            controller_speed=recovery_controller_speed(command.ball_target_speed,recovering,a.post_shoot_recovery_speed)
            output=policy(x,target_speed=controller_speed)
            if tick==0 and hasattr(policy,'standing_recovery_mode'):
                assert torch.equal(policy.initial_goal.goal,x[:,944:1200])
                torch.save(policy.initial_goal.goal.cpu().clone(),a.output/'initial_backward_goal.pt')
            if a.audit_post_pass_gate:
                audit_rng=torch.random.get_rng_state()
                audit_cuda_rng=torch.cuda.get_rng_state(device) if torch.device(device).type=='cuda' else None
                ungated=policy.planner.act(x)
                assert torch.equal(output['latent'][alive],ungated[alive]),('Live latent mismatch',tick)
                raw=policy.actor(x[:,:64],x[:,64:93],x[:,93:930],ungated)
                expected=policy.default+(raw*6).clamp(-6,6)*policy.scale
                assert torch.equal(output['joint_targets'][alive],expected[alive]),('Live target mismatch',tick)
                assert torch.equal(audit_rng,torch.random.get_rng_state())
                if audit_cuda_rng is not None:assert torch.equal(audit_cuda_rng,torch.cuda.get_rng_state(device))
                changed=(output['latent']!=ungated).any(-1)&~alive
                gate_audit['alive_states_checked']+=int(alive.sum())
                gate_audit['failed_states_with_different_latents']+=int(changed.sum())
                if changed.any() and gate_audit['first_failed_difference_tick'] is None:
                    gate_audit['first_failed_difference_tick']=tick
            if recorder is not None:
                recorder.record(tick,x,output['latent'],alive,bridge.term._scale,bridge.term._offset)
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
            if a.failed_lane_control=='zero' and (~alive).any():
                # Keep already-failed lanes from following different controllers.
                # They still simulate/reset, and never re-enter outcome counting.
                environment_action[~alive]=0
                zero_targets=bridge.expert_targets(torch.zeros_like(environment_action))
                executed_action=executed_action.clone()
                executed_action[~alive]=(zero_targets[~alive]-bridge.default)/(bridge.scale*6)
            obs,_,terminated,truncated,_=env.step(environment_action)
            done=terminated|truncated;alive &= ~done
            bridge.advance(executed_action,done);policy.reset(done);previous_contact[done]=False
            if tick%100==0:print('SEQUENCE_PROGRESS',tick,int(alive.sum()),int(stage.sum()),len(pass_records),len(shoot_records),flush=True)
    tick=a.steps
    with torch.inference_mode():record_impacts()
    both=[i for i in range(a.num_envs) if i in pass_records and i in shoot_records]
    accurate=[i for i in both if pass_records[i]['direction_error_deg']<=10 and shoot_records[i]['direction_error_deg']<=10]
    usable=[i for i in accurate if pass_records[i]['preferred_foot'] and shoot_records[i]['preferred_foot']]
    speed_usable=[i for i in usable if pass_records[i]['relative_forward_speed_error']<=.3 and shoot_records[i]['relative_forward_speed_error']<=.3]
    result=dict(contact_observations=contact_observations,recovery_adapter_shoot_only=a.recovery_adapter_shoot_only,post_kick_adapter_delay=a.post_kick_adapter_delay,post_shoot_recovery_speed=a.post_shoot_recovery_speed,initial_state_sha256=initial_state_sha256,handoff_seed=a.handoff_seed,incoming_pass_command=a.incoming_pass_command,continuous_after_kick=a.continuous_after_kick,post_shoot_recovery_steps=a.post_shoot_recovery_steps,recovery_switches=recovery_switches,recovery_started=int(recovering.sum()),checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),seed=a.seed,requested=a.num_envs,steps=a.steps,scene_base=a.scene_base,initial_skill=a.initial_skill,transition_steps=a.transition_steps,recovery_steps=a.recovery_steps,pass_impacts=pass_records,shoot_impacts=shoot_records,switched=int(stage.sum()),both_observed_kicks=len(both),both_direction_le_10deg=len(accurate),both_direction_and_preferred_foot=len(usable),both_direction_preferred_foot_and_speed_30pct=len(speed_usable),both_accurate_and_continuous_through_end=sum(bool(alive[i]) for i in usable),both_direction_foot_speed_and_continuous_through_end=sum(bool(alive[i]) for i in speed_usable),both_kicks_and_no_reset_through_end=sum(bool(alive[i]) for i in both),continuous_survivors=int(alive.sum()),switches=switches,first_resets=resets,one_policy_instance=not a.shoot_expert_diagnostic,expert_loaded=a.shoot_expert_diagnostic,shoot_controller='original_expert' if a.shoot_expert_diagnostic else 'unified_planner',scope='Scene selected by scene_base with the recorded initial_skill. With pass initialization, a static shooting ball is sampled after observed pass plus recovery delay; shoot initialization is only a scene diagnostic. First-observed impact diagnostics; no robot/history reset at handoff. Failed lanes never re-enter counting; original per-task quality metrics are not reused for second ball.')
    if hasattr(policy,'solver_records'):
        assert not a.shoot_expert_diagnostic
        result.update(shoot_controller='predicted_target_latent_solver',
            predicted_target_solver_steps=policy.solver_steps,predicted_target_solver_backend=policy.solver_backend,
            predicted_target_solver_expert_targets_used=False,real_time_50hz_verified=False)
        latency=torch.tensor([r['latency_seconds'] for r in policy.solver_records])
        result['policy_latency_seconds']={k:float(v) for k,v in zip(['median','p95','maximum'],torch.quantile(latency,torch.tensor([.5,.95,1.])))}
        (a.output/'predicted_target_solver.json').write_text(json.dumps(policy.solver_records))
    if getattr(policy,'refinement_scope','all')!='all':result['predicted_target_refinement_scope']=policy.refinement_scope
    if hasattr(policy,'standing_recovery_mode'):
        result['standing_recovery_mode']=policy.standing_recovery_mode
        result['standing_recovery_active_control_steps']=sum(r['standing_recovery_environments'] for r in policy.solver_records)
    if a.failed_lane_control!='policy':result['failed_lane_control']=a.failed_lane_control
    if policy.shoot_adapter_after_pass_only:result['shoot_adapter_after_pass_only']=True
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print('SEQUENCE_RESULT',result['both_observed_kicks'],result['both_direction_le_10deg'],flush=True)
    if a.audit_post_pass_gate:
        (a.output/'gate_runtime_audit.json').write_text(json.dumps(gate_audit,indent=2))
    if recorder is not None:
        payload=recorder.export(dict(seed=a.seed,handoff_seed=a.handoff_seed,purpose=a.collection_purpose,
            checkpoint=str(a.checkpoint),checkpoint_sha256=result['checkpoint_sha256'],
            initial_state_sha256=initial_state_sha256,steps=a.steps,dt=core.step_dt,
            initial_skill=a.initial_skill,scene_base=a.scene_base,physical_target_mapping='raw_action_clamp_10_then_affine_no_physical_clip',raw_action_clip=10.))
        torch.save(payload,a.output/'shoot_states.pt')
        (a.output/'shoot_collection.json').write_text(json.dumps(dict(payload['metadata'],
            states=len(payload['x']),environments=payload['environment'].unique().tolist()),indent=2))
    env.close()
finally:
    app.close()
