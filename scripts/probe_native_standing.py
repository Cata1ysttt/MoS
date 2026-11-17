"""Standing capability diagnostic; neither a soccer policy nor recovery evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import types
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--controller',choices=['direct_default','latent_default','direct_reset_pose','latent_reset_pose','backward_stationary'],required=True)
p.add_argument('--num-envs',type=int,default=16)
p.add_argument('--steps',type=int,default=1000)
p.add_argument('--seed',type=int,default=816401)
p.add_argument('--pass-config',type=Path,default=Path('/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/params/env.pkl'))
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();assert a.steps>0 and a.num_envs>0
app=AppLauncher(a).app
try:
    import torch
    import gymnasium as gym
    import whole_body_tracking.tasks
    from fiber_soccer.shared_scene import shared_scene_config,scene_contract
    from fiber_soccer.bridge import Bridge
    from fiber_soccer.paths import ASSETS,ROOT
    from fiber_soccer.native_bfm import load_native,NativeActor
    from fiber_soccer.native_backward import privileged_from_env,MAPPING
    from fiber_soccer.captured_latent_solver import CapturedLatentSolver
    _,shooting=shared_scene_config(a.pass_config,a.num_envs,a.device,a.seed)
    cfg=copy.deepcopy(shooting)
    cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device;cfg.seed=a.seed
    cfg.commands.motion.debug_vis=False
    cfg.episode_length_s=max(cfg.episode_length_s,(a.steps+1)*cfg.sim.dt*cfg.decimation)
    removed=getattr(cfg.terminations,'event_kick_elapsed',None)
    if removed is not None:
        assert removed.func.__name__=='event_kick_elapsed'
        cfg.terminations.event_kick_elapsed=None
    cfg.commands.motion.ball_resample_after_kick_enabled=False
    cfg.commands.motion.ball_reset_after_kick_enabled=False
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'scene_contract.json').write_text(json.dumps(scene_contract(cfg),indent=2,default=str)+'\n')
    actor_path=ROOT/'AH-BFM/AH-BFM.pt'
    actor_sha=hashlib.sha256(actor_path.read_bytes()).hexdigest()
    assert actor_sha=='9d8b9ae1b775c4b57034a49a051e527c8470456b51093ff9f7d6467dceb895d9'
    env=gym.make('Tracking-Flat-G1-Rolling-Pass-Event-v1-stage2',cfg=cfg)
    core=env.unwrapped;env.reset();device=core.device
    bridge=Bridge(core,ASSETS);robot=core.scene['robot'].data
    initial={k:v.cpu().clone() for k,v in dict(root=robot.root_state_w,joint_pos=robot.joint_pos,
        joint_vel=robot.joint_vel,ball=core.scene['ball'].data.root_state_w).items()}
    initial_sha=hashlib.sha256(b''.join(k.encode()+v.numpy().tobytes() for k,v in sorted(initial.items()))).hexdigest()
    torch.save(initial,a.output/'initial_state.pt')
    initial_xy=robot.root_pos_w[:,:2].clone()
    initial_joint_pose=robot.joint_pos[:,bridge.ids].clone()
    actor=NativeActor(load_native(device)).eval().requires_grad_(False)
    # Derive initialization from this author checkpoint, never the old BFM z_bank.
    with torch.no_grad():
        goal_state=bridge.observe()[0].clone()
        goal_privileged=privileged_from_env(core,bridge).clone()
        if a.controller=='backward_stationary':
            # Preserve the actual reset pose, but request zero joint/body velocity.
            assert goal_state.shape[-1]==64 and goal_privileged.shape[-1]==463
            goal_state[:,29:58]=0;goal_state[:,61:64]=0
            goal_privileged[:,-186:]=0  # 31 link linear and angular velocities.
        z=actor.model.backward_map(dict(state=goal_state,privileged_state=goal_privileged)).detach()
        torch.save(dict(state=goal_state.cpu(),privileged_state=goal_privileged.cpu(),latent=z.cpu()),a.output/'goal.pt')
    desired=bridge.default.expand(a.num_envs,-1) if a.controller.endswith('_default') else initial_joint_pose
    weight=torch.ones(29,device=device);weight[:12]=2;root_weight=(weight/weight.mean()).sqrt()
    solver=CapturedLatentSolver(actor,lambda raw:bridge.targets(raw)*root_weight,64)
    alive=torch.ones(a.num_envs,dtype=torch.bool,device=device);failures=[];records=[]
    original_reset=core._reset_idx
    def capture_reset(self,ids):
        for i in ids.tolist():
            if alive[i]:
                failures.append(dict(tick=tick,environment=i,
                    root_height=float(robot.root_pos_w[i,2]-core.scene.env_origins[i,2]),
                    tilt_rad=float(torch.acos((-robot.projected_gravity_b[i,2]).clamp(-1,1))),
                    terms={name:bool(self.termination_manager.get_term(name)[i]) for name in self.termination_manager.active_terms}))
        return original_reset(ids)
    core._reset_idx=types.MethodType(capture_reset,core)
    with torch.no_grad():
        for tick in range(a.steps):
            state=bridge.observe();info=None
            if a.controller in ['latent_default','latent_reset_pose']:
                z,info=solver.solve(state,desired*root_weight,z)
                raw=actor(*state,z);targets=bridge.targets(raw)
            elif a.controller=='backward_stationary':
                raw=actor(*state,z);targets=bridge.targets(raw)
            else:
                targets=desired;raw=(targets-bridge.default)/(6*bridge.scale)
            action=bridge.env_action(targets)
            if (~alive).any():
                action[~alive]=0
                zero_targets=bridge.expert_targets(torch.zeros_like(action))
                raw=raw.clone();raw[~alive]=(zero_targets[~alive]-bridge.default)/(6*bridge.scale)
            if tick%10==0:
                records.append(dict(tick=tick,alive=alive.cpu().tolist(),
                    height=(robot.root_pos_w[:,2]-core.scene.env_origins[:,2]).cpu().tolist(),
                    tilt_rad=torch.acos((-robot.projected_gravity_b[:,2]).clamp(-1,1)).cpu().tolist(),
                    displacement_xy=(robot.root_pos_w[:,:2]-initial_xy).norm(dim=-1).cpu().tolist(),
                    target_mse=(targets-desired).square().mean(-1).cpu().tolist(),solver=info))
            _,_,terminated,truncated,_=env.step(action)
            done=terminated|truncated;alive &= ~done;bridge.advance(raw,done)
            if tick%100==0:print('STANDING_PROGRESS',tick,int(alive.sum()),flush=True)
    assert len(failures)==a.num_envs-int(alive.sum())
    assert all(param.grad is None and not param.requires_grad for param in actor.parameters())
    assert hashlib.sha256(actor_path.read_bytes()).hexdigest()==actor_sha
    summary=dict(controller=a.controller,seed=a.seed,requested=a.num_envs,steps=a.steps,
        duration_s=a.steps*core.step_dt,survivors=int(alive.sum()),failures=failures,
        initial_state_sha256=initial_sha,actor_sha256=actor_sha,backward_mapping=MAPPING,
        backward_mapping_author_validated=False,expert_loaded=False,training=False,
        target='Bridge.default joint pose' if a.controller.endswith('_default') else 'Per-environment measured source reset joint pose',
        target_usage='Pose is only a diagnostic metric; actor follows a fixed backward goal with zero velocities' if a.controller=='backward_stationary' else 'Fixed joint target, directly commanded or inverted through frozen AH-BFM',
        goal_velocities_zeroed=a.controller=='backward_stationary',
        desired_normalized_action_abs_max=float(((desired-bridge.default)/(6*bridge.scale)).abs().max()),
        solver_steps=64 if a.controller in ['latent_default','latent_reset_pose'] else 0,
        failed_lane_control='zero',all_failure_terms_retained=True,
        removed_normal_termination=removed.to_dict() if removed is not None else None,
        scope='Standalone standing from source shooting reset, no previous kick. Direct controllers bypass AH-BFM and are physical comparators; latent and backward-goal controllers always use frozen AH-BFM. Neither is unified soccer or post-shot recovery success.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    (a.output/'trace.json').write_text(json.dumps(records)+'\n')
    print('STANDING_RESULT',summary['survivors'],flush=True)
    env.close()
finally:
    app.close()
