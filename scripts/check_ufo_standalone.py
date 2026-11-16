"""Frozen UFO commands in Isaac, original actuators, no soccer controller.

Step physics directly so soccer commands, rewards and episode resets cannot
interfere. Keep initialized robot physics/randomized gains; remove the ball.
"""
import argparse
import json
import pickle
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True, help='Only locates saved environment config')
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--num-envs', type=int, default=16)
p.add_argument('--steps', type=int, default=500)
p.add_argument('--seed', type=int, default=810311)
p.add_argument('--soccer-action-limits', action='store_true')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import joblib
import numpy as np
import torch
import whole_body_tracking.tasks
from fiber_soccer.paths import ASSETS
from fiber_soccer.ufo import Actor, UFOBridge

try:
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.checkpoint.parent / 'params/env.pkl').open('rb') as f:
        cfg = pickle.load(f)
    for name in ('mpc_ik_reference_based', 'mpc_ik_use_pelvis_velocity'):
        if name not in vars(cfg):
            setattr(cfg, name, False)
    import whole_body_tracking.tasks.tracking.mdp.rewards as rewards
    exec(compile((ASSETS / 'expert_rewards.py').read_text(), 'expert_rewards.py', 'exec'), rewards.__dict__)
    cfg.rewards.ready_pose.func = rewards.rolling_pass_ready_pose_l2
    cfg.commands.motion.debug_vis = False
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = gym.make('Tracking-Flat-G1-Rolling-Pass-Event-v1-stage2', cfg=cfg)
    core = env.unwrapped
    env.reset()
    robot = core.scene['robot']
    bridge = UFOBridge(core)
    actor = Actor(args.device)
    ref = np.load(args.reference)
    assert ref['joint_names'].tolist() == bridge.names
    assert abs(core.step_dt - float(ref['dt'])) < 1e-7
    assert args.steps <= len(ref['z'])
    def tensor(x):
        return torch.as_tensor(x, device=core.device, dtype=torch.float32)
    audit = dict(joint_names=robot.joint_names, kp=robot.data.joint_stiffness.cpu().tolist(),
                 kd=robot.data.joint_damping.cpu().tolist(), dt=core.step_dt,
                 mode='original_pd', soccer_managers_stepped=False, ball_removed=True,
                 history='zero before initial observation; newest first', seed=args.seed,
                 soccer_action_limits=args.soccer_action_limits)
    (args.output / 'contract.json').write_text(json.dumps(audit, indent=2))
    walking = joblib.load(ASSETS / 'ufo/g1_policy/reward_inference/reward_locomotion.pkl')
    commands = [('tracking_motion7', tensor(ref['z']))]
    commands += [(name, tensor(np.asarray(z).reshape(-1, 256)[0]).expand(args.steps, -1))
                 for name, z in walking.items()]
    results = {}
    for name, zs in commands:
        # Tracking starts at reference frame 0; reward uses official standing reset.
        robot.reset()
        root = tensor(ref['root_state_wxyz'][0]).repeat(args.num_envs, 1)
        root[:, :3] += core.scene.env_origins
        q = robot.data.default_joint_pos.clone()
        qd = torch.zeros_like(q)
        q[:, bridge.ids] = tensor(ref['joint_pos'][0])
        qd[:, bridge.ids] = tensor(ref['joint_vel'][0])
        if name != 'tracking_motion7':
            native = json.loads((ASSETS / 'ufo/g1_policy/config.json').read_text())
            init = native['env']['robot_training']['init_state']
            root.zero_()
            root[:, :3] = tensor(init['pos']) + core.scene.env_origins
            root[:, 3:7] = tensor(init['rot'])[[3, 0, 1, 2]]
            q[:, bridge.ids] = bridge.default
            qd.zero_()
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(q, qd)
        robot.set_joint_position_target(q)
        ball = core.scene['ball']
        ball_state = ball.data.default_root_state.clone()
        ball_state[:, :3] += core.scene.env_origins
        ball_state[:, 0] += 50
        ball.write_root_state_to_sim(ball_state)
        core.scene.write_data_to_sim()
        core.scene.update(core.physics_dt)
        bridge.last.zero_()
        bridge.history.zero_()
        bridge.observe(append=True)
        first_fall = torch.full((args.num_envs,), -1, device=core.device, dtype=torch.long)
        traces = []
        for tick in range(args.steps):
            with torch.no_grad():
                a = actor(*bridge.observe(), zs[tick].expand(args.num_envs, -1))
                target = bridge.targets(a)
                applied = bridge.expert_targets(bridge.env_action(target)) if args.soccer_action_limits else target
                q_target = q.clone()
                q_target[:, bridge.ids] = applied
                robot.set_joint_position_target(q_target)
                for _ in range(cfg.decimation):
                    core.scene.write_data_to_sim()
                    core.sim.step(render=False)
                    core.scene.update(core.physics_dt)
                bridge.advance(a, torch.zeros(args.num_envs, device=core.device, dtype=torch.bool))
                d = robot.data
                pos = d.root_pos_w - core.scene.env_origins
                tilt = torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1, 1))
                fallen = (pos[:, 2] < .4) | (tilt > torch.pi / 3) | ~torch.isfinite(pos).all(-1)
                first_fall[(first_fall < 0) & fallen] = tick + 1
                traces.append(dict(root=pos.cpu().clone(), velocity=d.root_lin_vel_b.cpu().clone(),
                                   tilt=tilt.cpu(), joint=d.joint_pos[:, bridge.ids].cpu().clone(),
                                   target_clip=(applied-target).abs().amax(-1).cpu()))
            if tick % 100 == 0:
                print('PROGRESS', name, tick, 'fallen', int((first_fall >= 0).sum()), flush=True)
        torch.save(traces, args.output / f'{name}_trace.pt')
        xyz = torch.stack([t['root'] for t in traces])
        joints = torch.stack([t['joint'] for t in traces])
        velocities = torch.stack([t['velocity'] for t in traces])
        result = dict(steps=args.steps, duration_s=args.steps*core.step_dt,
                      survived=int((first_fall < 0).sum()), total=args.num_envs,
                      first_fall_seconds=[None if x < 0 else x*core.step_dt for x in first_fall.tolist()],
                      fall_rule='root height <0.4m or tilt >60deg or nonfinite; no auto reset',
                      mean_forward_velocity_last_half=velocities[args.steps//2:, :, 0].mean(0).tolist(),
                      max_target_clip_rad=float(torch.stack([t['target_clip'] for t in traces]).max()))
        if name == 'tracking_motion7':
            result['joint_rmse_rad'] = ((joints-torch.tensor(ref['joint_pos'][1:args.steps+1])[:, None])**2).mean((0,2)).sqrt().tolist()
            result['root_position_rmse_m'] = ((xyz-torch.tensor(ref['root_state_wxyz'][1:args.steps+1,:3])[:,None])**2).sum(-1).mean(0).sqrt().tolist()
        results[name] = result
        (args.output / 'summary.json').write_text(json.dumps(results, indent=2))
        print('RESULT', name, json.dumps(result), flush=True)
    env.close()
finally:
    app.close()
