"""Short continuous-call smoke test in one passing scene, not a two-skill success test.

Both task observation histories advance on every simulator step. At the switch,
only the skill and requested speed change; robot, ball and histories are retained.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--num-envs', type=int, default=8)
p.add_argument('--steps', type=int, default=200)
p.add_argument('--switch-step', type=int, default=100)
p.add_argument('--seed', type=int, default=813101)
p.add_argument('--pass-config', type=Path, default=Path('/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/params/env.pkl'))
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
assert 0 < a.switch_step < a.steps
app = AppLauncher(a).app
try:
    import torch
    import gymnasium as gym
    import whole_body_tracking.tasks
    from whole_body_tracking.tasks.tracking import mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from fiber_soccer.paths import ASSETS
    from fiber_soccer.bridge import Bridge
    from fiber_soccer.planner import observation, SCHEMA_V4
    from fiber_soccer.planner_policy import SoccerPolicy

    a.output.mkdir(parents=True, exist_ok=False)
    with a.pass_config.open('rb') as f:
        cfg = pickle.load(f)
    for name in ['mpc_ik_reference_based', 'mpc_ik_use_pelvis_velocity']:
        if name not in vars(cfg):
            setattr(cfg, name, False)
    import whole_body_tracking.tasks.tracking.mdp.rewards as rewards
    exec(compile((ASSETS/'expert_rewards.py').read_text(), 'expert_rewards.py', 'exec'), rewards.__dict__)
    cfg.rewards.ready_pose.func = rewards.rolling_pass_ready_pose_l2
    source = '/workspace/batchmpc_issaclab/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/kick_env_cfg.py'
    spec = importlib.util.spec_from_file_location('shared_switch_shoot_cfg', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    shooting = module.G1FlatKickRefEnvCfg()
    # last_action histories are comparable only when raw action normalization agrees.
    for field in ['joint_names', 'scale', 'use_default_offset']:
        assert getattr(cfg.actions.joint_pos, field) == getattr(shooting.actions.joint_pos, field), field
    shot_obs = module.PassEventObservationsCfg()
    shot_obs.policy.ball_pos = ObsTerm(func=mdp.ball_position_body_yaw)
    shot_obs.policy.ball_vel = ObsTerm(func=mdp.ball_relative_velocity_body_yaw)
    shot_obs.policy.ball_target_dir = ObsTerm(func=mdp.ball_target_direction_body_yaw)
    shot_obs.rma_obs.ball_obs = ObsTerm(func=mdp.ball_rma_observation_body_yaw)
    cfg.observations.shoot_policy = copy.deepcopy(shot_obs.policy)
    cfg.observations.shoot_rma_obs = copy.deepcopy(shot_obs.rma_obs)
    cfg.scene.num_envs = a.num_envs
    cfg.sim.device = a.device
    cfg.seed = a.seed
    cfg.commands.motion.debug_vis = False
    env = gym.make('Tracking-Flat-G1-Rolling-Pass-Event-v1-stage2', cfg=cfg)
    core = env.unwrapped
    obs, _ = env.reset()
    assert [obs[k].shape[-1] for k in ['policy','rma_obs','shoot_policy','shoot_rma_obs']] == [1100,100,1030,150]
    bridge = Bridge(core, ASSETS)
    policy = SoccerPolicy(a.checkpoint, core.device)
    assert policy.planner.schema == SCHEMA_V4
    backward = policy.actor.model
    alive = torch.ones(a.num_envs, dtype=torch.bool, device=core.device)
    reset_counts = torch.zeros(a.num_envs, dtype=torch.long, device=core.device)
    switch = None
    def retained_state():
        robot = core.scene['robot'].data
        pieces = [robot.root_state_w, robot.joint_pos, robot.joint_vel,
                  core.scene['ball'].data.root_state_w, bridge.last, bridge.history]
        for groups in core.observation_manager._group_obs_term_history_buffer.values():
            for buffer in groups.values():
                pieces.extend([buffer._buffer, buffer._num_pushes])
        return [v.detach().clone() for v in pieces if v is not None]
    with torch.inference_mode():
        for tick in range(a.steps):
            if tick == a.switch_step:
                before = retained_state()
                core.command_manager.get_term('motion').ball_target_speed.fill_(6.)
                after = retained_state()
                preserved = all(torch.equal(x,y) for x,y in zip(before, after))
                assert preserved
                switch = dict(tick=tick, physical_and_history_state_preserved=preserved,
                              alive_before_switch=alive.cpu().tolist())
            skill = 'pass' if tick < a.switch_step else 'shoot'
            keys = ('policy','rma_obs') if skill == 'pass' else ('shoot_policy','shoot_rma_obs')
            history = torch.cat([obs[k] for k in keys], -1)
            x = observation(core, bridge, skill, policy.planner.schema, backward, history)
            output = policy(x)
            assert all(torch.isfinite(v).all() for v in output.values())
            obs, _, terminated, truncated, _ = env.step(bridge.env_action(output['joint_targets']))
            done = terminated | truncated
            reset_counts += done.long()
            alive &= ~done
            bridge.advance(output['action_normalized'], done)
            if tick % 50 == 0:
                print('SWITCH_PROGRESS',tick,int(alive.sum()),flush=True)
    result = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                  seed=a.seed, steps=a.steps, switch=switch, continuous_survivors=int(alive.sum()),
                  requested=a.num_envs, resets_per_environment=reset_counts.cpu().tolist(),
                  one_policy_instance=True, expert_loaded=False, task_success_measured=False,
                  protocol='Pass scene; pass-to-shoot command change with both causal task histories; no ball respawn at switch. Resets disqualify the entire lane.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    env.close()
finally:
    app.close()
