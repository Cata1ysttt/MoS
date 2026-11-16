"""Construct the shared passing scene with both exact task observation histories."""
from pathlib import Path

def shared_scene_config(pass_config, num_envs, device, seed):
    import copy
    import importlib.util
    import pickle
    import sys
    from whole_body_tracking.tasks.tracking import mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from fiber_soccer.paths import ASSETS
    pass_config=Path(pass_config)
    with pass_config.open('rb') as f:
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
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.seed = seed
    cfg.commands.motion.debug_vis = False
    return cfg, shooting
