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


def apply_shooting_reset(cfg, shooting):
    """Keep the source shooting robot pose and command reset distribution."""
    import copy
    cfg.scene.robot.init_state = copy.deepcopy(shooting.scene.robot.init_state)
    keys = ['pose_range', 'velocity_range', 'joint_position_range',
            'randomize_time_on_reset', 'event_reset_to_reference_pose',
            'prelaunch_training_probability']
    keys += [key for key in vars(shooting.commands.motion) if key.startswith('reset_joint_')]
    for key in keys:
        setattr(cfg.commands.motion, key, copy.deepcopy(getattr(shooting.commands.motion, key)))


def scene_contract(cfg):
    """Serializable physical/task configuration before environment construction."""
    return dict(sim_dt=cfg.sim.dt, decimation=cfg.decimation,
        episode_length_s=cfg.episode_length_s,
        scene=cfg.scene.to_dict(), sim=cfg.sim.to_dict(),
        robot=cfg.scene.robot.to_dict(), actions=cfg.actions.to_dict(),
        events=cfg.events.to_dict(), terminations=cfg.terminations.to_dict(),
        command=cfg.commands.motion.to_dict(), rewards=cfg.rewards.to_dict(),
        ball_dynamics={key:value for key,value in vars(cfg).items()
                       if key.startswith(('ball_', 'booster_'))})


def passing_in_shooting_scene(passing, shooting):
    """Train incoming passes in the physical scene used for continuous switching.

    Keep the complete passing command/rewards and its episode time limit.
    Ball resampling remains enabled as in passing training. Shooting failures
    remain active; only the normal post-kick episode ending is removed.
    """
    import copy
    cfg = copy.deepcopy(shooting)
    cfg.observations = copy.deepcopy(passing.observations)
    cfg.commands.motion = copy.deepcopy(passing.commands.motion)
    cfg.rewards = copy.deepcopy(passing.rewards)
    cfg.episode_length_s = passing.episode_length_s
    cfg.is_finite_horizon = passing.is_finite_horizon
    cfg.scene.num_envs = passing.scene.num_envs
    cfg.sim.device = passing.sim.device
    cfg.seed = passing.seed
    cfg.commands.motion.debug_vis = False
    apply_shooting_reset(cfg, shooting)
    term = getattr(cfg.terminations, 'event_kick_elapsed', None)
    if term is not None:
        if term.func.__name__ != 'event_kick_elapsed':
            raise ValueError('Unexpected normal post-kick termination')
        cfg.terminations.event_kick_elapsed = None
    return cfg


def shooting_recovery_scene(passing,shooting):
    """Static shot then learned recovery, in the continuous scene's physical contract."""
    import copy
    from isaaclab.managers import RewardTermCfg
    from isaaclab.utils import configclass
    from .recovery_training import recovery_stability,recovery_action_rate,recovery_failure
    cfg=passing_in_shooting_scene(passing,shooting)
    for key in vars(shooting.commands.motion):
        if key.startswith(('ball_init','ball_target','ball_launch','ball_sample_point')):
            setattr(cfg.commands.motion,key,copy.deepcopy(getattr(shooting.commands.motion,key)))
    cfg.commands.motion.ball_resample_after_kick_enabled=False
    cfg.commands.motion.ball_reset_after_kick_enabled=False
    cfg.episode_length_s=20.
    @configclass
    class RecoveryRewards:
        stability=RewardTermCfg(func=recovery_stability,weight=5.,params={'target_height':shooting.scene.robot.init_state.pos[2]})
        action_rate=RewardTermCfg(func=recovery_action_rate,weight=-.05)
        failure=RewardTermCfg(func=recovery_failure,weight=-100.)
    cfg.rewards=RecoveryRewards()
    return cfg
