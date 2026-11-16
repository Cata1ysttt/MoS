"""Unified state-conditioned latent sequence planner (privileged prototype)."""
import torch
from torch import nn
from .lpp import Block, project

OBS_DIM = 945
SCHEMA = 'soccer_state_v1_945'


def observation(core, bridge, skill):
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    robot = core.scene['robot'].data
    ball = core.scene['ball'].data
    command = core.command_manager.get_term('motion')
    yaw = yaw_quat(robot.root_quat_w)
    label = robot.root_pos_w.new_zeros((core.num_envs, 2))
    if isinstance(skill, str):
        label[:, ('pass', 'shoot').index(skill)] = 1
    else:
        label.scatter_(1, skill.long().reshape(-1, 1), 1)
    # True simulator state, independent of the two experts' observation layouts.
    x = torch.cat((*bridge.observe(),
        quat_apply_inverse(yaw, robot.root_lin_vel_w),
        quat_apply_inverse(yaw, ball.root_pos_w-robot.root_pos_w),
        quat_apply_inverse(yaw, ball.root_lin_vel_w-robot.root_lin_vel_w),
        quat_apply_inverse(yaw, command.ball_target_dir_w),
        command.ball_target_speed[:,None], label), -1)
    assert x.shape[-1] == OBS_DIM
    return x


class LatentPlanner(nn.Module):
    def __init__(self, horizon=5, width=512):
        super().__init__()
        self.horizon, self.width = horizon, width
        self.register_buffer('obs_mean', torch.zeros(OBS_DIM))
        self.register_buffer('obs_std', torch.ones(OBS_DIM))
        self.net = nn.Sequential(nn.Linear(OBS_DIM, width), nn.GELU(),
            Block(width), Block(width), Block(width), nn.LayerNorm(width),
            nn.Linear(width, horizon*256))

    def forward(self, obs):
        x = ((obs-self.obs_mean)/self.obs_std.clamp_min(.05)).clamp(-20,20)
        return project(self.net(x).reshape(-1,self.horizon,256))

    def act(self, obs):
        return self(obs)[:,0]


def load_planner(checkpoint, device):
    if checkpoint['observation_schema'] != SCHEMA:
        raise ValueError('Planner observation schema mismatch')
    model=LatentPlanner(checkpoint['horizon'],checkpoint['width']).to(device)
    model.load_state_dict(checkpoint['planner'])
    return model.eval()


def trajectory_windows(x, z, episode, target, horizon):
    """Align future commands with current state, excluding future reset episodes."""
    length=len(x)-horizon+1
    if length <= 0:
        raise ValueError('Trajectory shorter than prediction horizon')
    future=torch.stack([z[j:j+length] for j in range(horizon)],2)
    mask=torch.stack([episode[j:j+length]==episode[:length] for j in range(horizon)],2)
    return x[:length],future,mask,target[:length]
