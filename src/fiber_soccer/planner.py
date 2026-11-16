"""Unified state-conditioned latent sequence planner (privileged prototype)."""
import torch
from torch import nn
from .lpp import Block, project

OBS_DIM = 945
SCHEMA = 'soccer_state_v1_945'
SCHEMA_V2 = 'soccer_state_v2_950'
SCHEMA_V3 = 'soccer_state_v3_backward_1206'
SCHEMA_DIMS = {SCHEMA: 945, SCHEMA_V2: 950, SCHEMA_V3: 1206}


def observation(core, bridge, skill, schema=SCHEMA, backward_model=None):
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
    if schema not in SCHEMA_DIMS:raise ValueError('Unknown planner observation schema')
    extra=[]
    if schema in (SCHEMA_V2,SCHEMA_V3):
        # The passing teacher uses launch/kick phase and a world-relative
        # recovery anchor. Without these the current state is aliased.
        phase=robot.root_pos_w.new_zeros((core.num_envs,2))
        for i,key in enumerate(('ball_launch_started','ball_kicked')):
            value=getattr(command,key,None)
            if torch.is_tensor(value):phase[:,i]=value.float()
        q=robot.root_quat_w
        heading=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
        position=robot.root_pos_w[:,:2]-core.scene.env_origins[:,:2]
        extra=[phase,position,heading[:,None]]
    if schema == SCHEMA_V3:
        if backward_model is None:raise ValueError('Backward schema requires the paired native model')
        from .native_backward import privileged_from_env
        b=privileged_from_env(core,bridge)
        extra.append(backward_model.backward_map(dict(state=bridge.observe()[0],privileged_state=b)))
    # True simulator state, independent of the two experts' observation layouts.
    x = torch.cat((*bridge.observe(),
        quat_apply_inverse(yaw, robot.root_lin_vel_w),
        quat_apply_inverse(yaw, ball.root_pos_w-robot.root_pos_w),
        quat_apply_inverse(yaw, ball.root_lin_vel_w-robot.root_lin_vel_w),
        *extra,quat_apply_inverse(yaw, command.ball_target_dir_w),
        command.ball_target_speed[:,None], label), -1)
    assert x.shape[-1] == SCHEMA_DIMS[schema]
    return x


class LatentPlanner(nn.Module):
    def __init__(self, horizon=5, width=512, schema=SCHEMA):
        super().__init__()
        self.horizon, self.width = horizon, width
        self.schema=schema;self.obs_dim=SCHEMA_DIMS[schema]
        self.register_buffer('obs_mean', torch.zeros(self.obs_dim))
        self.register_buffer('obs_std', torch.ones(self.obs_dim))
        self.net = nn.Sequential(nn.Linear(self.obs_dim, width), nn.GELU(),
            Block(width), Block(width), Block(width), nn.LayerNorm(width),
            nn.Linear(width, horizon*256))
        if schema==SCHEMA_V3:
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        x = ((obs-self.obs_mean)/self.obs_std.clamp_min(.05)).clamp(-20,20)
        raw=self.net(x).reshape(-1,self.horizon,256)
        if self.schema==SCHEMA_V3:raw=raw+obs[:,-262:-6,None].transpose(1,2)
        return project(raw)

    def act(self, obs):
        return self(obs)[:,0]


def load_planner(checkpoint, device):
    if checkpoint['observation_schema'] not in SCHEMA_DIMS:
        raise ValueError('Planner observation schema mismatch')
    if checkpoint.get('planner_arch')=='flow_endpoint_v1':
        from .flow_planner import FlowLatentPlanner
        model=FlowLatentPlanner(checkpoint['horizon'],checkpoint['width'],checkpoint['observation_schema'],checkpoint['flow_steps'],checkpoint['noise_seed'],checkpoint.get('command_conditioning',False)).to(device)
    else:model=LatentPlanner(checkpoint['horizon'],checkpoint['width'],checkpoint['observation_schema']).to(device)
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
