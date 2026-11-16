"""Unified state-conditioned latent sequence planner (privileged prototype)."""
import torch
from torch import nn
from .lpp import Block, project

OBS_DIM = 945
SCHEMA = 'soccer_state_v1_945'
SCHEMA_V2 = 'soccer_state_v2_950'
SCHEMA_V3 = 'soccer_state_v3_backward_1206'
SCHEMA_V4 = 'soccer_state_v4_task_history_2406'
SCHEMA_DIMS = {SCHEMA: 945, SCHEMA_V2: 950, SCHEMA_V3: 1206, SCHEMA_V4: 2406}

def with_task_history(base, history):
    if base.shape[-1] != 1206 or history.shape[:-1] != base.shape[:-1] or not 0 < history.shape[-1] <= 1200:
        raise ValueError('Expected aligned v3 state and at most 1200 task-history features')
    padded = torch.nn.functional.pad(history, (0, 1200-history.shape[-1]))
    return torch.cat((base[..., :-6], padded, base[..., -6:]), -1)


def observation(core, bridge, skill, schema=SCHEMA, backward_model=None, task_observation=None):
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
    if schema in (SCHEMA_V2,SCHEMA_V3,SCHEMA_V4):
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
    if schema in (SCHEMA_V3, SCHEMA_V4):
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
    if schema == SCHEMA_V4:
        if task_observation is None:
            raise ValueError('Task-history schema requires the current observation-manager histories')
        x = with_task_history(x, task_observation)
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
        if schema in (SCHEMA_V3,SCHEMA_V4):
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        x = ((obs-self.obs_mean)/self.obs_std.clamp_min(.05)).clamp(-20,20)
        raw=self.net(x).reshape(-1,self.horizon,256)
        if self.schema in (SCHEMA_V3,SCHEMA_V4):raw=raw+obs[:,944:1200,None].transpose(1,2)
        return project(raw)

    def act(self, obs):
        return self(obs)[:,0]


class DirectLatentPlanner(nn.Module):
    """Deterministic latent regression ablation with the same state/command encoder."""
    def __init__(self, horizon=1, width=512, schema=SCHEMA_V4, command_conditioning=True):
        super().__init__()
        if horizon < 1 or schema not in (SCHEMA_V3, SCHEMA_V4):
            raise ValueError('Direct latent planner requires a positive horizon and backward-state schema')
        self.horizon, self.width, self.schema = horizon, width, schema
        self.obs_dim = SCHEMA_DIMS[schema]
        self.command_conditioning = command_conditioning
        self.register_buffer('obs_mean', torch.zeros(self.obs_dim))
        self.register_buffer('obs_std', torch.ones(self.obs_dim))
        self.condition = nn.Sequential(nn.Linear(self.obs_dim,width),nn.GELU(),Block(width),nn.LayerNorm(width))
        self.command_encoder = nn.Sequential(nn.Linear(6,width),nn.SiLU(),nn.Linear(width,width),nn.LayerNorm(width)) if command_conditioning else None
        self.head = nn.Sequential(Block(width),Block(width),Block(width),nn.LayerNorm(width),nn.Linear(width,horizon*256))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @classmethod
    def from_context(cls, model, horizon=1):
        direct = cls(horizon,model.width,model.schema,model.command_conditioning).to(model.obs_mean.device)
        direct.obs_mean.copy_(model.obs_mean)
        direct.obs_std.copy_(model.obs_std)
        direct.condition.load_state_dict(model.condition.state_dict())
        if direct.command_encoder is not None:
            direct.command_encoder.load_state_dict(model.command_encoder.state_dict())
        return direct

    def encode(self, obs):
        normalized = ((obs-self.obs_mean)/self.obs_std.clamp_min(.05)).clamp(-20,20)
        context = self.condition(normalized)
        if self.command_encoder is not None:
            context = context+self.command_encoder(normalized[:,-6:])
        return context

    def raw(self, obs):
        residual = self.head(self.encode(obs)).reshape(-1,self.horizon,256)
        return residual+obs[:,944:1200,None].transpose(1,2)

    def forward(self, obs):
        return project(self.raw(obs))

    def act(self, obs):
        return self(obs)[:,0]


def load_planner(checkpoint, device):
    if checkpoint['observation_schema'] not in SCHEMA_DIMS:
        raise ValueError('Planner observation schema mismatch')
    if checkpoint.get('planner_arch')=='direct_latent_v1':
        model=DirectLatentPlanner(checkpoint['horizon'],checkpoint['width'],checkpoint['observation_schema'],checkpoint.get('command_conditioning',True)).to(device)
    elif checkpoint.get('planner_arch') in ('flow_endpoint_v1','flow_endpoint_adapter_v1'):
        from .flow_planner import FlowLatentPlanner
        model=FlowLatentPlanner(checkpoint['horizon'],checkpoint['width'],checkpoint['observation_schema'],checkpoint['flow_steps'],checkpoint['noise_seed'],checkpoint.get('command_conditioning',False),checkpoint.get('adapter_width',0),checkpoint.get('adapter_skill',0)).to(device)
    else:model=LatentPlanner(checkpoint['horizon'],checkpoint['width'],checkpoint['observation_schema']).to(device)
    model.load_state_dict(checkpoint['planner'])
    return model.eval()


def trajectory_windows(x, z, episode, target, horizon, label_confidence=None):
    """Align future commands with current state, excluding future reset episodes."""
    length=len(x)-horizon+1
    if length <= 0:
        raise ValueError('Trajectory shorter than prediction horizon')
    future=torch.stack([z[j:j+length] for j in range(horizon)],2)
    mask=torch.stack([episode[j:j+length]==episode[:length] for j in range(horizon)],2)
    if label_confidence is not None:
        if label_confidence.shape != episode.shape:
            raise ValueError("Label confidence must align with trajectory episodes")
        if not torch.isfinite(label_confidence).all() or not ((label_confidence > 0) & (label_confidence <= 1)).all():
            raise ValueError("Label confidence must be finite and in (0, 1]")
        mask = mask * torch.stack([label_confidence[j:j+length] for j in range(horizon)], 2)
    return x[:length],future,mask,target[:length]


def balanced_skill_mean(values, weights):
    """Average two equally sized skill batches without confidence changing task balance."""
    if len(values) % 2 or values.shape != weights.shape:
        raise ValueError("Expected equal pass/shoot batches with aligned weights")
    return torch.stack([(v*w).sum()/w.sum().clamp_min(1e-12)
                        for v, w in zip(values.chunk(2), weights.chunk(2))]).mean()


def observed_contact_window(x, episode, before=10, after=5):
    """Mark frames around observed kicked-phase onsets, never across episode resets.

    Callers must use a schema whose kicked flag is at index 940 (v2/v3/v4).
    An onset is not necessarily a good kick; this is sampling, not evaluation.
    """
    if x.shape[:2] != episode.shape or x.shape[-1] < 941 or before < 0 or after < 0:
        raise ValueError("Invalid contact-window inputs")
    kicked = x[:, :, 940] > .5
    onset = torch.zeros_like(kicked)
    onset[1:] = kicked[1:] & ~kicked[:-1] & (episode[1:] == episode[:-1])
    window = torch.zeros_like(onset)
    for offset in range(-before, after+1):
        if offset < 0:
            window[:offset] |= onset[-offset:] & (episode[:offset] == episode[-offset:])
        elif offset > 0:
            window[offset:] |= onset[:-offset] & (episode[offset:] == episode[:-offset])
        else:
            window |= onset
    return window
