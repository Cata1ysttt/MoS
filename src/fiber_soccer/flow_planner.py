"""Conditional endpoint flow baseline, adapted to privileged soccer observations.

Linear Gaussian interpolation, raw clean-endpoint regression, Euler sampling;
sphere projection occurs only for executable output. This is an MLP baseline,
not the FIBER depth/language DiT architecture.
"""
import math
import torch
from torch import nn
from .lpp import Block,project
from .planner import SCHEMA_DIMS,SCHEMA_V3,SCHEMA_V4


class FlowLatentPlanner(nn.Module):
    def __init__(self,horizon=5,width=512,schema=SCHEMA_V3,flow_steps=8,noise_seed=812150,command_conditioning=False):
        super().__init__()
        if horizon<1 or flow_steps<1:raise ValueError("Horizon and flow steps must be positive")
        self.horizon=horizon;self.width=width;self.schema=schema;self.obs_dim=SCHEMA_DIMS[schema]
        self.flow_steps=flow_steps;self.noise_seed=noise_seed;self.command_conditioning=command_conditioning
        self.register_buffer('obs_mean',torch.zeros(self.obs_dim));self.register_buffer('obs_std',torch.ones(self.obs_dim))
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,math.log(1000),16)))
        generator=torch.Generator().manual_seed(noise_seed)
        self.register_buffer('execution_noise',torch.randn(1,horizon,256,generator=generator))
        self.condition=nn.Sequential(nn.Linear(self.obs_dim,width),nn.GELU(),Block(width),nn.LayerNorm(width))
        self.command_encoder=nn.Sequential(nn.Linear(6,width),nn.SiLU(),nn.Linear(width,width),nn.LayerNorm(width)) if command_conditioning else None
        self.field=nn.Sequential(nn.Linear(horizon*256+width+32,width),nn.GELU(),
            Block(width),Block(width),Block(width),Block(width),nn.LayerNorm(width),nn.Linear(width,horizon*256))
        if schema in (SCHEMA_V3,SCHEMA_V4):
            nn.init.zeros_(self.field[-1].weight);nn.init.zeros_(self.field[-1].bias)

    def encode(self,obs):
        normalized=((obs-self.obs_mean)/self.obs_std.clamp_min(.05)).clamp(-20,20)
        context=self.condition(normalized)
        if self.command_encoder is not None:context=context+self.command_encoder(normalized[:,-6:])
        return context

    def endpoint(self,noisy,time,obs,context=None):
        if context is None:context=self.encode(obs)
        phase=time.reshape(-1,1)*self.frequencies[None]
        embedding=torch.cat((phase.sin(),phase.cos()),-1)
        value=self.field(torch.cat((noisy.flatten(1),context,embedding),-1)).reshape(-1,self.horizon,256)
        if self.schema in (SCHEMA_V3,SCHEMA_V4):value=value+obs[:,944:1200,None].transpose(1,2)
        return value

    def forward(self,obs,noise=None):
        value=self.execution_noise.expand(len(obs),-1,-1).clone() if noise is None else noise.clone()
        context=self.encode(obs)
        for step in range(self.flow_steps):
            t=step/self.flow_steps
            time=obs.new_full((len(obs),),t)
            clean=self.endpoint(value,time,obs,context)
            value=value+(clean-value)/(self.flow_steps-step)
        return project(value)

    def act(self,obs):return self(obs)[:,0]


def expand_task_history(model, mean, std):
    """Preserve a v3 policy while adding zero-weight sensor-history inputs."""
    if model.schema != SCHEMA_V3 or mean.shape != (1200,) or std.shape != (1200,):
        raise ValueError('Expansion requires a v3 flow model and 1200 history statistics')
    device = model.obs_mean.device
    expanded = FlowLatentPlanner(model.horizon, model.width, SCHEMA_V4,
                                 model.flow_steps, model.noise_seed, model.command_conditioning).to(device)
    state = dict(model.state_dict())
    weight = torch.zeros_like(expanded.condition[0].weight)
    weight[:, :1200] = state['condition.0.weight'][:, :1200]
    weight[:, -6:] = state['condition.0.weight'][:, -6:]
    state['condition.0.weight'] = weight
    state['obs_mean'] = torch.cat((model.obs_mean[:-6], mean.to(device), model.obs_mean[-6:]))
    state['obs_std'] = torch.cat((model.obs_std[:-6], std.to(device).clamp_min(.05), model.obs_std[-6:]))
    expanded.load_state_dict(state)
    return expanded.eval()
