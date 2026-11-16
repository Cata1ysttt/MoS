"""Shared internal target prediction followed by a frozen-BFM latent mapping."""
import torch
from torch import nn
from .planner import SCHEMA_V4


class TargetRegressor(nn.Module):
    def __init__(self, mean, std, default, scale):
        super().__init__()
        if mean.shape!=(2406,) or std.shape!=(2406,) or default.shape!=(29,) or scale.shape!=(29,):
            raise ValueError('Target regressor requires v4 observations and 29 physical targets')
        if not (scale>0).all():raise ValueError('Physical target scales must be positive')
        self.register_buffer('mean',mean.clone())
        self.register_buffer('std',std.clamp_min(.05).clone())
        self.register_buffer('default',default.clone())
        self.register_buffer('scale',scale.clone())
        self.net=nn.Sequential(nn.Linear(2406,512),nn.ELU(),nn.Linear(512,512),nn.ELU(),
                               nn.Linear(512,512),nn.ELU(),nn.Linear(512,29))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        return self.default+self.net(((x-self.mean)/self.std).clamp(-20,20))*self.scale

    def bounded_targets(self,x):
        return self.default+((self(x)-self.default)/self.scale).clamp(-1,1)*self.scale


class TargetGuidedLatentPlanner(nn.Module):
    """One shared correction for both skills, conditioned on predicted targets.

The predictor's physical targets are internal features/training objectives only.
The executable output remains a radius-16 latent decoded by AH-BFM.
"""
    def __init__(self,prior,target_predictor,decoder_width=512):
        super().__init__()
        if prior.schema!=SCHEMA_V4 or getattr(prior,'adapter',None) is not None:
            raise ValueError('Target guidance requires an unadapted v4 prior')
        if decoder_width<1:raise ValueError('Decoder width must be positive')
        self.prior=prior.requires_grad_(False)
        self.target_predictor=target_predictor.requires_grad_(False)
        self.schema=prior.schema;self.obs_dim=prior.obs_dim;self.horizon=prior.horizon
        self.width=prior.width;self.decoder_width=decoder_width
        # Additional modules must not shift simulator randomization on loading.
        with torch.random.fork_rng(devices=[]):
            self.decoder=nn.Sequential(nn.Linear(2406+256+29,decoder_width),nn.SiLU(),
                nn.Linear(decoder_width,decoder_width),nn.SiLU(),nn.Linear(decoder_width,256)).to(prior.obs_mean)
            nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)

    @property
    def obs_mean(self):return self.prior.obs_mean

    @property
    def obs_std(self):return self.prior.obs_std

    def decode(self,x,prior_latent,desired_targets):
        if x.shape!=(len(x),2406) or prior_latent.shape!=(len(x),256) or desired_targets.shape!=(len(x),29):
            raise ValueError('Expected aligned v4 observations, latents, and physical targets')
        normalized=self.prior.normalize_observation(x)
        targets=((desired_targets-self.target_predictor.default)/self.target_predictor.scale).clamp(-1,1)
        correction=self.decoder(torch.cat((prior_latent,normalized,targets),-1))
        candidate=prior_latent+correction
        squared=candidate.square().sum(-1,keepdim=True)
        value=candidate*(prior_latent.square().sum(-1,keepdim=True)/squared.clamp_min(1e-12)).sqrt()
        return torch.where(squared>1e-12,value,prior_latent)

    def forward(self,x):
        prior=self.prior(x)
        desired=self.target_predictor.bounded_targets(x)
        latent=self.decode(x,prior[:,0],desired)
        return torch.cat((latent[:,None],prior[:,1:]),1)

    def act(self,x):return self(x)[:,0]


def target_guided_payload(prior_header,model,config):
    if prior_header['planner_arch']!='flow_endpoint_v1':
        raise ValueError('Only an unadapted flow checkpoint can be a target-guidance prior')
    keys=['planner_arch','observation_schema','horizon','width','flow_steps','noise_seed','command_conditioning']
    prior_config={k:prior_header[k] for k in keys if k in prior_header}
    return dict(prior_header,planner_arch='target_guided_latent_v1',planner=model.state_dict(),
        prior_config=prior_config,decoder_width=model.decoder_width,config=config,
        shoot_adapter_after_pass_only=False)


def load_target_guided(checkpoint,device):
    from .planner import load_planner
    config=checkpoint['prior_config']
    if config.get('planner_arch')!='flow_endpoint_v1' or config.get('observation_schema')!=SCHEMA_V4:
        raise ValueError('Invalid target-guidance prior architecture/schema')
    state=checkpoint['planner']
    prior_state={k.removeprefix('prior.'):v for k,v in state.items() if k.startswith('prior.')}
    prior=load_planner(dict(config,planner=prior_state),device)
    with torch.random.fork_rng(devices=[]):
        predictor=TargetRegressor(*[state['target_predictor.'+key].cpu() for key in ['mean','std','default','scale']]).to(device)
        model=TargetGuidedLatentPlanner(prior,predictor,checkpoint['decoder_width'])
    return model
