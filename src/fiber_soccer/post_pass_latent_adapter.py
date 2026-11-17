"""Trainable latent correction on top of the verified history/recovery controller.

The deployed action still comes from one frozen AH-BFM. Only pre-contact
shooting after an observed pass may change. Training must reuse the saved
conditioned input and base latent, never replay the stateful controller during
minibatch optimization.
"""
import time
import torch
from torch import nn
from .initial_goal_recovery import InitialGoalRecoveryPolicy
from .skill_transition import PostPassShootingGate
from .new_ball_rma import initialize_new_ball_history
from .new_ball_task_history import initialize_new_ball_task_history


class LatentResidual(nn.Module):
    def __init__(self,mean,std,width=128):
        super().__init__()
        if mean.shape!=(2406,) or std.shape!=mean.shape:raise ValueError('Expected v4 normalization')
        if isinstance(width,bool) or not isinstance(width,int) or width<1:raise ValueError('Invalid adapter width')
        self.width=width
        self.register_buffer('mean',mean.detach().clone())
        self.register_buffer('std',std.detach().clone().clamp_min(.05))
        with torch.random.fork_rng(devices=[]):
            self.net=nn.Sequential(nn.Linear(2406+256,width),nn.SiLU(),nn.Linear(width,256)).to(mean)
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,base):
        if x.ndim!=2 or x.shape[1]!=2406 or base.shape!=(len(x),256):raise ValueError('Expected aligned v4 inputs and latents')
        normalized=((x-self.mean)/self.std).clamp(-20,20)
        candidate=base+self.net(torch.cat((normalized,base),-1))
        squared=candidate.square().sum(-1,keepdim=True)
        projected=candidate*(base.square().sum(-1,keepdim=True)/squared.clamp_min(1e-12)).sqrt()
        return torch.where(squared>1e-12,projected,base)


class PostPassLatentAdapterPolicy(InitialGoalRecoveryPolicy):
    def __init__(self,checkpoint,device='cpu',**kwargs):
        config=checkpoint.get('post_pass_latent_adapter')
        if not isinstance(config,dict) or config.get('version')!=1:raise ValueError('Unknown post-pass latent adapter')
        if checkpoint.get('predicted_target_refinement_scope')!='initial_pass':raise ValueError('Adapter requires unchanged initial-pass solver')
        if checkpoint.get('new_ball_history_mode')!='initialize_shoot_ball_task':raise ValueError('Adapter requires full ball history')
        super().__init__(checkpoint,device,**kwargs)
        self.post_pass_adapter=LatentResidual(self.planner.obs_mean,self.planner.obs_std,config['width'])
        self.post_pass_adapter.load_state_dict(config['model'],strict=True)
        self.post_pass_adapter.eval().requires_grad_(False)
        self.post_pass_adapter_gate=PostPassShootingGate()
        self.post_pass_latent_adapter_version=1
        self.last_adapter_input=None;self.last_base_latent=None;self.last_adapter_active=None

    def reset(self,env_ids=None):
        super().reset(env_ids);self.post_pass_adapter_gate.reset(env_ids)
        self.last_adapter_input=None;self.last_base_latent=None;self.last_adapter_active=None

    @torch.no_grad()
    def forward(self,observation,skill=None,target_direction=None,target_speed=None):
        device=observation.device
        if device.type=='cuda':torch.cuda.synchronize(device)
        started=time.perf_counter()
        raw=self.condition_observation(observation,skill,target_direction,target_speed)
        output=super().forward(raw)
        # Reconstruct the already-applied input transformation without advancing
        # any state a second time. The parent increments age after conditioning.
        memory=self.new_ball_history;age=memory.age-1
        x=initialize_new_ball_history(raw,age,memory.initial_measurement)
        x=initialize_new_ball_task_history(x,age,memory.initial_task_measurement)
        skills=x[:,-2:].argmax(-1);kicked=x[:,940]>.5
        active=self.post_pass_adapter_gate(kicked,skills)&~kicked
        self.last_adapter_input=x;self.last_base_latent=output['latent'];self.last_adapter_active=active
        if active.any():
            z=self.post_pass_adapter(x,output['latent'])
            action=self.actor(x[:,:64],x[:,64:93],x[:,93:930],z);scaled=(6*action).clamp(-6,6)
            proposed=dict(latent=z,action_normalized=action,action_scaled=scaled,joint_targets=self.default+scaled*self.scale)
            output={key:torch.where(active[:,None],proposed[key],value) for key,value in output.items()}
        if device.type=='cuda':torch.cuda.synchronize(device)
        self.solver_records[-1].update(post_pass_latent_adapter_environments=int(active.sum()),
            latency_seconds=time.perf_counter()-started)
        return output
