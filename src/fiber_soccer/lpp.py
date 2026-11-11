"""FIBER Appendix III primitives. No B checkpoint means action-only, never full LPP."""
import torch
from torch import nn
import torch.nn.functional as F


def project(z):
    fallback=torch.zeros_like(z);fallback[...,0]=16
    return torch.where(z.norm(dim=-1,keepdim=True)>1e-8,F.normalize(z,dim=-1)*16,fallback)


def slerp(a,b,alpha):
    a=project(a);b=project(b)
    cosine=(a*b).sum(-1,keepdim=True).div(256).clamp(-1+1e-6,1-1e-6)
    theta=cosine.acos()
    curved=(torch.sin((1-alpha)*theta)*a+torch.sin(alpha*theta)*b)/theta.sin()
    return project(torch.where(cosine>.9995,(1-alpha)*a+alpha*b,curved))


class Block(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
    def forward(self,x):return x+self.net(x)


class Projector(nn.Module):
    def __init__(self,obs_dim,width=1024):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(obs_dim),nn.Linear(obs_dim,width),*[Block(width) for _ in range(4)],nn.LayerNorm(width),nn.Linear(width,256))
    def forward(self,obs):return project(self.net(obs))


def alignment_loss(target,prediction,mask):
    return ((target-prediction).square().sum(-1)*mask).sum()/(29*mask.sum().clamp_min(1))


def objective(target,prediction,mask,z,b_state=None,mode='action_only'):
    loss=alignment_loss(target,prediction,mask)
    if mode=='full_lpp':
        if b_state is None:raise ValueError('full_lpp requires matching official B encoder')
        loss=loss+.1*(z-b_state.detach()).square().sum(-1).mean()
    elif mode!='action_only':raise ValueError(mode)
    return loss


def execute_latent(previous, endpoint, alpha, mode="slerp"):
    """Shared training/deployment command convention."""
    if mode == "slerp":
        return slerp(previous, endpoint, alpha)
    if mode == "hold":
        return project(endpoint)
    raise ValueError(f"Unknown latent execution mode: {mode}")
