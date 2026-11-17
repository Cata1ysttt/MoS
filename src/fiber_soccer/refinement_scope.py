"""Experimental scope for internal-target refinement, separate from task scoring."""
import torch
from torch import nn


class RefinementScope(nn.Module):
    """Optionally refine only the initial passing phase of each environment.

After a shooting command, all subsequent commands use the frozen shared prior
until that environment resets. This tests whether refinement is harmful in
continuous handoff/recovery states; it does not reset the robot or its history.
"""
    def __init__(self,scope='all'):
        super().__init__()
        if scope not in ['all','initial_pass']:raise ValueError('Unknown refinement scope')
        self.scope=scope
        self.register_buffer('seen_shoot',None,persistent=False)

    @torch.inference_mode()
    def reset(self,env_ids=None):
        if env_ids is None:self.seen_shoot=None
        elif self.seen_shoot is not None:self.seen_shoot[env_ids]=False

    def forward(self,skills):
        if skills.ndim!=1 or not ((skills==0)|(skills==1)).all():
            raise ValueError('Expected one pass/shoot skill ID per environment')
        if self.scope=='all':return torch.ones_like(skills,dtype=torch.bool)
        if self.seen_shoot is None:
            with torch.inference_mode(False):
                self.seen_shoot=torch.zeros_like(skills,dtype=torch.bool)
        if self.seen_shoot.shape!=skills.shape or self.seen_shoot.device!=skills.device:
            raise ValueError('Reset refinement scope before replacing the environment batch')
        self.seen_shoot |= skills==1
        return (skills==0)&~self.seen_shoot
