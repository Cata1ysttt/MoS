"""Experimental scope for internal-target refinement, separate from task scoring."""
import torch
from torch import nn


class RefinementScope(nn.Module):
    """Optionally refine only the initial passing phase of each environment.

After a shooting command, all subsequent commands use the frozen shared prior
until that environment resets. The optional post-pass scope also refines shooting
before contact, only after an observed passing contact. Native shooting and
post-shot recovery remain on the shared prior. This tests whether refinement is harmful in
continuous handoff/recovery states; it does not reset the robot or its history.
"""
    def __init__(self,scope='all'):
        super().__init__()
        if scope not in ['all','initial_pass','initial_pass_and_post_pass_shoot']:raise ValueError('Unknown refinement scope')
        self.scope=scope
        self.post_pass=None
        if scope=='initial_pass_and_post_pass_shoot':
            from .skill_transition import PostPassShootingGate
            self.post_pass=PostPassShootingGate()
        self.register_buffer('seen_shoot',None,persistent=False)

    @torch.inference_mode()
    def reset(self,env_ids=None):
        if self.post_pass is not None:self.post_pass.reset(env_ids)
        if env_ids is None:self.seen_shoot=None
        elif self.seen_shoot is not None:self.seen_shoot[env_ids]=False

    def forward(self,skills,kicked=None):
        if skills.ndim!=1 or not ((skills==0)|(skills==1)).all():
            raise ValueError('Expected one pass/shoot skill ID per environment')
        if self.post_pass is not None and (kicked is None or kicked.dtype!=torch.bool or kicked.shape!=skills.shape):
            raise ValueError('Post-pass refinement requires boolean contact flags')
        if self.scope=='all':return torch.ones_like(skills,dtype=torch.bool)
        if self.seen_shoot is None:
            with torch.inference_mode(False):
                self.seen_shoot=torch.zeros_like(skills,dtype=torch.bool)
        if self.seen_shoot.shape!=skills.shape or self.seen_shoot.device!=skills.device:
            raise ValueError('Reset refinement scope before replacing the environment batch')
        self.seen_shoot |= skills==1
        initial_pass=(skills==0)&~self.seen_shoot
        if self.post_pass is None:return initial_pass
        return initial_pass|(self.post_pass(kicked,skills)&~kicked)
