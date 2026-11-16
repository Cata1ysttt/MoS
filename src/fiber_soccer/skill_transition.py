"""Optional per-environment latent interpolation when a skill command changes."""
import torch
from .lpp import slerp


class SkillTransition(torch.nn.Module):
    """Keep a stable environment batch; reset lanes after their environment resets.

    Runtime buffers are excluded from checkpoints. A new command during a blend
    starts another blend from the last executed latent, without resetting history.
    """
    def __init__(self, steps):
        super().__init__()
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('Transition steps must be a positive integer')
        self.steps = steps
        for name in ['last', 'start', 'skills', 'remaining', 'valid']:
            self.register_buffer(name, None, persistent=False)

    def reset(self, env_ids=None):
        if env_ids is None:
            for name in ['last', 'start', 'skills', 'remaining', 'valid']:
                setattr(self, name, None)
        elif self.valid is not None:
            self.valid[env_ids] = False
            self.remaining[env_ids] = 0

    @torch.no_grad()
    def forward(self, desired, skills):
        if desired.ndim != 2 or skills.shape != desired.shape[:1]:
            raise ValueError('Expected batched latents and one skill ID per environment')
        if self.last is None:
            self.last = desired.clone()
            self.start = desired.clone()
            self.skills = skills.clone()
            self.remaining = torch.zeros_like(skills, dtype=torch.long)
            self.valid = torch.ones_like(skills, dtype=torch.bool)
            return desired
        if self.last.shape != desired.shape or self.last.device != desired.device:
            raise ValueError('Reset all transition state before changing the environment batch or device')
        changed = self.valid & (self.skills != skills)
        self.start[changed] = self.last[changed]
        self.remaining[changed] = self.steps
        alpha = (self.steps-self.remaining+1).to(desired.dtype)[:,None]/self.steps
        blended = slerp(self.start, desired, alpha.clamp(0,1))
        output = torch.where((self.remaining > 1)[:,None], blended, desired)
        self.remaining = (self.remaining-1).clamp_min(0)
        self.last = output.clone()
        self.skills = skills.clone()
        self.valid.fill_(True)
        return output
