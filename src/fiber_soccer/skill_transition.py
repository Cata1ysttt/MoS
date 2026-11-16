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


class PostKickAdapterDelay(torch.nn.Module):
    """Delay adapter activation after observed contact; one call per control step."""
    def __init__(self,steps):
        super().__init__()
        if not isinstance(steps,int) or steps<1:raise ValueError('Delay must be positive integer steps')
        self.steps=steps
        self.register_buffer('elapsed',None,persistent=False)

    def reset(self,env_ids=None):
        if env_ids is None:self.elapsed=None
        elif self.elapsed is not None:self.elapsed[env_ids]=-1

    def forward(self,kicked):
        if kicked.ndim!=1 or kicked.dtype!=torch.bool:raise ValueError('Expected boolean contact flags')
        if self.elapsed is None:self.elapsed=torch.full_like(kicked,-1,dtype=torch.long)
        if self.elapsed.shape!=kicked.shape or self.elapsed.device!=kicked.device:
            raise ValueError('Reset delay state before changing environment batch/device')
        self.elapsed=torch.where(kicked,self.elapsed+1,torch.full_like(self.elapsed,-1))
        return self.elapsed>=self.steps


class ContactSkillGate(torch.nn.Module):
    """Latch the commanded skill at first observed contact until contact clears.

    Call once per control step, including before recovery switches. A new instance
    cannot infer a prior shooting contact from a passing post-contact observation.
    """
    def __init__(self, skill=1):
        super().__init__()
        if skill not in (0, 1):
            raise ValueError('Contact skill must be 0 (pass) or 1 (shoot)')
        self.skill = skill
        self.register_buffer('origin', None, persistent=False)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.origin = None
        elif self.origin is not None:
            self.origin[env_ids] = -1

    def forward(self, kicked, skills):
        if kicked.ndim != 1 or kicked.dtype != torch.bool or skills.shape != kicked.shape:
            raise ValueError('Expected boolean contact flags and one skill per environment')
        if not ((skills == 0) | (skills == 1)).all():
            raise ValueError('Skill IDs must be 0 (pass) or 1 (shoot)')
        if self.origin is None:
            self.origin = torch.full_like(skills, -1, dtype=torch.long)
        if self.origin.shape != kicked.shape or self.origin.device != kicked.device:
            raise ValueError('Reset contact skill state before changing environment batch/device')
        self.origin = torch.where(kicked, self.origin, torch.full_like(self.origin, -1))
        self.origin = torch.where(kicked & (self.origin < 0), skills, self.origin)
        return kicked & (self.origin == self.skill)


class PostPassShootingGate(torch.nn.Module):
    """Enable shooting adaptation only after a passing contact in this episode.

    Track contact origin so switching to passing during post-shot recovery does
    not count as a completed pass. New-ball contact clearing keeps pass history;
    environment reset clears both contact origin and pass history.
    """
    def __init__(self):
        super().__init__()
        self.contact_origin=ContactSkillGate(0)
        self.register_buffer('seen_pass',None,persistent=False)

    def reset(self,env_ids=None):
        self.contact_origin.reset(env_ids)
        if env_ids is None:self.seen_pass=None
        elif self.seen_pass is not None:self.seen_pass[env_ids]=False

    def forward(self,kicked,skills):
        passed=self.contact_origin(kicked,skills)
        if self.seen_pass is None:self.seen_pass=torch.zeros_like(kicked)
        self.seen_pass=self.seen_pass|passed
        return self.seen_pass&(skills==1)
