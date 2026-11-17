"""Experimental input-only initialization of the shooting RMA track for a new ball.

The verified v4 layout stores 50 oldest-first [ball x, ball y, detection] frames
at 2230:2380. Robot/AH histories and environment buffers are never modified.
This helper is not enabled by the current deployment loader.
"""
import torch
from .skill_transition import ContactSkillGate

START,END,FRAMES=2230,2380,50


@torch.no_grad()
def initialize_new_ball_history(x,age,initial_measurement):
    if x.ndim!=2 or x.shape[-1]!=2406 or age.shape!=(len(x),) or age.dtype!=torch.long:
        raise ValueError('Expected v4 input and one integer track age per environment')
    if initial_measurement.shape!=(len(x),3) or initial_measurement.device!=x.device or age.device!=x.device:
        raise ValueError('Expected aligned initial ball measurements')
    active=(age>=0)&(age<FRAMES)
    if not active.any():return x
    if not torch.isfinite(initial_measurement[active]).all():raise ValueError('Nonfinite initial ball measurement')
    old=active[:,None] & (torch.arange(FRAMES,device=x.device)[None,:]<(FRAMES-age)[:,None])
    history=x[:,START:END].reshape(-1,FRAMES,3)
    result=x.clone()
    result[:,START:END]=torch.where(old[:,:,None],initial_measurement[:,None,:],history).flatten(1)
    return result


class NewBallRmaHistory(torch.nn.Module):
    def __init__(self):
        super().__init__();self.contact=ContactSkillGate(0)
        for name in ['age','last_skill','seen_pass_contact','initial_measurement']:
            self.register_buffer(name,None,persistent=False)

    @torch.inference_mode()
    def reset(self,env_ids=None):
        self.contact.reset(env_ids)
        if env_ids is None:
            self.age=None;self.last_skill=None;self.seen_pass_contact=None;self.initial_measurement=None
        elif self.age is not None:
            self.age[env_ids]=-1;self.last_skill[env_ids]=-1;self.seen_pass_contact[env_ids]=False

    @torch.no_grad()
    def forward(self,x):
        if x.ndim!=2 or x.shape[-1]!=2406:raise ValueError('New-ball RMA history requires v4 input')
        skills=x[:,-2:].argmax(-1);kicked=x[:,940]>.5
        if self.age is None:
            with torch.inference_mode(False):
                self.age=torch.full_like(skills,-1);self.last_skill=torch.full_like(skills,-1)
                self.seen_pass_contact=torch.zeros_like(skills,dtype=torch.bool)
                self.initial_measurement=torch.zeros((len(x),3),device=x.device,dtype=x.dtype)
        if self.age.shape!=skills.shape or self.age.device!=x.device:
            raise ValueError('Reset track memory before changing batch/device')
        self.seen_pass_contact |= self.contact(kicked,skills)
        start=(self.last_skill==0)&(skills==1)&self.seen_pass_contact&~kicked
        self.age[start]=0;self.age[skills!=1]=-1
        self.initial_measurement[start,:2]=x[start,933:935]
        self.initial_measurement[start,2]=1.  # Current simulator ball is observed in the true-state planner input.
        result=initialize_new_ball_history(x,self.age,self.initial_measurement)
        self.age[(skills==1)&(self.age>=0)]+=1
        self.last_skill.copy_(skills)
        return result
