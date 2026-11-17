"""Optionally initialize new-ball policy terms together with the RMA track.

The ten-frame ball/task terms occupy 2130:2230 in v4 shooting input. Robot
proprioceptive and AH histories remain unchanged. Enabled only by initialize_shoot_ball_task checkpoint metadata.
"""
import torch
from .new_ball_rma import NewBallRmaHistory

TERMS=((2130,2160,0,3),(2160,2190,3,6),(2190,2220,6,9),(2220,2230,9,10))


@torch.no_grad()
def initialize_new_ball_task_history(x,age,initial_measurement):
    if x.ndim!=2 or x.shape[-1]!=2406 or age.shape!=(len(x),) or age.dtype!=torch.long:
        raise ValueError('Expected v4 input and integer object ages')
    if initial_measurement.shape!=(len(x),10) or initial_measurement.device!=x.device or age.device!=x.device:
        raise ValueError('Expected aligned initial ball/task measurements')
    active=(age>=0)&(age<10)
    if not active.any():return x
    if not torch.isfinite(initial_measurement[active]).all():raise ValueError('Nonfinite initial ball/task measurement')
    old=active[:,None] & (torch.arange(10,device=x.device)[None,:]<(10-age)[:,None])
    result=x.clone()
    for start,end,a,b in TERMS:
        history=x[:,start:end].reshape(-1,10,b-a)
        result[:,start:end]=torch.where(old[:,:,None],initial_measurement[:,None,a:b],history).flatten(1)
    return result


class NewBallTaskHistory(NewBallRmaHistory):
    def __init__(self):
        super().__init__()
        self.register_buffer('initial_task_measurement',None,persistent=False)

    @torch.inference_mode()
    def reset(self,env_ids=None):
        super().reset(env_ids)
        if env_ids is None:self.initial_task_measurement=None

    @torch.no_grad()
    def forward(self,x):
        result=super().forward(x)
        if self.initial_task_measurement is None:
            with torch.inference_mode(False):
                self.initial_task_measurement=torch.zeros((len(x),10),device=x.device,dtype=x.dtype)
        start=self.age==1  # RMA helper increments the new track after emitting its first input.
        self.initial_task_measurement[start]=torch.cat((x[start,933:939],x[start,2400:2404]),dim=-1)
        return initialize_new_ball_task_history(result,self.age-1,self.initial_task_measurement)
