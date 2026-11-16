"""Joint first-pass/first-shot scoring and privileged critic event memory.

The deployable policy observation is unchanged. Original six shooting reward
components are delegated to the existing implementation for both study arms.
"""
import math
import torch
from torch import nn
from .continuous_shooting import ContinuousShootingPhase
from .pass_measurement import first_kick_samples
from .latent_ppo import critic_features


class JointQualityPhase(ContinuousShootingPhase):
    def __init__(self,n,device,joint_impact_bonus=0.,joint_timeout_bonus=0.,**kwargs):
        super().__init__(n,device,**kwargs)
        for value in [joint_impact_bonus,joint_timeout_bonus]:
            if not math.isfinite(value) or value<0:raise ValueError('Invalid joint bonus')
        self.joint_impact_bonus=joint_impact_bonus;self.joint_timeout_bonus=joint_timeout_bonus
        self.pass_quality=torch.zeros_like(self.quality)
        self.pass_accurate=torch.zeros_like(self.accurate)
        self.passes=0;self.accurate_passes=0;self.joint_accurate_shots=0;self.joint_successful_timeouts=0

    def reset(self,done):
        super().reset(done);self.pass_quality[done]=0;self.pass_accurate[done]=False

    def observe(self,velocity,direction,speed,kicked,preferred,eligible,age):
        valid,values=first_kick_samples(velocity,direction,speed,kicked,eligible)
        passed=valid&(self.stage==0)&(self.first_pass<0)
        shot=valid&(self.stage==1)&(self.first_shot<0)
        quality=torch.exp(-(values[:,2]/10).square()-(values[:,4]/.3).square())*preferred
        self.pass_quality[passed]=quality[passed]
        self.pass_accurate[passed]=preferred[passed]&(values[passed,2]<=10)&(values[passed,4]<=.3)
        self.passes+=int(passed.sum());self.accurate_passes+=int((passed&self.pass_accurate).sum())
        super().observe(velocity,direction,speed,kicked,preferred,eligible,age)
        self.joint_accurate_shots+=int((shot&self.joint_accurate()).sum())

    def joint_accurate(self):
        return self.pass_accurate&self.accurate&(self.first_pass>=0)&(self.first_shot>=0)

    def rewards(self,distance,stability,action_rate,terminated,truncated,dt):
        original=super().rewards(distance,stability,action_rate,terminated,truncated,dt)
        joint=self.joint_accurate()
        impact=self.joint_impact_bonus*(joint&self.new_shot).float()
        finish=joint&truncated&~terminated&(self.stage>0)
        self.joint_successful_timeouts+=int(finish.sum())
        return torch.cat((original,impact[:,None],self.joint_timeout_bonus*finish.float()[:,None]),-1)

    def memory(self,horizon_steps):
        if horizon_steps<=0:raise ValueError('Invalid horizon')
        passed=self.first_pass>=0;shot=self.first_shot>=0
        pass_age=(self.age-self.first_pass).clamp_min(0).float()*passed
        shot_age=(self.age-self.first_shot).clamp_min(0).float()*shot
        pass_wait=(1-pass_age/self.pass_delay).clamp(0,1)*passed
        shot_wait=(1-shot_age/self.shoot_delay).clamp(0,1)*shot if self.shoot_delay else torch.zeros_like(shot_age)
        return torch.stack((passed.float(),self.pass_quality,self.pass_accurate.float(),
            shot.float(),self.quality,self.accurate.float(),(pass_age/horizon_steps).clamp(0,1),
            (shot_age/horizon_steps).clamp(0,1),pass_wait,shot_wait),-1)


class MemorySequenceCritic(nn.Module):
    """Zero-initialized memory branch around a restored SequenceCritic."""
    def __init__(self,base):
        super().__init__();self.base=base
        with torch.random.fork_rng(devices=[]):
            self.memory_projection=nn.Linear(10,base.net[0].out_features,bias=False).to(base.mean)
            nn.init.zeros_(self.memory_projection.weight)

    def forward(self,x,stage,age,memory):
        if memory.shape!=(len(x),10):raise ValueError('Expected ten event-memory features per environment')
        features=((critic_features(x)-self.base.mean)/self.base.std).clamp(-20,20)
        phase=torch.nn.functional.one_hot(stage.long(),3).to(x.dtype)
        remaining=(1-age.to(x.dtype)/self.base.horizon_steps).clamp(0,1)[:,None]
        hidden=self.base.net[0](torch.cat((features,phase,remaining),-1))+self.memory_projection(memory.to(x))
        for layer in list(self.base.net)[1:]:hidden=layer(hidden)
        return hidden.squeeze(-1)
