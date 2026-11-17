"""Use the episode's initial backward goal after passing follow-through.

No physical reset, expert action or new weight is introduced. The caller still
controls handoff timing; this policy only changes passing post-contact actions.
"""
import time
import torch
from .initial_goal_recovery import InitialGoalRecoveryPolicy
from .skill_transition import ContactSkillGate,PostKickAdapterDelay


class PostPassSettleGate(torch.nn.Module):
    def __init__(self,delay_steps=10):
        super().__init__()
        if isinstance(delay_steps,bool) or not isinstance(delay_steps,int) or delay_steps<1:
            raise ValueError('Passing follow-through delay must be a positive integer')
        self.contact=ContactSkillGate(0);self.delay=PostKickAdapterDelay(delay_steps)

    @torch.inference_mode()
    def reset(self,env_ids=None):
        self.contact.reset(env_ids);self.delay.reset(env_ids)

    def forward(self,kicked,skills):
        passing_contact=self.contact(kicked,skills)&(skills==0)
        return self.delay(passing_contact)


class PostPassSettlePolicy(InitialGoalRecoveryPolicy):
    def __init__(self,checkpoint,device='cpu',**kwargs):
        steps=checkpoint.get('post_pass_settle_delay_steps')
        if isinstance(steps,bool) or not isinstance(steps,int) or steps<1:
            raise ValueError('Missing or invalid passing follow-through delay')
        if 'post_pass_latent_adapter' in checkpoint:raise ValueError('Settling study does not combine with learned correction')
        if checkpoint.get('predicted_target_refinement_scope')!='initial_pass':raise ValueError('Settling requires initial-pass solver scope')
        if checkpoint.get('new_ball_history_mode')!='initialize_shoot_ball_task':raise ValueError('Settling requires full ball history')
        super().__init__(checkpoint,device,**kwargs)
        self.post_pass_settle_delay_steps=steps
        self.settle_gate=PostPassSettleGate(steps)
        self.last_settle_active=None

    def reset(self,env_ids=None):
        with torch.inference_mode():
            super().reset(env_ids);self.settle_gate.reset(env_ids);self.last_settle_active=None

    @torch.no_grad()
    def forward(self,observation,skill=None,target_direction=None,target_speed=None):
        device=observation.device
        if device.type=='cuda':torch.cuda.synchronize(device)
        started=time.perf_counter()
        x=self.condition_observation(observation,skill,target_direction,target_speed)
        output=super().forward(x)
        active=self.settle_gate(x[:,940]>.5,x[:,-2:].argmax(-1));self.last_settle_active=active
        if active.any():
            goal=self.initial_goal.goal
            action=self.actor(x[:,:64],x[:,64:93],x[:,93:930],goal);scaled=(6*action).clamp(-6,6)
            standing=dict(latent=goal,action_normalized=action,action_scaled=scaled,joint_targets=self.default+scaled*self.scale)
            output={key:torch.where(active[:,None],standing[key],value) for key,value in output.items()}
        if device.type=='cuda':torch.cuda.synchronize(device)
        self.solver_records[-1].update(post_pass_settle_environments=int(active.sum()),
            post_pass_settle_delay_steps=self.post_pass_settle_delay_steps,latency_seconds=time.perf_counter()-started)
        return output
