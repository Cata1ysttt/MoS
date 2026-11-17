"""Remember a reset-state backward goal and use it only after shooting contact.

The caller must start this stateful policy at episode reset, maintain stable
batch slots, and call reset(done). The standing goal is episode memory, not a
new expert or a change to the frozen AH-BFM weights.
"""
import time
import torch
from .skill_transition import ContactSkillGate
from .predicted_target_solver import PredictedTargetSolverPolicy

MODE='initial_backward_goal'


class InitialBackwardGoal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.contact=ContactSkillGate(1)
        self.register_buffer('goal',None,persistent=False)
        self.register_buffer('valid',None,persistent=False)

    def reset(self,env_ids=None):
        with torch.inference_mode():
            self.contact.reset(env_ids)
            if env_ids is None:self.goal=None;self.valid=None
            elif self.valid is not None:self.valid[env_ids]=False

    @torch.no_grad()
    def forward(self,x):
        if x.ndim!=2 or x.shape[-1]!=2406:raise ValueError('Initial goal requires v4 observations')
        current=x[:,944:1200]
        if self.goal is None:
            with torch.inference_mode(False):
                self.goal=torch.zeros_like(current)
                self.valid=torch.zeros(len(x),device=x.device,dtype=torch.bool)
        if self.goal.shape!=current.shape or self.goal.device!=x.device:
            raise ValueError('Reset initial goal before changing batch/device')
        capture=~self.valid
        if capture.any():
            values=current[capture]
            if not torch.isfinite(values).all() or not torch.allclose(values.norm(dim=-1),values.new_full((len(values),),16.),atol=1e-3,rtol=0):
                raise ValueError('Initial backward goals must be finite radius-16 vectors')
            self.goal[capture]=values;self.valid[capture]=True
        skills=x[:,-2:].argmax(-1)
        active=self.contact(x[:,940]>.5,skills)&(skills==0)
        return self.goal,active


class InitialGoalRecoveryPolicy(PredictedTargetSolverPolicy):
    def __init__(self,checkpoint,device='cpu',**kwargs):
        if type(self) is InitialGoalRecoveryPolicy and any(k in checkpoint for k in ['post_pass_latent_adapter','post_pass_settle_delay_steps','post_pass_settle_trigger']):
            raise ValueError('Use load_soccer_policy for post-pass extensions')
        if checkpoint.get('standing_recovery_mode')!=MODE:
            raise ValueError('Unknown standing recovery mode')
        if checkpoint.get('predicted_target_refinement_scope') not in ('initial_pass','initial_pass_and_post_pass_shoot'):
            raise ValueError('Initial-goal recovery requires initial-pass refinement scope')
        mode=checkpoint.get('new_ball_history_mode')
        if 'new_ball_history_mode' in checkpoint and mode not in ('initialize_shoot_rma','initialize_shoot_ball_task'):
            raise ValueError('Unknown new-ball history mode')
        super().__init__(checkpoint,device,**kwargs)
        self.standing_recovery_mode=MODE
        self.initial_goal=InitialBackwardGoal().eval()
        self.new_ball_history=None
        if mode is not None:
            from .new_ball_rma import NewBallRmaHistory
            self.new_ball_history_mode=mode
            if mode=='initialize_shoot_ball_task':
                from .new_ball_task_history import NewBallTaskHistory
                self.new_ball_history=NewBallTaskHistory().eval()
            else:self.new_ball_history=NewBallRmaHistory().eval()

    def reset(self,env_ids=None):
        with torch.inference_mode():
            super().reset(env_ids)
            self.initial_goal.reset(env_ids)
            if self.new_ball_history is not None:self.new_ball_history.reset(env_ids)

    @torch.no_grad()
    def forward(self,observation,skill=None,target_direction=None,target_speed=None):
        device=observation.device
        if device.type=='cuda':torch.cuda.synchronize(device)
        started=time.perf_counter()
        x=self.condition_observation(observation,skill,target_direction,target_speed)
        if self.new_ball_history is not None:x=self.new_ball_history(x)
        goal,active=self.initial_goal(x)
        output=super().forward(x)
        if active.any():
            raw=self.actor(x[:,:64],x[:,64:93],x[:,93:930],goal)
            scaled=(6*raw).clamp(-6,6)
            standing=dict(joint_targets=self.default+scaled*self.scale,
                action_scaled=scaled,action_normalized=raw,latent=goal)
            output={key:torch.where(active[:,None],standing[key],value) for key,value in output.items()}
        if device.type=='cuda':torch.cuda.synchronize(device)
        self.solver_records[-1].update(latency_seconds=time.perf_counter()-started,
            standing_recovery_mode=MODE,standing_recovery_environments=int(active.sum()))
        if self.new_ball_history is not None:
            age=self.new_ball_history.age
            self.solver_records[-1].update(new_ball_history_mode=self.new_ball_history_mode,
                new_ball_history_environments=int(((age>0)&(age<=50)).sum()))
            if self.new_ball_history_mode=='initialize_shoot_ball_task':
                self.solver_records[-1]['new_ball_task_history_environments']=int(((age>0)&(age<=10)).sum())
        return output
