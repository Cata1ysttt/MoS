"""Single deployable policy call: unified observations/commands -> joint targets.

SoccerPolicy is feedforward. load_soccer_policy also supports checkpoints that
explicitly request internal-target latent refinement. Neither loads an expert.
The caller supplies the checkpoint schema from planner.observation and advances
its BFM state/action history with the actual executed action.
Optional skill transitions, adapter delays and contact-origin gating require stable batch slots, one call per
control step, and reset(done_mask) after environment resets.
"""
import json
from pathlib import Path
import torch
from .planner import load_planner
from .native_bfm import NativeActor,load_native
from .paths import ASSETS


class SoccerPolicy(torch.nn.Module):
    def __init__(self, checkpoint, device='cpu', transition_steps=0, post_kick_adapter_delay=0, recovery_adapter_shoot_only=False, shoot_adapter_after_pass_only=None):
        super().__init__()
        if isinstance(checkpoint,(str,Path)):
            checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if type(self) is SoccerPolicy and any(k in checkpoint for k in ['predicted_target_solver_steps','predicted_target_solver_backend','predicted_target_refinement_scope','standing_recovery_mode']):
            raise ValueError('Use load_soccer_policy for internally refined checkpoints')
        self.planner=load_planner(checkpoint,device)
        self.actor=NativeActor(load_native(device))
        meta=json.loads((ASSETS/'actor_meta.json').read_text())
        scene=json.loads((ASSETS/'pickup_scene.json').read_text())
        self.joint_names=meta['joint_names']
        self.register_buffer('default',torch.tensor(scene['default_q'],device=device))
        self.register_buffer('scale',torch.tensor(meta['action_rescale'],device=device)*.25)
        if not isinstance(transition_steps,int) or transition_steps<0:
            raise ValueError('Transition steps must be a nonnegative integer')
        from .skill_transition import SkillTransition,PostKickAdapterDelay,ContactSkillGate,PostPassShootingGate
        self.transition=SkillTransition(transition_steps) if transition_steps else None
        if not isinstance(post_kick_adapter_delay,int) or post_kick_adapter_delay<0:
            raise ValueError('Post-kick adapter delay must be nonnegative integer steps')
        if post_kick_adapter_delay and getattr(self.planner,'adapter',None) is not None and not getattr(self.planner,'adapter_post_kick_only',False):
            raise ValueError('Delayed activation requires a post-kick-only adapter')
        self.post_kick_delay=PostKickAdapterDelay(post_kick_adapter_delay) if post_kick_adapter_delay and getattr(self.planner,'adapter_post_kick_only',False) else None
        if not isinstance(recovery_adapter_shoot_only,bool):
            raise ValueError('Recovery adapter shooting-origin gate must be boolean')
        if recovery_adapter_shoot_only and getattr(self.planner,'adapter',None) is not None and not getattr(self.planner,'adapter_post_kick_only',False):
            raise ValueError('Shooting-origin gating requires a post-kick-only adapter')
        self.contact_skill_gate=ContactSkillGate(1) if recovery_adapter_shoot_only and getattr(self.planner,'adapter_post_kick_only',False) else None
        if shoot_adapter_after_pass_only is None:
            shoot_adapter_after_pass_only=checkpoint.get('shoot_adapter_after_pass_only',False)
        if not isinstance(shoot_adapter_after_pass_only,bool):
            raise ValueError('Post-pass shooting gate must be boolean')
        self.shoot_adapter_after_pass_only=shoot_adapter_after_pass_only
        if shoot_adapter_after_pass_only and getattr(self.planner,'adapter',None) is not None:
            if self.planner.adapter_skill!=1 or self.planner.adapter_post_kick_only:
                raise ValueError('Post-pass gating requires a shooting adapter without post-kick-only gating')
            self.post_pass_gate=PostPassShootingGate()
        else:self.post_pass_gate=None
        self.eval().requires_grad_(False)

    def reset(self, env_ids=None):
        if self.transition is not None:self.transition.reset(env_ids)
        if self.post_kick_delay is not None:self.post_kick_delay.reset(env_ids)
        if self.contact_skill_gate is not None:self.contact_skill_gate.reset(env_ids)
        if self.post_pass_gate is not None:self.post_pass_gate.reset(env_ids)

    def condition_observation(self, observation, skill=None, target_direction=None, target_speed=None):
        if observation.shape[-1]!=self.planner.obs_dim:raise ValueError('Planner observation schema mismatch')
        x=observation.clone()
        if skill is not None:
            if isinstance(skill,str):
                skill=torch.full((len(x),),('pass','shoot').index(skill),device=x.device)
            ids=torch.as_tensor(skill,device=x.device).long().expand(len(x))
            if not ((ids>=0)&(ids<2)).all():raise ValueError('Skill IDs must be 0 (pass) or 1 (shoot)')
            x[:,-2:]=0;x[:,-2:].scatter_(1,ids[:,None],1)
        if target_direction is not None:x[:,-6:-3]=torch.as_tensor(target_direction,device=x.device)
        if target_speed is not None:x[:,-3]=torch.as_tensor(target_speed,device=x.device)
        return x

    @torch.no_grad()
    def forward(self, observation, skill=None, target_direction=None, target_speed=None):
        x=self.condition_observation(observation,skill,target_direction,target_speed)
        mask=self.post_kick_delay(x[:,940]>.5) if self.post_kick_delay is not None else None
        if self.contact_skill_gate is not None:
            origin_mask=self.contact_skill_gate(x[:,940]>.5,x[:,-2:].argmax(-1))
            mask=origin_mask if mask is None else mask & origin_mask
        if self.post_pass_gate is not None:
            context_mask=self.post_pass_gate(x[:,940]>.5,x[:,-2:].argmax(-1))
            mask=context_mask if mask is None else mask & context_mask
        z=self.planner.act(x,adapter_mask=mask) if mask is not None else self.planner.act(x)
        if self.transition is not None:z=self.transition(z,x[:,-2:].argmax(-1))
        action=self.actor(x[:,:64],x[:,64:93],x[:,93:930],z)
        scaled=(action*6).clamp(-6,6)
        return dict(joint_targets=self.default+scaled*self.scale,
                    action_scaled=scaled,action_normalized=action,latent=z)


def load_soccer_policy(checkpoint,device='cpu',**kwargs):
    """Load the controller declared by a checkpoint through one common interface."""
    if isinstance(checkpoint,(str,Path)):
        checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if 'standing_recovery_mode' in checkpoint:
        from .initial_goal_recovery import InitialGoalRecoveryPolicy
        with torch.inference_mode(False):
            return InitialGoalRecoveryPolicy(checkpoint,device,**kwargs)
    if 'predicted_target_solver_steps' in checkpoint:
        from .predicted_target_solver import PredictedTargetSolverPolicy
        # Refinement differentiates with respect to z, even when the caller
        # loads/runs its policy inside inference_mode. Keep actor tensors normal.
        with torch.inference_mode(False):
            return PredictedTargetSolverPolicy(checkpoint,device,**kwargs)
    if any(k in checkpoint for k in ['predicted_target_solver_backend','predicted_target_refinement_scope']):
        raise ValueError('Solver backend/scope requires a declared iteration count')
    return SoccerPolicy(checkpoint,device,**kwargs)


def recovery_controller_speed(task_speed, recovering, override=None):
    """Optional planner-only recovery command; never change task/scoring tensors."""
    if override is None:
        return None
    import math
    if not math.isfinite(override) or override <= 0:
        raise ValueError('Recovery speed must be finite and positive')
    if recovering.shape != task_speed.shape or recovering.dtype != torch.bool:
        raise ValueError('Expected one boolean recovery flag per task speed')
    return torch.where(recovering, torch.full_like(task_speed, override), task_speed)
