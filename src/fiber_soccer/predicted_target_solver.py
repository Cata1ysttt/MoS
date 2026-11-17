"""Diagnostic latent refinement toward a policy's own predicted targets.

This executes no expert targets and never changes actor weights. It deliberately
uses online optimization to test whether the internal target predictor can drive
closed-loop skills. It is not evidence of a real-time feedforward controller.
"""
import time
import torch
from .latent_oracle import fit_latent
from .planner_policy import SoccerPolicy


def refine(actor, x, previous, desired, default, scale, iterations):
    if isinstance(iterations,bool) or not isinstance(iterations,int) or not 1<=iterations<=64:
        raise ValueError('Expected 1..64 diagnostic refinement steps')
    if any(p.requires_grad for p in actor.parameters()):
        raise ValueError('Diagnostic refinement requires a frozen actor')
    with torch.inference_mode(False),torch.enable_grad():
        # Native evaluators may wrap the entire control loop in inference mode.
        state=tuple(v.detach().clone() for v in [x[:,:64],x[:,64:93],x[:,93:930]])
        previous=previous.detach().clone();desired=desired.detach().clone()
        default=default.detach().clone();scale=scale.detach().clone()
        weight=torch.ones(29,device=x.device);weight[:12]=2
        root_weight=(weight/weight.mean()).sqrt()
        return fit_latent(actor,state,desired*root_weight,previous,
            lambda raw:(default+(6*raw).clamp(-6,6)*scale)*root_weight,iterations=iterations)


class PredictedTargetSolverPolicy(SoccerPolicy):
    """Experimental one-call controller using the shared observation/command interface."""
    def __init__(self,checkpoint,device='cpu',**kwargs):
        if type(self) is PredictedTargetSolverPolicy and 'standing_recovery_mode' in checkpoint:
            raise ValueError('Use load_soccer_policy for standing recovery checkpoints')
        steps=checkpoint.get('predicted_target_solver_steps')
        if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=64:
            raise ValueError('Missing or invalid predicted-target solver steps')
        if checkpoint.get('planner_arch')!='target_guided_latent_v1':
            raise ValueError('Refinement needs the shared internal target predictor')
        super().__init__(checkpoint,device,**kwargs)
        if any(value is not None for value in [self.transition,self.post_kick_delay,self.contact_skill_gate,self.post_pass_gate]):
            raise ValueError('Diagnostic refinement does not combine with transition gates')
        self.solver_steps=steps;self.solver_records=[]
        self.refinement_scope=checkpoint.get('predicted_target_refinement_scope','all')
        from .refinement_scope import RefinementScope
        self.refinement_gate=RefinementScope(self.refinement_scope).eval() if self.refinement_scope!='all' else None
        self.solver_backend=checkpoint.get('predicted_target_solver_backend','eager')
        if self.solver_backend not in ['eager','cuda_graph']:raise ValueError('Unknown latent solver backend')
        self.captured_solver=None
        if self.solver_backend=='cuda_graph':
            from .captured_latent_solver import CapturedLatentSolver
            weight=torch.ones(29,device=self.default.device);weight[:12]=2
            self.register_buffer('solver_root_weight',(weight/weight.mean()).sqrt())
            self.captured_solver=CapturedLatentSolver(self.actor,
                lambda raw:(self.default+(6*raw).clamp(-6,6)*self.scale)*self.solver_root_weight,steps)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if self.refinement_gate is not None:self.refinement_gate.reset(env_ids)

    def _refine(self,observation,initial,desired):
        if self.captured_solver is None:
            return refine(self.actor,observation,initial,desired,self.default,self.scale,self.solver_steps)
        return self.captured_solver.solve((observation[:,:64],observation[:,64:93],observation[:,93:930]),
            desired*self.solver_root_weight,initial)

    @torch.no_grad()
    def forward(self,observation,skill=None,target_direction=None,target_speed=None):
        device=observation.device
        if device.type=='cuda':torch.cuda.synchronize(device)
        started=time.perf_counter()
        observation=self.condition_observation(observation,skill,target_direction,target_speed)
        if self.refinement_gate is None:
            initial=super().forward(observation)
            desired=self.planner.target_predictor.bounded_targets(observation)
            z,info=self._refine(observation,initial['latent'],desired)
            refined=len(observation)
        else:
            active=self.refinement_gate(observation[:,-2:].argmax(-1))
            prior=self.planner.prior.act(observation)
            refined=int(active.sum())
            if refined:
                desired=self.planner.target_predictor.bounded_targets(observation)
                initial=self.planner.decode(observation,prior,desired)
                proposed,info=self._refine(observation,initial,desired)
                z=torch.where(active[:,None],proposed,prior)
                info['solver_metrics_scope']='proposed_refinement_before_scope_selection'
            else:
                z=prior
                info=dict(initial_mse=None,final_mse=None,max_mse=None,updates=0,
                    min_state_updates=0,max_state_updates=0,solver_version='shared_prior_only',
                    gradient_reduction='none',solver_metrics_scope='refinement_not_executed')
        raw=self.actor(observation[:,:64],observation[:,64:93],observation[:,93:930],z)
        scaled=(6*raw).clamp(-6,6)
        if device.type=='cuda':torch.cuda.synchronize(device)
        self.solver_records.append(dict(batch_size=len(observation),refined_environments=refined,
            refinement_scope=self.refinement_scope,latency_seconds=time.perf_counter()-started,**info))
        return dict(joint_targets=self.default+scaled*self.scale,action_scaled=scaled,action_normalized=raw,latent=z)
