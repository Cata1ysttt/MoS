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
    """Experimental one-call controller; commands are already in the observation."""
    def __init__(self,checkpoint,device='cpu',**kwargs):
        steps=checkpoint.get('predicted_target_solver_steps')
        if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=64:
            raise ValueError('Missing or invalid predicted-target solver steps')
        if checkpoint.get('planner_arch')!='target_guided_latent_v1':
            raise ValueError('Refinement needs the shared internal target predictor')
        super().__init__(checkpoint,device,**kwargs)
        if any(value is not None for value in [self.transition,self.post_kick_delay,self.contact_skill_gate,self.post_pass_gate]):
            raise ValueError('Diagnostic refinement does not combine with transition gates')
        self.solver_steps=steps;self.solver_records=[]
        self.solver_backend=checkpoint.get('predicted_target_solver_backend','eager')
        if self.solver_backend not in ['eager','cuda_graph']:raise ValueError('Unknown latent solver backend')
        self.captured_solver=None
        if self.solver_backend=='cuda_graph':
            from .captured_latent_solver import CapturedLatentSolver
            weight=torch.ones(29,device=self.default.device);weight[:12]=2
            self.register_buffer('solver_root_weight',(weight/weight.mean()).sqrt())
            self.captured_solver=CapturedLatentSolver(self.actor,
                lambda raw:(self.default+(6*raw).clamp(-6,6)*self.scale)*self.solver_root_weight,steps)

    @torch.no_grad()
    def forward(self,observation):
        device=observation.device
        if device.type=='cuda':torch.cuda.synchronize(device)
        started=time.perf_counter()
        initial=super().forward(observation)
        desired=self.planner.target_predictor.bounded_targets(observation)
        if self.captured_solver is None:
            z,info=refine(self.actor,observation,initial['latent'],desired,self.default,self.scale,self.solver_steps)
        else:
            z,info=self.captured_solver.solve((observation[:,:64],observation[:,64:93],observation[:,93:930]),
                desired*self.solver_root_weight,initial['latent'])
        raw=self.actor(observation[:,:64],observation[:,64:93],observation[:,93:930],z)
        scaled=(6*raw).clamp(-6,6)
        if device.type=='cuda':torch.cuda.synchronize(device)
        self.solver_records.append(dict(batch_size=len(observation),latency_seconds=time.perf_counter()-started,**info))
        return dict(joint_targets=self.default+scaled*self.scale,action_scaled=scaled,action_normalized=raw,latent=z)
