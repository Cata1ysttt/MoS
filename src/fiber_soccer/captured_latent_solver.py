"""Fixed-length latent Adam loop suitable for CUDA graph replay.

Retains the independent stopping mask and best latent from the reference solver.
Inactive rows still execute kernels, but cannot change their returned best value.
Only latent variables receive gradients; this does not train model parameters.
"""
import torch
from .lpp import project


@torch.enable_grad()
def fixed_adam(actor,state,target,previous,to_targets,iterations=64):
    z=previous.detach().clone().requires_grad_(True)
    moment=torch.zeros_like(z);variance=torch.zeros_like(z)
    best=project(z.detach());best_error=torch.full((len(z),),float('inf'),device=z.device)
    active=torch.ones(len(z),dtype=torch.bool,device=z.device)
    updates=torch.zeros(len(z),dtype=torch.long,device=z.device)
    finite=torch.ones((),dtype=torch.bool,device=z.device)
    for step in range(iterations+1):
        error=(to_targets(actor(*state,project(z)))-target).square().mean(-1)
        with torch.no_grad():
            finite=finite & torch.isfinite(error).all()
            improved=active & (error<best_error)
            best=torch.where(improved[:,None],project(z.detach()),best)
            best_error=torch.where(active,torch.minimum(best_error,error.detach()),best_error)
            active=active & (best_error>=1e-5)
            if step==0:initial=error.detach().clone()
        if step==iterations:break
        grad,=torch.autograd.grad((error*active).sum(),z)
        with torch.no_grad():
            # Match the installed reference Adam's foreach operation order.
            # Ordinary elementwise expressions can accumulate different rounding
            # and diverge on this actor's nonlinear latent-to-action path.
            torch._foreach_lerp_([moment],[grad],1-.9)
            torch._foreach_mul_([variance],.999)
            torch._foreach_addcmul_([variance],[grad],[grad],1-.999)
            denom=torch._foreach_sqrt([variance])
            torch._foreach_div_(denom,[(1-.999**(step+1))**.5])
            torch._foreach_add_(denom,1e-8)
            torch._foreach_addcdiv_([z],[moment],denom,[-.05/(1-.9**(step+1))])
            updates.add_(active)
    return best,initial,best_error,updates,finite


class CapturedLatentSolver:
    """Cache one CUDA graph per batch shape for a frozen actor and target transform."""
    def __init__(self,actor,to_targets,iterations=64):
        if isinstance(iterations,bool) or not isinstance(iterations,int) or not 1<=iterations<=64:
            raise ValueError('Expected 1..64 refinement steps')
        if any(p.requires_grad for p in actor.parameters()):raise ValueError('Actor must be frozen')
        self.actor=actor;self.to_targets=to_targets;self.iterations=iterations;self.cache={}

    def solve(self,state,target,previous):
        if previous.device.type!='cuda':raise ValueError('CUDA graph solver requires CUDA inputs')
        key=(tuple(previous.shape),previous.device,previous.dtype)
        with torch.inference_mode(False),torch.enable_grad():
            if key not in self.cache:
                buffers=tuple(v.detach().clone() for v in (*state,target,previous))
                stream=torch.cuda.Stream(device=previous.device)
                stream.wait_stream(torch.cuda.current_stream(previous.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):fixed_adam(self.actor,buffers[:-2],buffers[-2],buffers[-1],self.to_targets,self.iterations)
                torch.cuda.current_stream(previous.device).wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    outputs=fixed_adam(self.actor,buffers[:-2],buffers[-2],buffers[-1],self.to_targets,self.iterations)
                self.cache[key]=(buffers,graph,outputs)
            buffers,graph,outputs=self.cache[key]
            for dest,value in zip(buffers,(*state,target,previous)):
                if dest.shape!=value.shape or dest.device!=value.device or dest.dtype!=value.dtype:
                    raise ValueError('State/target shape, device, or dtype changed within a batch shape')
                dest.copy_(value)
            graph.replay()
            best,initial,error,updates,finite=outputs
            if not bool(finite):raise RuntimeError('Nonfinite captured latent solver error')
            return best.detach().clone(),dict(initial_mse=float(initial.mean()),final_mse=float(error.mean()),
                max_mse=float(error.max()),updates=int(updates.max()),gradient_reduction='sum_per_state',
                solver_version='fixed_adam_cuda_graph_v1',min_state_updates=int(updates.min()),max_state_updates=int(updates.max()))
