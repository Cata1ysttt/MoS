"""Expert-assisted action inversion for capability diagnosis, never a policy result."""
import torch
from .lpp import project

SOLVER_VERSION = "sum_per_state_independent_stop_v2"


@torch.enable_grad()
def fit_latent(actor, state, target, previous, to_targets, iterations=64):
    z=torch.nn.Parameter(previous.detach().clone())
    optimizer=torch.optim.Adam([z],lr=.05)
    best=project(z.detach());best_error=torch.full((len(z),),float('inf'),device=z.device)
    active=torch.ones(len(z),dtype=torch.bool,device=z.device)
    state_updates=torch.zeros(len(z),dtype=torch.long,device=z.device)
    for step in range(iterations+1):
        output=actor(*state,project(z))
        error=(to_targets(output)-target).square().mean(-1)
        if not torch.isfinite(error).all():raise RuntimeError('Nonfinite latent oracle error')
        with torch.no_grad():
            improved=active & (error<best_error)
            best[improved]=project(z.detach())[improved]
            best_error=torch.where(active,torch.minimum(best_error,error.detach()),best_error)
            active &= best_error>=1e-5
        if step==0:initial=error.detach().clone()
        if step==iterations or not active.any():break
        # Optimize each independent latent with the same gradient scale,
        # regardless of how many environments are solved together.
        optimizer.zero_grad();(error*active).sum().backward();optimizer.step()
        state_updates+=active.long()
    return best,dict(initial_mse=float(initial.mean()),final_mse=float(best_error.mean()),max_mse=float(best_error.max()),updates=step,gradient_reduction='sum_per_state',solver_version=SOLVER_VERSION,min_state_updates=int(state_updates.min()),max_state_updates=int(state_updates.max()))
