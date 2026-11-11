"""Expert-assisted action inversion for capability diagnosis, never a policy result."""
import torch
from .lpp import project


@torch.enable_grad()
def fit_latent(actor, state, target, previous, to_targets, iterations=64):
    z=torch.nn.Parameter(previous.detach().clone())
    optimizer=torch.optim.Adam([z],lr=.05)
    best=project(z.detach());best_error=torch.full((len(z),),float('inf'),device=z.device)
    for step in range(iterations+1):
        output=actor(*state,project(z))
        error=(to_targets(output)-target).square().mean(-1)
        if not torch.isfinite(error).all():raise RuntimeError('Nonfinite latent oracle error')
        with torch.no_grad():
            improved=error<best_error
            best[improved]=project(z.detach())[improved]
            best_error=torch.minimum(best_error,error.detach())
        if step==0:initial=error.detach().clone()
        if step==iterations or best_error.max()<1e-5:break
        optimizer.zero_grad();error.mean().backward();optimizer.step()
    return best,dict(initial_mse=float(initial.mean()),final_mse=float(best_error.mean()),max_mse=float(best_error.max()),updates=step)
