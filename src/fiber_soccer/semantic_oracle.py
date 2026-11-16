"""Experimental semantic regularization for training labels, never deployment."""
import torch
from .lpp import project


@torch.enable_grad()
def fit_regularized_latent(actor,state,target,initial,prior,to_targets,weight=.001,iterations=64):
    if weight<0:raise ValueError('Semantic weight must be nonnegative')
    z=torch.nn.Parameter(initial.detach().clone());prior=prior.detach()
    optimizer=torch.optim.Adam([z],lr=.05)
    best=project(z.detach());best_value=torch.full((len(z),),float('inf'),device=z.device)
    for step in range(iterations+1):
        latent=project(z)
        action=(to_targets(actor(*state,latent))-target).square().mean(-1)
        semantic=(latent-prior).square().mean(-1)
        objective=action+weight*semantic
        if not torch.isfinite(objective).all():raise RuntimeError('Nonfinite semantic inversion')
        with torch.no_grad():
            improve=objective<best_value;best[improve]=latent.detach()[improve];best_value=torch.minimum(best_value,objective.detach())
        if step==iterations:break
        # Each row owns independent latent parameters; batch averaging changes
        # Adam updates through epsilon and makes labels depend on chunk size.
        optimizer.zero_grad();objective.sum().backward();optimizer.step()
    with torch.no_grad():
        action=(to_targets(actor(*state,best))-target).square().mean()
        semantic=(best-prior).square().mean()
    return best,dict(action_mse=float(action),semantic_mse=float(semantic),objective=float(best_value.mean()),weight=weight,iterations=iterations,gradient_reduction='sum_per_state')
