"""Versioned deployment bundle; old loaders cannot silently ignore task heads."""
from pathlib import Path
import torch
from .dual_task_latent import load_dual_task_policy

FORMAT='fiber_soccer_dual_task_v1'

def load_soccer_policy(checkpoint, device='cpu', **kwargs):
    if isinstance(checkpoint,(str,Path)):
        checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if checkpoint.get('format')!=FORMAT:
        raise ValueError('Expected an exported fiber_soccer_dual_task_v1 bundle')
    return load_dual_task_policy(checkpoint['policy'],device,**kwargs)
