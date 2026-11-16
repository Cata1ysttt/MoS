"""Author-supplied AH-BFM adapter with differentiable latent-to-action path."""
import importlib.util
import sys
from pathlib import Path
import torch
from .paths import ROOT


def load_native(device='cpu', package=None):
    package = Path(package or ROOT / 'AH-BFM')
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    spec = importlib.util.spec_from_file_location('fiber_author_ah_bfm', package/'load_ah_bfm.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_ah_bfm(package/'AH-BFM.pt', device=device).requires_grad_(False)


class NativeActor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state, last_action, history_actor, z):
        obs = dict(state=state, last_action=last_action, history_actor=history_actor)
        # Public inference API uses no_grad; training needs gradients to z,
        # while the author model parameters remain frozen.
        normalized = self.model._obs_normalizer(obs)
        return self.model._actor(normalized, z, .05).mean.float()
