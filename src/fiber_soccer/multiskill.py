"""Explicit skill conditioning for the pass/shoot action-only pilot."""
import torch
import torch.nn.functional as F

SKILLS = ('pass', 'shoot')
OBS_WIDTH = 1200


def condition(observation, skill):
    """Pad heterogeneous expert observations and append an explicit skill label.

    This is a privileged-state pilot, not the FIBER perceptive flow planner.
    """
    if observation.shape[-1] > OBS_WIDTH:
        raise ValueError(f'Unsupported observation width {observation.shape[-1]}')
    index = SKILLS.index(skill)
    label = observation.new_zeros(observation.shape[0], len(SKILLS))
    label[:, index] = 1
    return torch.cat((F.pad(observation, (0, OBS_WIDTH-observation.shape[-1])), label), -1)


class ConditionedProjector(torch.nn.Module):
    def __init__(self, projector, skill):
        super().__init__()
        self.projector = projector
        self.skill = skill

    def forward(self, observation):
        return self.projector(condition(observation, self.skill))


class WarmStartedJointProjector(torch.nn.Module):
    """Shared s1 trunk with a learned task input offset; preserves pass at init.

    Keep the original 1200D LayerNorm. Appending skill bits inside that norm
    changes an existing policy even when new linear columns are zero.
    """
    def __init__(self):
        super().__init__()
        from .lpp import Projector
        self.base = Projector(OBS_WIDTH)
        self.skill_offset = torch.nn.Parameter(torch.zeros(len(SKILLS), OBS_WIDTH))

    def forward(self, observation):
        return self.base(observation[:, :OBS_WIDTH] + observation[:, OBS_WIDTH:] @ self.skill_offset)


def load_projector(checkpoint, device):
    from .lpp import Projector
    if checkpoint.get('projector_arch') == 'warm_shared_v2':
        model = WarmStartedJointProjector()
    else:
        model = Projector(checkpoint['obs_dim'])
    model.load_state_dict(checkpoint['projector'])
    return model.to(device)
