import json
from types import SimpleNamespace
import numpy as np
import torch
from fiber_soccer.planner_collection import expert_observation_layout


def test_expert_observation_layout_serializes_numpy_dimensions():
    manager = SimpleNamespace(
        active_terms={'policy': ['velocity', 'position'], 'rma_obs': ['ball_obs']},
        group_obs_term_dim={'policy': [(np.int64(30),), (np.int64(290),)],
                            'rma_obs': [(np.int64(150),)]})
    obs = {'policy': torch.zeros(2, 320), 'rma_obs': torch.zeros(2, 150)}
    restored = json.loads(json.dumps(expert_observation_layout(manager, obs)))
    assert restored['policy'] == dict(width=320, terms=['velocity', 'position'],
                                      term_shapes=[[30], [290]])
    assert restored['rma_obs']['term_shapes'] == [[150]]
