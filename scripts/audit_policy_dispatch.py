"""Verify both task calls and a mixed batch through one deployed policy instance.

Uses held-out recorded inputs, not a rollout or continuous skill-switch test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.planner import SCHEMA_V4, with_task_history
from fiber_soccer.planner_policy import SoccerPolicy

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--data', type=Path, nargs=2, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
policy = SoccerPolicy(a.checkpoint)
inputs = {}
for path in a.data:
    data = torch.load(path, map_location='cpu', weights_only=True)
    skill = data['metadata']['skill']
    assert skill not in inputs
    selected = []
    for tick in torch.linspace(0, len(data['rows'])-1, 4).long().tolist():
        row = data['rows'][tick]
        lane = len(row['x'])*3//4
        x = row['x'][lane:lane+1].clone()
        if policy.planner.schema == SCHEMA_V4:
            x = with_task_history(x, row['expert_observation'][lane:lane+1])
        selected.append(x)
    inputs[skill] = torch.cat(selected)
    del data
assert set(inputs) == {'pass','shoot'}
x = torch.cat([inputs[skill] for skill in ['pass','shoot']])
skills = torch.tensor([0]*len(inputs['pass'])+[1]*len(inputs['shoot']))
mixed = policy(x, skill=skills)
separate = [policy(inputs[skill], skill=skill) for skill in ['pass','shoot']]
errors = {}
for key, value in mixed.items():
    expected = torch.cat([result[key] for result in separate])
    torch.testing.assert_close(value, expected, atol=1e-4, rtol=1e-4)
    assert torch.isfinite(value).all()
    errors[key] = float((value-expected).abs().max())
torch.testing.assert_close(mixed['latent'].norm(dim=-1), torch.full((len(x),),16.))
assert all(not parameter.requires_grad for parameter in policy.parameters())
result = dict(checkpoint=str(a.checkpoint),
              checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              schema=policy.planner.schema, shared_policy_instances=1,
              samples_per_task={key:len(value) for key,value in inputs.items()},
              mixed_vs_separate_max_abs_error=errors,
              expert_policy_loaded=False, online_latent_optimization=False,
              continuous_switching_verified=False, rollout=False)
a.output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
