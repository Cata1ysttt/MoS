"""Quantify target clipping and imitation error on first episodes only."""
import argparse
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
args = p.parse_args()
rows = torch.load(args.run / 'action_diagnostics.pt', weights_only=True)
names = json.loads((Path(__file__).resolve().parents[1] / 'assets/actor_meta.json').read_text())['joint_names']
groups = {'all': [], 'within_0.6m': [], 'beyond_0.6m': []}
for row in rows:
    distance = (row['ball_position'][:, :2] - row['robot_position'][:, :2]).norm(dim=-1)
    for label, selection in [('all', row['active']),
                             ('within_0.6m', row['active'] & (distance <= .6)),
                             ('beyond_0.6m', row['active'] & (distance > .6))]:
        if selection.any():
            effective = row['effective_targets'][selection]
            groups[label].append(torch.stack((
                effective - row['expert_targets'][selection],
                effective - row['bfm_targets'][selection]), dim=1))
report = {'note': '10 Hz first-episode samples; distance is horizontal ball-to-root distance, not contact phase.'}
for label, values in groups.items():
    if not values:
        report[label] = {'samples': 0}
        continue
    errors = torch.cat(values)
    report[label] = {'samples': len(errors),
                    'imitation_mse': errors[:, 0].square().mean().item(),
                    'clipping_mse': errors[:, 1].square().mean().item(),
                    'joints': {name: {'imitation_rmse_rad': errors[:, 0, j].square().mean().sqrt().item(),
                                     'clipping_rmse_rad': errors[:, 1, j].square().mean().sqrt().item(),
                                     'clipping_max_rad': errors[:, 1, j].abs().max().item()}
                               for j, name in enumerate(names)}}
(args.run / 'action_error_analysis.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: {a: b for a, b in v.items() if a != 'joints'} if isinstance(v, dict) else v
                  for k, v in report.items()}, indent=2))
