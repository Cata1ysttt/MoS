"""Measure expert-action inversion fidelity separately on student/teacher states."""
import argparse
import json
from pathlib import Path
import torch
from fiber_soccer.native_bfm import NativeActor, load_native
from fiber_soccer.paths import ASSETS

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
torch.set_num_threads(4)
data = torch.load(args.data, map_location='cpu', weights_only=True)
actor = NativeActor(load_native(args.device))
default = torch.tensor(json.loads((ASSETS / 'pickup_scene.json').read_text())['default_q'], device=args.device)
scale = torch.tensor(json.loads((ASSETS / 'actor_meta.json').read_text())['action_rescale'], device=args.device) * 1.5
samples = {False: [], True: []}
with torch.inference_mode():
    for start in range(0, len(data['rows']), 16):
        rows = data['rows'][start:start + 16]
        x = torch.cat([row['x'] for row in rows]).to(args.device)
        label = torch.cat([row.get('label_z', row['z']) for row in rows]).to(args.device)
        executed = torch.cat([row['z'] for row in rows]).to(args.device)
        target = torch.cat([row['target'] for row in rows]).to(args.device)
        teacher = torch.cat([row['teacher_executed'] for row in rows])
        state = (x[:, :64], x[:, 64:93], x[:, 93:930])
        predicted = default + actor(*state, label).clamp(-1, 1) * scale
        actual = default + actor(*state, executed).clamp(-1, 1) * scale
        error = torch.stack(((predicted - target).square().mean(-1),
                             (actual - target).square().mean(-1)), -1).cpu()
        for flag in samples:
            samples[flag].append(error[teacher == flag])
result = dict(metadata=data['metadata'], note='Training-state audit, not autonomous success', groups={})
for flag, parts in samples.items():
    values = torch.cat(parts)
    if not len(values):
        continue
    result['groups']['teacher_executed' if flag else 'student_executed'] = dict(
        states=len(values), label_mse_mean=float(values[:, 0].mean()),
        executed_mse_mean=float(values[:, 1].mean()),
        label_mse_quantiles={str(q): float(values[:, 0].quantile(q)) for q in [.5, .9, .99]},
        fraction_label_mse_above_001=float((values[:, 0] > .01).float().mean()))
args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result['groups'], indent=2))
