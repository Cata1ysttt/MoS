"""Validate active expert assets and ONNX inference without launching physics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml

root = Path(__file__).resolve().parents[1]
config = json.loads((root / 'config/soccer_experts.json').read_text())['pass']
checkpoint = root / config['checkpoint']
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == config['checkpoint_sha256']
package = checkpoint.parent
old = Path('/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/params/env.yaml')
new = yaml.load((package / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
previous = yaml.load(old.read_text(), Loader=yaml.BaseLoader)
contracts = {key: new[key] == previous[key] for key in ['observations', 'actions', 'sim', 'decimation']}
assert all(contracts.values()), contracts
options = ort.SessionOptions()
options.intra_op_num_threads = 2
sessions = []
for path in [root / config['onnx'], package / '2026-11-16_13-44-27_rolling_pass_v1_stage2.onnx']:
    onnx.checker.check_model(str(path))
    sessions.append(ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider']))
inputs = {item.name: item.shape for item in sessions[0].get_inputs()}
assert inputs == {'obs': [1, 1100], 'rma_obs': [1, 100], 'time_step': [1, 1]}, inputs
rng = np.random.default_rng(818201)
max_export_difference = 0.
max_time_difference = 0.
max_action = 0.
for _ in range(64):
    feed = {name: rng.normal(0, .2, shape).astype(np.float32) for name, shape in inputs.items()}
    feed['time_step'].fill(0)
    a, b = [session.run(['actions'], feed)[0] for session in sessions]
    assert a.shape == (1, 29) and np.isfinite(a).all() and np.isfinite(b).all()
    max_export_difference = max(max_export_difference, float(np.max(np.abs(a-b))))
    max_action = max(max_action, float(np.max(np.abs(a))))
    feed['time_step'].fill(1)
    other = sessions[0].run(['actions'], feed)[0]
    max_time_difference = max(max_time_difference, float(np.max(np.abs(a-other))))
assert max_export_difference < 1e-5
result = dict(checkpoint_sha256=config['checkpoint_sha256'], inputs=inputs, action_shape=[1, 29],
              matching_saved_contracts=contracts, samples=64, max_export_action_difference=max_export_difference,
              max_time_zero_one_action_difference=max_time_difference, max_abs_action=max_action,
              scope='Synthetic input/interface audit only; not closed-loop performance or PT/ONNX equivalence.')
output = root / 'reports/pass_expert_20261116_interface.json'
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
