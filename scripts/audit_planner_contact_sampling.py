"""Count observed contact windows in training lanes; no success claim."""
import argparse
import json
from pathlib import Path
import torch
from fiber_soccer.planner import observed_contact_window

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, nargs='+', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
result = []
for path in a.data:
    d = torch.load(path, map_location='cpu', weights_only=True)
    assert d['metadata']['schema'] in ('soccer_state_v2_950', 'soccer_state_v3_backward_1206')
    rows = d['rows']
    x = torch.stack([r['x'] for r in rows])
    ep = torch.stack([r['episode'] for r in rows])
    n = max(1, x.shape[1]*3//4)
    x, ep = x[:, :n], ep[:, :n]
    kicked = x[:, :, 940] > .5
    onset = torch.zeros_like(kicked)
    onset[1:] = kicked[1:] & ~kicked[:-1] & (ep[1:] == ep[:-1])
    contact = observed_contact_window(x, ep)
    reset = x[:, :, 93:930].abs().sum(-1) < 1e-8
    near = (x[:, :, 933:935].norm(dim=-1) < 1.) & ~kicked
    contact_fraction = float(contact.float().mean())
    reset_contact = float(contact[reset].float().mean()) if reset.any() else 0.
    result.append(dict(path=str(path), skill=d['metadata']['skill'], training_states=contact.numel(),
                       observed_kick_onsets=int(onset.sum()), contact_window_fraction=contact_fraction,
                       precontact_near_ball_fraction=float(near.float().mean()),
                       current_expected_contact_sampling_fraction=.75*contact_fraction+.25*reset_contact,
                       note='Window: 10 frames before to 5 after observed kicked phase, same episode; may include bad kicks'))
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
