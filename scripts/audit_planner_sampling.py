"""Audit pre-contact approach coverage under contact-biased training."""
import argparse
import json
from pathlib import Path
import torch
from fiber_soccer.planner import SCHEMA_V3, observed_contact_window

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, nargs='+', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--radius', type=float, default=.6)
p.add_argument('--horizon', type=int, default=5)
p.add_argument('--contact-fraction', type=float, default=.25)
a = p.parse_args()
assert a.radius > 0 and a.horizon > 0 and 0 <= a.contact_fraction <= .75
torch.set_num_threads(4)
pools = {}
for path in a.data:
    data = torch.load(path, map_location='cpu', weights_only=True)
    assert data['metadata']['schema'] == SCHEMA_V3
    rows = data['rows']
    cut = len(rows[0]['x'])*3//4
    length = len(rows)-a.horizon+1
    assert length > 0
    x = torch.stack([r['x'][:cut] for r in rows])
    episodes = torch.stack([r['episode'][:cut] for r in rows])
    contact = observed_contact_window(x, episodes)[:length].flatten()
    x = x[:length].flatten(0,1)
    near = (x[:,933:935].norm(dim=-1) <= a.radius) & (x[:,939] > .5) & (x[:,940] < .5)
    reset = x[:,93:930].abs().sum(-1) < 1e-8
    teacher = torch.stack([r['teacher_executed'][:cut] for r in rows[:length]]).flatten().bool()
    skill = data['metadata']['skill']
    values = [near, reset, contact, teacher]
    pools[skill] = [torch.cat((old,new)) for old,new in zip(pools[skill], values)] if skill in pools else values
results = {}
for skill, (near, reset, contact, teacher) in pools.items():
    # Match 64-state minibatches: 16 reset, int(64*fraction) contact, rest uniform.
    reset_weight = .25 if reset.any() else 0.
    contact_weight = int(64*a.contact_fraction)/64 if contact.any() else 0.
    probability = torch.full((len(near),), (1-reset_weight-contact_weight)/len(near), dtype=torch.float64)
    if reset_weight:
        probability[reset] += reset_weight/int(reset.sum())
    if contact_weight:
        probability[contact] += contact_weight/int(contact.sum())
    assert abs(float(probability.sum())-1) < 1e-10
    groups = {}
    for label, mask in [('all_near',near),('near_outside_contact_window',near & ~contact),
                        ('student_near_outside_contact_window',near & ~contact & ~teacher),
                        ('teacher_near',near & teacher),('student_near',near & ~teacher)]:
        groups[label] = dict(states=int(mask.sum()),uniform_fraction=float(mask.float().mean()),
                            expected_minibatch_fraction=float(probability[mask].sum()))
    results[skill] = dict(training_states=len(near),reset_states=int(reset.sum()),contact_states=int(contact.sum()),groups=groups)
report = dict(data=[str(path) for path in a.data], radius=a.radius, horizon=a.horizon,
              contact_fraction=a.contact_fraction, results=results,
              note='Training lanes only. Near means launched, not yet kicked, horizontal ball-to-root distance within radius. Outside contact window does not by itself prove a failed episode.')
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps(results,indent=2))
