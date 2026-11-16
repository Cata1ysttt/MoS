"""Compare original and regularized latent labels before observed training contacts."""
import argparse
import json
from pathlib import Path
import torch
from fiber_soccer.native_bfm import NativeActor, load_native
from fiber_soccer.paths import ASSETS
from fiber_soccer.planner import observed_contact_window

p = argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a = p.parse_args()
torch.set_num_threads(4)
data = torch.load(a.data,map_location='cpu',weights_only=True)
rows = data['rows']
x = torch.stack([row['x'] for row in rows])
episode = torch.stack([row['episode'] for row in rows])
precontact = observed_contact_window(x,episode,before=25,after=0) & (x[:,:,940]<.5)
teacher = torch.stack([row['teacher_executed'] for row in rows])
actor = NativeActor(load_native('cpu'))
default = torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'])
scale = torch.tensor(json.loads((ASSETS/'actor_meta.json').read_text())['action_rescale'])*1.5
raw = torch.stack([row['previous_label_z'] for row in rows])
semantic = torch.stack([row['label_z'] for row in rows])
target = torch.stack([row['target'] for row in rows])
results = {}
with torch.inference_mode():
    for flag in [False,True]:
        mask = precontact & (teacher==flag)
        states, before, after, desired = x[mask],raw[mask],semantic[mask],target[mask]
        values = []
        for start in range(0,len(states),512):
            xb = states[start:start+512]
            inputs = (xb[:,:64],xb[:,64:93],xb[:,93:930])
            y0 = default+actor(*inputs,before[start:start+512]).clamp(-1,1)*scale
            y1 = default+actor(*inputs,after[start:start+512]).clamp(-1,1)*scale
            t = desired[start:start+512]
            values.append(torch.stack(((y0-t).square().mean(-1),(y1-t).square().mean(-1),
                                      (y0[:,:12]-t[:,:12]).square().mean(-1),
                                      (y1[:,:12]-t[:,:12]).square().mean(-1),
                                      (y1-y0).square().mean(-1)),-1))
        if not values:
            continue
        v = torch.cat(values)
        names = ['raw_action_mse','regularized_action_mse','raw_leg_mse','regularized_leg_mse','decoded_change_mse']
        results['teacher_executed' if flag else 'student_executed'] = dict(states=len(v),
            **{name:dict(mean=float(v[:,i].mean()),median=float(v[:,i].median()),p90=float(v[:,i].quantile(.9))) for i,name in enumerate(names)})
result = dict(data=str(a.data),groups=results,
              note='Training states in the 25 frames before observed contact, excluding contact and episode resets; not an autonomous task result')
a.output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
