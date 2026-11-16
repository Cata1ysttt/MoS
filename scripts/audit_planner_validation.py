"""Compare frozen planners on identical held-out states and decoded action labels."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.native_bfm import NativeActor, load_native
from fiber_soccer.paths import ASSETS
from fiber_soccer.planner import SCHEMA_V3, SCHEMA_V4, load_planner, with_task_history, balanced_skill_mean

p = argparse.ArgumentParser()
p.add_argument('--data',type=Path,nargs='+',required=True)
p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cpu')
p.add_argument('--sampling-horizon',type=int,default=5)
a = p.parse_args()
torch.set_num_threads(4)
pools = {}
for path in a.data:
    d = torch.load(path,map_location='cpu',weights_only=True)
    assert d['metadata']['schema'] == SCHEMA_V3
    rows = d['rows'];length = len(rows)-a.sampling_horizon+1
    assert length > 0
    cut = len(rows[0]['x'])*3//4
    values = [torch.stack([row[key][cut:] for row in rows[:length]]).flatten(0,1)
              for key in ['x','label_z','target','expert_observation']]
    values[3] = torch.nn.functional.pad(values[3],(0,1200-values[3].shape[-1]))
    skill = d['metadata']['skill']
    pools[skill] = [torch.cat((old,new)) for old,new in zip(pools[skill],values)] if skill in pools else values
assert set(pools) == {'pass','shoot'}
torch.manual_seed(812151)
selected, indices = [], {}
for skill in ['pass','shoot']:
    values = pools[skill]
    starts = torch.where(values[0][:,93:930].abs().sum(-1)<1e-8)[0]
    count = 64 if len(starts) else 0
    ids = torch.randint(len(values[0]),(256-count,))
    if count:
        ids = torch.cat((ids,starts[torch.randint(len(starts),(count,))]))
    indices[skill] = ids.tolist()
    selected.append([v[ids].to(a.device) for v in values])
x,z,target,histories = [torch.cat([v[i] for v in selected]) for i in range(4)]
actor = NativeActor(load_native(a.device))
default = torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=a.device)
scale = torch.tensor(json.loads((ASSETS/'actor_meta.json').read_text())['action_rescale'],device=a.device)*1.5
joint_weight = torch.ones(29,device=a.device);joint_weight[:12] = 2
results = {}
with torch.inference_mode():
    state = (x[:,:64],x[:,64:93],x[:,93:930])
    label = actor(*state,z).clamp(-1,1)
    confidence = 1/(1+(default+label*scale-target).square().mean(-1)/.01)
    for path in a.checkpoints:
        assert (path.parent/'complete.json').exists(), 'Use a completed training run'
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        model = load_planner(torch.load(path,map_location='cpu',weights_only=False),a.device)
        assert model.schema in (SCHEMA_V3,SCHEMA_V4)
        inputs = with_task_history(x,histories) if model.schema==SCHEMA_V4 else x
        action = actor(*state,model.act(inputs)).clamp(-1,1)
        error = (((action-label)*scale).square()*joint_weight).sum(-1)/joint_weight.sum()
        expert_error=(((default+action*scale-target).square())*joint_weight).sum(-1)/joint_weight.sum()
        phase = {}
        for skill, xb, eb, wb in zip(['pass','shoot'],x.chunk(2),error.chunk(2),confidence.chunk(2)):
            mask = (xb[:,939]>.5)&(xb[:,940]<.5)
            phase[skill] = dict(states=int(mask.sum()),weighted_action_mse=float((eb[mask]*wb[mask]).sum()/wb[mask].sum()) if mask.any() else None)
        results[path.parent.name] = dict(checkpoint=str(path),checkpoint_sha256=checksum,
            expert_target_weighted_action_mse=float(balanced_skill_mean(expert_error,confidence)),
            expert_target_task_weighted_action_mse={skill:float((e*w).sum()/w.sum()) for skill,e,w in zip(['pass','shoot'],expert_error.chunk(2),confidence.chunk(2))},
            uniform_action_mse=float(error.mean()),confidence_weighted_action_mse=float(balanced_skill_mean(error,confidence)),
            task_weighted_action_mse={skill:float((e*w).sum()/w.sum()) for skill,e,w in zip(['pass','shoot'],error.chunk(2),confidence.chunk(2))},
            launched_before_contact=phase)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==checksum
result = dict(validation_seed=812151,samples_per_task=256,sampling_horizon=a.sampling_horizon,
              reset_sample_fraction=.25,label_error_scale=.01,leg_weight=2,
              state_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
              data=[str(path) for path in a.data],indices=indices,results=results,
              note='Matched offline action fidelity, not an autonomous success measurement')
a.output.write_text(json.dumps(result,indent=2))
print(json.dumps(results,indent=2))
