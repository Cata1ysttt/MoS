"""Verify a completed skill adapter retains its source policy on the other skill."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.planner import load_planner,with_task_history,SCHEMA_V3,SCHEMA_V4
from fiber_soccer.native_bfm import NativeActor,load_native

p=argparse.ArgumentParser()
p.add_argument('--base',type=Path,required=True)
p.add_argument('--adapted',type=Path,required=True)
p.add_argument('--data',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cpu')
a=p.parse_args()
torch.set_num_threads(4)
for path in [a.base,a.adapted]:
    assert (path.parent/'complete.json').exists(),'Audit completed training checkpoints only'
headers=[torch.load(path,map_location='cpu',weights_only=False) for path in [a.base,a.adapted]]
# Policy construction must not change the simulator's later noise sequence.
rng=torch.random.get_rng_state()
base=load_planner(headers[0],a.device)
after_base=torch.random.get_rng_state()
torch.random.set_rng_state(rng)
adapted=load_planner(headers[1],a.device)
assert torch.equal(after_base,torch.random.get_rng_state()),'Adapter loading shifts global RNG'
assert base.adapter is None and adapted.adapter is not None
for attr in ['horizon','width','schema','flow_steps','noise_seed','command_conditioning']:
    assert getattr(base,attr)==getattr(adapted,attr),attr
for key,value in base.state_dict().items():
    assert torch.equal(value,adapted.state_dict()[key]),'Changed shared tensor: '+key
assert all(not p.requires_grad for name,p in adapted.named_parameters() if not name.startswith('adapter.'))
inputs=[]
for path in a.data:
    data=torch.load(path,map_location='cpu',weights_only=True)
    rows=data['rows'];n=len(rows[0]['x']);start=n*3//4
    assert start<n
    for tick in torch.linspace(0,len(rows)-1,min(8,len(rows))).long().tolist():
        row=rows[tick];x=row['x'][start:min(n,start+8)]
        if data['metadata']['schema']==SCHEMA_V3 and base.schema==SCHEMA_V4:
            x=with_task_history(x,row['expert_observation'][start:min(n,start+8)])
        else:assert data['metadata']['schema']==base.schema
        inputs.append(x)
x=torch.cat(inputs).to(a.device)
other_skill=x[:,-2:].argmax(-1)!=adapted.adapter_skill
assert other_skill.any() and (~other_skill).any(),'Supply both skills'
precontact=(~other_skill)&(x[:,940]<=.5) if adapted.adapter_post_kick_only else torch.zeros_like(other_skill)
protected=other_skill|precontact
actor=NativeActor(load_native(a.device))
with torch.inference_mode():
    original=base(x);corrected=adapted(x)
    assert torch.equal(original[protected],corrected[protected]),'Protected latent changed'
    state=(x[:,:64],x[:,64:93],x[:,93:930])
    old_action=actor(*state,original[:,0]).clamp(-1,1)
    new_action=actor(*state,corrected[:,0]).clamp(-1,1)
    assert torch.equal(old_action[protected],new_action[protected]),'Protected actor output changed'
    outside=torch.zeros_like(protected)
    if adapted.adapter_ball_distance_gate is not None:
        outside=(~protected)&(x[:,933:935].norm(dim=-1)>=adapted.adapter_ball_distance_gate[1])
        assert torch.equal(original[outside],corrected[outside]),'Outside-gate latent changed'
        assert torch.equal(old_action[outside],new_action[outside]),'Outside-gate actor output changed'
result=dict(base=str(a.base),adapted=str(a.adapted),
    base_sha256=hashlib.sha256(a.base.read_bytes()).hexdigest(),adapted_sha256=hashlib.sha256(a.adapted.read_bytes()).hexdigest(),
    states=len(x),protected_precontact_states=int(precontact.sum()),adapter_post_kick_only=adapted.adapter_post_kick_only,protected_states=int(protected.sum()),outside_gate_states=int(outside.sum()),outside_gate_outputs_exact=True if outside.any() else None,adapter_skill=adapted.adapter_skill,adapter_space=adapted.adapter_space,
    inputs_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),shared_tensors_exact=True,
    protected_latents_exact=True,protected_actor_outputs_exact=True,loading_rng_preserved=True,
    adapted_skill_latent_rms_change=float((original[~protected]-corrected[~protected]).square().mean().sqrt()) if (~protected).any() else None,
    scope='Fixed held-out rows; proves structural preservation and sampled numerical identity, not task success or continuous skill switching.')
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
