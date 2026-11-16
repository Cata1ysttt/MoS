"""Measure task-gradient alignment for decoded-action fitting on training states.

A local gradient diagnostic is not proof of multi-task interference or a rollout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.planner import load_planner, with_task_history, observed_contact_window, SCHEMA_V3, SCHEMA_V4
from fiber_soccer.flow_planner import FlowLatentPlanner
from fiber_soccer.native_bfm import NativeActor, load_native
from fiber_soccer.paths import ASSETS

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--data',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cuda:0')
p.add_argument('--batches',type=int,default=16)
a=p.parse_args()
assert a.batches>0 and (a.checkpoint.parent/'complete.json').exists()
checkpoint_hash=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
torch.set_num_threads(4)
model=load_planner(torch.load(a.checkpoint,map_location='cpu',weights_only=False),a.device)
assert isinstance(model,FlowLatentPlanner) and model.schema==SCHEMA_V4
actor=NativeActor(load_native(a.device))
scale=torch.tensor(json.loads((ASSETS/'actor_meta.json').read_text())['action_rescale'],device=a.device)*1.5
weight=torch.ones(29,device=a.device);weight[:12]=2
pools={}
for path in a.data:
    d=torch.load(path,map_location='cpu',weights_only=True);rows=d['rows']
    assert d['metadata']['schema']==SCHEMA_V3
    cut=len(rows[0]['x'])*3//4;length=len(rows)-model.horizon+1
    base=torch.stack([r['x'][:cut] for r in rows])
    episode=torch.stack([r['episode'][:cut] for r in rows])
    contact=observed_contact_window(base,episode)[:length].flatten()
    history=torch.stack([r['expert_observation'][:cut] for r in rows[:length]])
    x=with_task_history(base[:length],history).flatten(0,1)
    z=torch.stack([r['label_z'][:cut] for r in rows[:length]]).flatten(0,1)
    values=[x,z,contact]
    skill=d['metadata']['skill']
    pools[skill]=[torch.cat((old,new)) for old,new in zip(pools[skill],values)] if skill in pools else values
assert set(pools)=={'pass','shoot'}
indices={skill:dict(reset=torch.where(x[:,93:930].abs().sum(-1)<1e-8)[0],contact=torch.where(c)[0]) for skill,(x,z,c) in pools.items()}
params=list(model.named_parameters())
last=f'field.{len(model.field)-1}.'
groups={'all':list(range(len(params))),'output_head':[i for i,(name,_) in enumerate(params) if name.startswith(last)],'shared_features':[i for i,(name,_) in enumerate(params) if not name.startswith(last)]}
generator=torch.Generator().manual_seed(813151)
records=[]
for batch in range(a.batches):
    gradients=[];losses={};saturation={}
    for skill in ['pass','shoot']:
        x,z,_=pools[skill]
        chosen=[];remaining=64
        for phase in ['reset','contact']:
            ids=indices[skill][phase]
            if len(ids):
                chosen.append(ids[torch.randint(len(ids),(16,),generator=generator)]);remaining-=16
        chosen.append(torch.randint(len(x),(remaining,),generator=generator));ids=torch.cat(chosen)
        xb=x[ids].to(a.device);zb=z[ids].to(a.device)
        state=(xb[:,:64],xb[:,64:93],xb[:,93:930])
        with torch.no_grad():target=actor(*state,zb).clamp(-1,1)
        raw=actor(*state,model.act(xb));prediction=raw.clamp(-1,1)
        errors=((prediction-target)*scale).square()*weight
        saturated=raw.abs()>1
        near=(xb[:,933:935].norm(dim=-1)<=.6)&(xb[:,939]>.5)&(xb[:,940]<.5)
        saturation[skill]={}
        for phase,mask in [('all',torch.ones(len(xb),dtype=torch.bool,device=xb.device)),('near',near)]:
            e=errors[mask];dead=saturated[mask]
            saturation[skill][phase]=dict(states=int(mask.sum()),saturated_joint_fraction=float(dead.float().mean()) if mask.any() else None,weighted_error_sum=float(e.sum().detach()),saturated_weighted_error_sum=float((e*dead).sum().detach()))
        loss=errors.sum(-1).div(weight.sum()).mean()
        gradients.append([g.detach() for g in torch.autograd.grad(loss,[p for _,p in params])]);losses[skill]=float(loss)
    alignment={}
    for label,ids in groups.items():
        dot=sum((gradients[0][i]*gradients[1][i]).sum() for i in ids)
        norms=[sum(g[i].square().sum() for i in ids).sqrt() for g in gradients]
        alignment[label]=dict(cosine=float(dot/(norms[0]*norms[1]).clamp_min(1e-20)),pass_norm=float(norms[0]),shoot_norm=float(norms[1]))
    records.append(dict(batch=batch,loss=losses,alignment=alignment,saturation=saturation))
summary={label:dict(mean_cosine=sum(r['alignment'][label]['cosine'] for r in records)/len(records),negative_batches=sum(r['alignment'][label]['cosine']<0 for r in records),batches=len(records)) for label in groups}
saturation_summary={}
for skill in ['pass','shoot']:
    saturation_summary[skill]={}
    for phase in ['all','near']:
        rows=[r['saturation'][skill][phase] for r in records]
        total=sum(r['weighted_error_sum'] for r in rows)
        saturation_summary[skill][phase]=dict(states=sum(r['states'] for r in rows),fraction_error_at_saturated_joints=sum(r['saturated_weighted_error_sum'] for r in rows)/max(total,1e-20))
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==checkpoint_hash
a.output.write_text(json.dumps(dict(checkpoint=str(a.checkpoint),checkpoint_sha256=checkpoint_hash,data=[str(path) for path in a.data],saturation=saturation_summary,seed=813151,training_lanes_only=True,objective='Decoded regularized-label action MSE; leg weight 2; no flow loss',sampling='64/task; 25% reset and 25% contact when available; remaining uniform',summary=summary,records=records),indent=2))
print(json.dumps(dict(alignment=summary,saturation=saturation_summary),indent=2))
