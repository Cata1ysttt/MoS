"""Compare latent planners and saved labels on identical training or held-out precontact states."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.native_bfm import NativeActor, load_native
from fiber_soccer.paths import ASSETS
from fiber_soccer.planner import load_planner, with_task_history, trajectory_event_ids, observed_contact_window, SCHEMA_V4

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,nargs='+',required=True)
p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cpu')
p.add_argument('--split',choices=['train','valid'],default='valid',help='First three quarters or last quarter of recorded environment lanes')
a=p.parse_args()
assert not a.output.exists(),'Preserve previous audit'
torch.set_num_threads(4)
selected=[]
for path in a.data:
    assert (path.parent/'complete.json').exists()
    data=torch.load(path,map_location='cpu',weights_only=True)
    assert data['metadata']['skill']=='pass'
    rows=data['rows'];cut=len(rows[0]['x'])*3//4
    lanes=slice(None,cut) if a.split=='train' else slice(cut,None)
    x=torch.stack([row['x'][lanes] for row in rows])
    history=torch.stack([row['expert_observation'][lanes] for row in rows])
    x=with_task_history(x,history)
    episodes=torch.stack([row['episode'][lanes] for row in rows])
    present=['ball_episode' in row for row in rows]
    assert all(present) or not any(present)
    ball=torch.stack([row['ball_episode'][lanes] for row in rows]) if all(present) else None
    events=trajectory_event_ids(x,episodes,ball)
    mask=observed_contact_window(x,events,before=10,after=0)&(x[:,:,940]<.5)
    values=[x]+[torch.stack([row[key][lanes] for row in rows]) for key in ['label_z','target','teacher_executed']]
    selected.append([value[mask] for value in values])
x,z,target,teacher=[torch.cat([values[i] for values in selected]) for i in range(4)]
assert len(x)>0
meta=json.loads((ASSETS/'actor_meta.json').read_text())
actor=NativeActor(load_native(a.device))
default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=a.device)
scale=torch.tensor(meta['action_rescale'],device=a.device)*1.5
weights=torch.ones(29);weights[:12]=2

def measure(model):
    errors=[];clipped=[];commands=[];latents=[]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            xb=x[start:start+128].to(a.device)
            latent=z[start:start+128].to(a.device) if model is None else model.act(xb)
            action=actor(xb[:,:64],xb[:,64:93],xb[:,93:930],latent)
            physical=default+action.clamp(-1,1)*scale
            errors.append((physical-target[start:start+128].to(a.device)).square().cpu())
            commands.append(physical.cpu());latents.append(latent.cpu())
            clipped.append((action.abs()>1).cpu())
    error=torch.cat(errors);saturated=torch.cat(clipped)
    result={}
    for name,mask in [('all',torch.ones(len(x),dtype=torch.bool)),('teacher_executed',teacher.bool()),('student_executed',~teacher.bool())]:
        if not mask.any():
            result[name]=dict(states=0)
            continue
        per_joint=error[mask].mean(0)
        result[name]=dict(states=int(mask.sum()),physical_action_mse=float(per_joint.mean()),leg_mse=float(per_joint[:12].mean()),
            weighted_action_mse=float((per_joint*weights).sum()/weights.sum()),
            per_joint_mse=dict(zip(meta['joint_names'],per_joint.tolist())),
            saturated_joint_fraction=float(saturated[mask].float().mean()))
    return result,torch.cat(commands),torch.cat(latents)

results={'saved_semantic_labels':measure(None)[0]}
reference_targets=reference_latents=None
for path in a.checkpoints:
    assert (path.parent/'complete.json').exists()
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    model=load_planner(torch.load(path,map_location='cpu',weights_only=False),a.device)
    assert model.schema==SCHEMA_V4
    raw_normalized=(x-model.obs_mean.detach().cpu())/model.obs_std.detach().cpu().clamp_min(.05)
    clipped=raw_normalized.abs()>20
    normalization={}
    for name,begin,end in [('actor_state',0,930),('ball_root_state',930,944),('backward_prior',944,1200),('task_history',1200,2400),('command',2400,2406)]:
        region=clipped[:,begin:end]
        normalization[name]=dict(clipped_value_fraction=float(region.float().mean()),
            states_with_any_clipping=int(region.any(-1).sum()),max_abs_normalized=float(raw_normalized[:,begin:end].abs().max()))
    counts=clipped.sum(0)
    indices=counts.argsort(descending=True)[:10]
    normalization['most_clipped_features']=[dict(index=int(i),states=int(counts[i])) for i in indices if counts[i]>0]
    groups,commanded,latents=measure(model)
    if reference_targets is None:reference_targets,reference_latents=commanded,latents
    corrections={}
    for name,mask in [('all',torch.ones(len(x),dtype=torch.bool)),('teacher_executed',teacher.bool()),('student_executed',~teacher.bool())]:
        if not mask.any():continue
        delta=commanded[mask]-reference_targets[mask]
        error=reference_targets[mask]-target[mask]
        corrections[name]=dict(states=int(mask.sum()),joint_target_rms_rad=float(delta.square().mean().sqrt()),
            leg_target_rms_rad=float(delta[:,:12].square().mean().sqrt()),
            first_latent_rms=float((latents[mask]-reference_latents[mask]).square().mean().sqrt()),
            target_delta_rms_over_reference_expert_error_rms=float((delta.square().mean()/error.square().mean().clamp_min(1e-12)).sqrt()))
    results[str(path)]=dict(checkpoint_sha256=checksum,groups=groups,observation_normalization=normalization,correction_from_first_checkpoint=corrections)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==checksum
report=dict(split=a.split,correction_reference=str(a.checkpoints[0]),states=len(x),state_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),data=list(map(str,a.data)),results=results,
    scope=('First three quarters' if a.split=='train' else 'Last quarter')+' of recorded passing environments, 10 frames before observed kicks excluding the kicked state; saved semantic labels are a recorded reference, not an optimal-action bound. Corrections compare deterministic first latents and commanded BFM joint targets to the first supplied checkpoint, before scene-specific clipping or delay. Expert-action error is diagnostic, not the task objective. No environment rollout or parameter updates.')
a.output.write_text(json.dumps(report,indent=2))
for key,value in results.items():
    groups=value.get('groups',value)
    print(key,{k:{m:g[m] for m in ['states','weighted_action_mse','leg_mse'] if m in g} for k,g in groups.items()},flush=True)
