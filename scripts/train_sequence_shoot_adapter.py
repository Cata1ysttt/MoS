"""Adapt shooting latents on post-pass states while anchoring original shooting outputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from fiber_soccer.planner import load_planner,with_task_history,SCHEMA_V4
from fiber_soccer.native_bfm import NativeActor,load_native
from fiber_soccer.paths import ASSETS,ROOT
from fiber_soccer.sequence_adaptation import split_sequence_labels,EnvironmentSampler

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--labels',type=Path,required=True)
p.add_argument('--retention-data',type=Path,nargs='+',required=True)
p.add_argument('--pass-probe-data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--updates',type=int,default=10000)
p.add_argument('--batch-size',type=int,default=128)
p.add_argument('--retention-weight',type=float,default=1.)
p.add_argument('--learning-rate',type=float,default=2e-4)
p.add_argument('--seed',type=int,default=815211)
p.add_argument('--device',default='cuda:0')
p.add_argument('--smoke-test',action='store_true')
a=p.parse_args()
assert a.updates>0 and a.batch_size>=2 and a.batch_size%2==0
assert a.retention_weight>0 and a.learning_rate>0
assert not a.smoke_test or (a.updates<=4 and a.batch_size<=32)
a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(a.seed)
started=time.monotonic()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def state(stage,**details):
    (a.output/'status.json').write_text(json.dumps({'stage':stage,'elapsed':time.monotonic()-started,**details},indent=2))
try:
    state('validating_data')
    label_report=json.loads(a.labels.with_suffix('.json').read_text())
    assert label_report['label_sha256']==sha(a.labels)
    data=torch.load(a.labels,map_location='cpu',weights_only=True)
    base_sha=sha(a.base);expert_sha=sha(ROOT/'body_yaw_49999.onnx')
    train_ids,valid_ids=split_sequence_labels(data,base_sha,expert_sha)
    assert label_report['source_states_sha256']==data['metadata']['source_states_sha256']
    assert label_report['expert_sha256']==expert_sha
    assert sha(a.labels.parent/'shoot_states.pt')==data['metadata']['source_states_sha256']
    new_x=data['x'];new_target=data['target'];new_env=data['environment']
    header=torch.load(a.base,map_location='cpu',weights_only=False)
    base=load_planner(header,a.device).requires_grad_(False)
    assert base.schema==SCHEMA_V4 and base.adapter is None
    actor=NativeActor(load_native(a.device)).eval().requires_grad_(False)
    meta=json.loads((ASSETS/'actor_meta.json').read_text())
    default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=a.device)
    scale=torch.tensor(meta['action_rescale'],device=a.device)*.25
    weights=torch.ones(29,device=a.device);weights[:12]=2
    def targets(model,x):
        z=model.act(x)
        action=actor(x[:,:64],x[:,64:93],x[:,93:930],z)
        return default+(action*6).clamp(-6,6)*scale
    old_x=[];old_env=[];old_train=[];retention_metadata=[]
    for group,path in enumerate(a.retention_data):
        d=torch.load(path,map_location='cpu',weights_only=True)
        assert d['metadata']['skill']=='shoot' and d['metadata']['seed'] not in (811314,815101,815102)
        rows=d['rows'];n=len(rows[0]['x'])
        x=with_task_history(torch.stack([r['x'] for r in rows]),torch.stack([r['expert_observation'] for r in rows]))
        env=torch.arange(n).expand(len(rows),-1)
        old_x.append(x.flatten(0,1));old_env.append((env+group*100000).flatten())
        old_train.append((env<n*3//4).flatten())
        retention_metadata.append(dict(path=str(path),sha256=sha(path),seed=d['metadata']['seed'],states=x.shape[0]*n))
    old_x=torch.cat(old_x);old_env=torch.cat(old_env);old_train=torch.cat(old_train)
    old_train_ids=old_train.nonzero().flatten();old_valid_ids=(~old_train).nonzero().flatten()
    if a.smoke_test:
        keep=torch.cat([ids[torch.linspace(0,len(ids)-1,min(64,len(ids))).long()] for ids in [old_train_ids,old_valid_ids]])
        old_x,old_env,old_train=old_x[keep],old_env[keep],old_train[keep]
        old_train_ids=old_train.nonzero().flatten();old_valid_ids=(~old_train).nonzero().flatten()
        train_ids=train_ids[torch.linspace(0,len(train_ids)-1,min(128,len(train_ids))).long()]
        valid_ids=valid_ids[torch.linspace(0,len(valid_ids)-1,min(128,len(valid_ids))).long()]
    state('caching_original_shooting_targets',states=len(old_x))
    with torch.inference_mode():
        old_target=torch.cat([targets(base,old_x[i:i+512].to(a.device)).cpu() for i in range(0,len(old_x),512)])
    pass_data=torch.load(a.pass_probe_data,map_location='cpu',weights_only=True)
    assert pass_data['metadata']['skill']=='pass'
    rows=pass_data['rows'];n=len(rows[0]['x']);cut=n*3//4
    pass_x=torch.cat([with_task_history(rows[t]['x'][cut:cut+8],rows[t]['expert_observation'][cut:cut+8])
        for t in torch.linspace(0,len(rows)-1,8).long().tolist()]).to(a.device)
    assert (pass_x[:,-2:].argmax(-1)==0).all()
    with torch.inference_mode():pass_z=base.act(pass_x);pass_targets=targets(base,pass_x)
    model=copy.deepcopy(base).enable_skill_adapter(1,width=128,observation_input=True,space='latent')
    optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=a.learning_rate,weight_decay=1e-4)
    sampler=EnvironmentSampler(new_env,train_ids);old_sampler=EnvironmentSampler(old_env,old_train_ids)
    generator=torch.Generator().manual_seed(a.seed)
    config=dict(method='continuous_shooting_latent_adapter_v1',base=str(a.base),base_sha256=base_sha,
        labels=str(a.labels),labels_sha256=label_report['label_sha256'],collection_metadata=data['metadata'],
        retention_data=retention_metadata,retention_target='frozen_r3_physical_targets',retention_weight=a.retention_weight,
        seed=a.seed,updates=a.updates,batch_size=a.batch_size,learning_rate=a.learning_rate,smoke_test=a.smoke_test,
        train_states=len(train_ids),valid_states=len(valid_ids),train_environments=int(new_env[train_ids].unique().numel()),
        valid_environments=int(new_env[valid_ids].unique().numel()),retention_train_states=len(old_train_ids),
        retention_valid_states=len(old_valid_ids),sampling='Uniform environment then uniform state, equal new/retention batches',
        validation='All held-out states, mean per environment, no rollout success implied',checkpoint_selection='final_iteration',
        adapter_skill=1,adapter_space='latent',adapter_width=128,adapter_observation_input=True,
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        frozen_parameters=sum(v.numel() for v in model.parameters() if not v.requires_grad))
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    snapshots=a.output/'source';snapshots.mkdir()
    for file in [Path(__file__),ROOT/'src/fiber_soccer/sequence_adaptation.py',ROOT/'src/fiber_soccer/flow_planner.py',ROOT/'src/fiber_soccer/planner_policy.py']:
        shutil.copyfile(file,snapshots/file.name)
    def errors(predicted,target):return ((predicted-target).square()*weights).sum(-1)/weights.sum()
    def evaluate(x,target,env,ids):
        values=[]
        with torch.inference_mode():
            for batch in ids.split(256):
                values.append(errors(targets(model,x[batch].to(a.device)),target[batch].to(a.device)).cpu())
        value=torch.cat(values);_,inverse=env[ids].unique(return_inverse=True)
        totals=torch.zeros(int(inverse.max())+1);counts=torch.zeros_like(totals)
        totals.scatter_add_(0,inverse,value);counts.scatter_add_(0,inverse,torch.ones_like(value))
        return float((totals/counts).mean())
    def metrics(update,loss=None):
        model.eval()
        row=dict(update=update,elapsed=time.monotonic()-started,
            new_validation_action_mse=evaluate(new_x,new_target,new_env,valid_ids),
            retention_validation_action_mse=evaluate(old_x,old_target,old_env,old_valid_ids))
        if loss is not None:row.update(loss)
        with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(row,flush=True);state('training',**row)
        return row
    initial=metrics(0)
    for update in range(1,a.updates+1):
        model.train();ids=sampler.sample(a.batch_size//2,generator);old_ids=old_sampler.sample(a.batch_size//2,generator)
        x=torch.cat((new_x[ids],old_x[old_ids])).to(a.device)
        target=torch.cat((new_target[ids],old_target[old_ids])).to(a.device)
        error=errors(targets(model,x),target)
        new_loss=error[:len(ids)].mean();retention_loss=error[len(ids):].mean()
        loss=100*(new_loss+a.retention_weight*retention_loss)
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),1.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        optimizer.step()
        if update%200==0 or update==a.updates:
            final=metrics(update,dict(new_training_action_mse=float(new_loss.detach()),
                retention_training_action_mse=float(retention_loss.detach()),gradient_norm=float(norm)))
    state('verifying_export')
    model.eval()
    assert any(torch.count_nonzero(v).item() for v in model.adapter[-1].parameters())
    assert all(v.grad is None for v in actor.parameters())
    for key,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[key]),key
    with torch.inference_mode():
        assert torch.equal(model.act(pass_x),pass_z)
        assert torch.equal(targets(model,pass_x),pass_targets)
    payload=dict(header,planner=model.state_dict(),planner_arch='flow_latent_adapter_v1',adapter_space='latent',
        adapter_width=128,adapter_skill=1,adapter_observation_input=True,adapter_ball_distance_gate=None,
        adapter_post_kick_only=False,updates=a.updates,config=config)
    torch.save(payload,a.output/'final.pt')
    from fiber_soccer.planner_policy import SoccerPolicy
    deployed=SoccerPolicy(a.output/'final.pt',a.device)
    check=torch.cat((pass_x,new_x[valid_ids[:64]].to(a.device)))
    with torch.inference_mode():
        actual=deployed(check)
        assert torch.equal(actual['latent'],model.act(check))
        assert torch.equal(actual['joint_targets'],targets(model,check))
    complete=dict(updates=a.updates,smoke_test=a.smoke_test,checkpoint_sha256=sha(a.output/'final.pt'),
        shared_tensors_unchanged=True,pass_probe_states=len(pass_x),pass_probe_outputs_unchanged=True,
        actor_frozen=True,deployable_roundtrip=True,initial=initial,final=final)
    (a.output/'complete.json').write_text(json.dumps(complete,indent=2));state('completed',**complete)
except Exception as error:
    state('failed',error=repr(error));raise
