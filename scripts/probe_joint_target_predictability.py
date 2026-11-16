"""Offline shared-regressor diagnostic for expert physical-target predictability.

Uses existing training trajectories, with the last quarter of lanes held out.
This predicts physical targets solely to diagnose the latent-planning bottleneck;
it is not a deployable SoccerPolicy and executes no motor commands.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
from fiber_soccer.planner import with_task_history, trajectory_event_ids, observed_contact_window, SCHEMA_V3, SCHEMA_V4, load_planner
from fiber_soccer.native_bfm import NativeActor, load_native
from fiber_soccer.paths import ASSETS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TargetRegressor(nn.Module):
    def __init__(self, mean, std, default, scale):
        super().__init__()
        self.register_buffer('mean', mean.clone())
        self.register_buffer('std', std.clamp_min(.05).clone())
        self.register_buffer('default', default.clone())
        self.register_buffer('scale', scale.clone())
        self.net = nn.Sequential(nn.Linear(2406,512),nn.ELU(),nn.Linear(512,512),nn.ELU(),
                                 nn.Linear(512,512),nn.ELU(),nn.Linear(512,29))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.default + self.net(((x-self.mean)/self.std).clamp(-20,20))*self.scale


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,nargs='+',required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=20000)
    p.add_argument('--seed',type=int,default=816001)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--smoke-test',action='store_true')
    a=p.parse_args()
    assert a.updates>0 and (not a.smoke_test or a.updates<=4)
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    started=time.monotonic()
    def write(name,data):
        (a.output/name).write_text(json.dumps(data,indent=2)+'\n')
    def state(stage,**kwargs):
        write('status.json',dict(stage=stage,elapsed=time.monotonic()-started,**kwargs))
    try:
        state('loading_data')
        base_hash=sha(a.base)
        header=torch.load(a.base,map_location='cpu',weights_only=False)
        assert header['observation_schema']==SCHEMA_V4
        mean=header['planner']['obs_mean'];std=header['planner']['obs_std']
        meta=json.loads((ASSETS/'actor_meta.json').read_text())
        default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'])
        scale=torch.tensor(meta['action_rescale'])*1.5
        parts={skill:[] for skill in ['pass','shoot']};validation={skill:[] for skill in parts};provenance=[]
        for path in a.data:
            assert (path.parent/'complete.json').exists()
            data=torch.load(path,map_location='cpu',weights_only=True);m=data['metadata'];rows=data['rows'];skill=m['skill']
            assert skill in parts and m['seed'] not in {811314,815701,815702,815801,815802,815803,815804}
            x=torch.stack([r['x'] for r in rows]);target=torch.stack([r['target'] for r in rows])
            if m['schema']==SCHEMA_V3:x=with_task_history(x,torch.stack([r['expert_observation'] for r in rows]))
            else:assert m['schema']==SCHEMA_V4
            assert x.shape[-1]==2406 and target.shape==(*x.shape[:2],29)
            assert torch.isfinite(x).all() and torch.isfinite(target).all()
            assert (x[:,:,-2:].argmax(-1)==int(skill=='shoot')).all()
            episodes=torch.stack([r['episode'] for r in rows])
            present=['ball_episode' in r for r in rows];assert all(present) or not any(present)
            ball=torch.stack([r['ball_episode'] for r in rows]) if all(present) else None
            contact=observed_contact_window(x,trajectory_event_ids(x,episodes,ball),before=10,after=0)&(x[:,:,940]<.5)
            n=x.shape[1];cut=n*3//4;assert 0<cut<n
            train_x=x[:,:cut].flatten(0,1);train_y=target[:,:cut].flatten(0,1);train_contact=contact[:,:cut].flatten()
            if a.smoke_test:
                ids=torch.cat((torch.arange(min(256,len(train_x))),train_contact.nonzero().flatten()[:128])).unique()
                train_x,train_y,train_contact=train_x[ids],train_y[ids],train_contact[ids]
            parts[skill].append((train_x,train_y,train_contact))
            # Full precontact held-out set; no validation rows enter optimization.
            mask=contact[:,cut:]
            teacher=torch.stack([r['teacher_executed'] for r in rows])[:,cut:]
            validation[skill].append((x[:,cut:][mask],target[:,cut:][mask],teacher[mask]))
            provenance.append(dict(path=str(path),sha256=sha(path),metadata=m,training_lanes=[0,cut],validation_lanes=[cut,n]))
            del data,rows,x,target,episodes,ball,contact,teacher,train_x,train_y,train_contact
        training={skill:tuple(torch.cat([entry[i] for entry in values]) for i in range(3)) for skill,values in parts.items()}
        validation={skill:tuple(torch.cat([entry[i] for entry in values]) for i in range(3)) for skill,values in validation.items()}
        del parts
        assert all(len(values[0])>0 and values[2].any() for values in training.values())
        assert all(len(values[0])>0 for values in validation.values())
        model=TargetRegressor(mean,std,default,scale).to(a.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        weights=torch.ones(29,device=a.device);weights[:12]=2
        generator=torch.Generator().manual_seed(a.seed)
        contact_ids={skill:values[2].nonzero().flatten() for skill,values in training.items()}
        config=dict(method='offline_shared_physical_target_regressor_diagnostic_v1',deployable=False,
            base=str(a.base),base_sha256=base_hash,seed=a.seed,updates=a.updates,smoke_test=a.smoke_test,
            learning_rate=2e-4,batch_per_skill=64,precontact_fraction=.25,leg_weight=2,
            architecture=[2406,512,512,512,29],parameters=sum(v.numel() for v in model.parameters()),
            input='Same v4 observation and frozen r3 normalization; target action is not an input',
            validation='All precontact states in last quarter of source lanes; previously used historical diagnostic split, not fresh closed-loop evaluation',
            data=provenance,training_states={k:len(v[0]) for k,v in training.items()},
            validation_states={k:len(v[0]) for k,v in validation.items()},
            validation_state_sha256={k:hashlib.sha256(v[0].numpy().tobytes()).hexdigest() for k,v in validation.items()})
        write('config.json',config)
        (a.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        def evaluate(predict):
            result={}
            with torch.no_grad():
                for skill,(x,y,teacher) in validation.items():
                    predictions=torch.cat([predict(batch.to(a.device)).cpu() for batch in x.split(256)])
                    error=(predictions-y).square();clipped=default+((predictions-default)/scale).clamp(-1,1)*scale
                    groups={}
                    for name,mask in [('all',torch.ones(len(x),dtype=torch.bool)),('teacher_executed',teacher.bool()),('student_executed',~teacher.bool())]:
                        if not mask.any():continue
                        weighted=(error[mask]*weights.cpu()).sum(-1)/weights.sum().cpu()
                        clip_error=((clipped[mask]-y[mask]).square()*weights.cpu()).sum(-1)/weights.sum().cpu()
                        groups[name]=dict(states=int(mask.sum()),weighted_action_mse=float(weighted.mean()),
                            clipped_weighted_action_mse=float(clip_error.mean()),leg_mse=float(error[mask,:12].mean()),
                            targets_outside_bfm_command_range=float((((y[mask]-default)/scale).abs()>1).float().mean()))
                    result[skill]=groups
            return result
        state('measuring_baseline')
        base=load_planner(header,a.device).requires_grad_(False)
        actor=NativeActor(load_native(a.device)).eval().requires_grad_(False)
        def baseline(x):
            return default.to(x)+(actor(x[:,:64],x[:,64:93],x[:,93:930],base.act(x))).clamp(-1,1)*scale.to(x)
        baseline_metrics=evaluate(baseline)
        del base,actor,header
        write('baseline.json',baseline_metrics)
        def metrics(step,loss=None):
            record=dict(update=step,elapsed=time.monotonic()-started,validation=evaluate(model),loss=loss)
            with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True);state('training',update=step)
            return record
        metrics(0)
        for step in range(1,a.updates+1):
            xs=[];ys=[]
            for skill,(x,y,_) in training.items():
                selected=torch.cat((torch.randint(len(x),(48,),generator=generator),
                    contact_ids[skill][torch.randint(len(contact_ids[skill]),(16,),generator=generator)]))
                xs.append(x[selected]);ys.append(y[selected])
            x=torch.cat(xs).to(a.device);y=torch.cat(ys).to(a.device)
            loss=((model(x)-y).square()*weights).sum(-1).div(weights.sum()).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            if step%1000==0 or step==a.updates:latest=metrics(step,float(loss.detach()))
        torch.save(dict(model=model.cpu().state_dict(),config=config,deployable=False),a.output/'final.pt')
        restored=TargetRegressor(mean,std,default,scale)
        restored.load_state_dict(torch.load(a.output/'final.pt',map_location='cpu',weights_only=True)['model'])
        probe=torch.cat([values[0][:8] for values in validation.values()])
        with torch.no_grad():assert torch.equal(model(probe),restored(probe))
        assert sha(a.base)==base_hash
        complete=dict(updates=a.updates,baseline=baseline_metrics,final=latest,config=config,
            checkpoint_sha256=sha(a.output/'final.pt'),deployable=False,checkpoint_roundtrip_exact=True,
            scope='Offline target predictability only. This diagnostic neither executes direct actions nor establishes a latent policy or closed-loop improvement.')
        write('complete.json',complete);state('completed')
    except Exception as error:
        state('failed',error=repr(error));raise


if __name__=='__main__':main()
