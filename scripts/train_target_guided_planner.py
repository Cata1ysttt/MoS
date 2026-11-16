"""Fit a shared latent mapping to internally predicted physical joint targets.

The predictor, r3 flow prior and AH-BFM stay frozen. No expert action is supplied
as a decoder input or used as its optimization target. Expert labels are scored
only to check whether target-prediction accuracy survives latent decoding.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from fiber_soccer.planner import load_planner
from fiber_soccer.target_guided import TargetRegressor,TargetGuidedLatentPlanner,target_guided_payload
from fiber_soccer.guidance_data import load_guidance_data
from fiber_soccer.latent_supervision import load_labels,confidence,distillation_loss
from fiber_soccer.native_bfm import NativeActor,load_native
from fiber_soccer.planner_policy import SoccerPolicy
from fiber_soccer.paths import ASSETS,ROOT


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--predictor',type=Path,required=True)
    p.add_argument('--data',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=20000)
    p.add_argument('--decoder-width',type=int,default=512)
    p.add_argument('--seed',type=int,default=816011)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--warm-start',type=Path)
    p.add_argument('--latent-labels',type=Path)
    a=p.parse_args();assert a.updates>0 and (not a.smoke_test or a.updates<=4)
    assert bool(a.warm_start)==bool(a.latent_labels)
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(a.seed);started=time.monotonic()
    def write(name,data):(a.output/name).write_text(json.dumps(data,indent=2)+'\n')
    def state(stage,**details):write('status.json',dict(stage=stage,elapsed=time.monotonic()-started,**details))
    try:
        state('loading')
        parent_hash=sha(a.base);predictor_hash=sha(a.predictor)
        header=torch.load(a.base,map_location='cpu',weights_only=False)
        h=torch.load(a.predictor,map_location='cpu',weights_only=True)
        assert not h['config']['smoke_test'] and h['config']['base_sha256']==parent_hash
        assert json.loads((a.predictor.parent/'complete.json').read_text())['checkpoint_sha256']==predictor_hash
        prior=load_planner(header,a.device)
        predictor=TargetRegressor(*[h['model'][k] for k in ['mean','std','default','scale']]).to(a.device)
        predictor.load_state_dict(h['model'])
        assert torch.equal(predictor.mean,prior.obs_mean) and torch.equal(predictor.std,prior.obs_std.clamp_min(.05))
        model=TargetGuidedLatentPlanner(prior,predictor,a.decoder_width)
        actor=NativeActor(load_native(a.device)).eval().requires_grad_(False)
        meta=json.loads((ASSETS/'actor_meta.json').read_text())
        default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=a.device)
        scale=torch.tensor(meta['action_rescale'],device=a.device)*.25
        weights=torch.ones(29,device=a.device);weights[:12]=2
        def physical(x,z):return default+(6*actor(x[:,:64],x[:,64:93],x[:,93:930],z)).clamp(-6,6)*scale
        training,validation,provenance=load_guidance_data(a.data,a.smoke_test and not a.latent_labels)
        assert [(d['path'],d['sha256']) for d in provenance]==[(d['path'],d['sha256']) for d in h['config']['data']]
        probes=torch.cat([values[0][:16] for values in validation.values()]).to(a.device)
        with torch.no_grad():
            assert torch.equal(model.act(probes),prior.act(probes))
            assert torch.equal(physical(probes,model.act(probes)),physical(probes,prior.act(probes)))
        label_audit={};labels={};warm_hash=None
        if a.warm_start:
            warm_hash=sha(a.warm_start)
            warm=torch.load(a.warm_start,map_location='cpu',weights_only=False)
            assert warm['planner_arch']=='target_guided_latent_v1' and not warm['config']['smoke_test']
            assert warm['config']['base_sha256']==parent_hash and warm['config']['predictor_sha256']==predictor_hash
            model.load_state_dict(warm['planner'])
            with torch.no_grad():
                assert torch.equal(model.act(probes),load_planner(warm,a.device).act(probes))
            labels,label_audit=load_labels(a.latent_labels,training,provenance,warm_hash,parent_hash,predictor_hash)
            write('label_audit.json',label_audit)
            if a.smoke_test:
                # Validate the full row alignment first, then subset all tensors together.
                for skill,(x,y,contact) in list(training.items()):
                    ids=torch.cat((torch.arange(min(256,len(x))),contact.nonzero().flatten()[:128])).unique()
                    training[skill]=(x[ids],y[ids],contact[ids])
                    for key in ['z','desired_targets','contact','initial_mapping_mse','final_mapping_mse']:
                        labels[skill][key]=labels[skill][key][ids]
        state('caching_frozen_prior_and_predicted_targets')
        cached={}
        with torch.no_grad():
            for skill,(x,_,contact) in training.items():
                zs=[];targets=[]
                for batch in x.split(1024):
                    batch=batch.to(a.device);zs.append(prior.act(batch).cpu());targets.append(predictor.bounded_targets(batch).cpu())
                desired=torch.cat(targets)
                if labels:
                    torch.testing.assert_close(labels[skill]['desired_targets'],desired,atol=2e-6,rtol=1e-5)
                cached[skill]=(x,torch.cat(zs),desired,contact.nonzero().flatten())
        config=dict(method='target_guided_latent_mapping_v1',base=str(a.base),base_sha256=parent_hash,
            predictor=str(a.predictor),predictor_sha256=predictor_hash,actor_sha256=sha(ROOT/'AH-BFM/AH-BFM.pt'),
            updates=a.updates,seed=a.seed,smoke_test=a.smoke_test,decoder_width=a.decoder_width,
            trainable_parameters=sum(p.numel() for p in model.decoder.parameters()),learning_rate=2e-4,
            batch_per_skill=64,precontact_fraction=.25,leg_weight=2,data=provenance,
            optimization_target='Frozen shared predictor physical targets clipped to the BFM command range',
            expert_targets_used_for_optimization=False,checkpoint_selection='final_iteration',
            deployment='One SoccerPolicy: predicted internal targets -> latent mapping -> frozen AH-BFM',
            validation_states={k:len(v[0]) for k,v in validation.items()},
            validation_state_sha256={k:hashlib.sha256(v[0].numpy().tobytes()).hexdigest() for k,v in validation.items()})
        if labels:
            config.update(method='target_guided_latent_distillation_v1',warm_start=str(a.warm_start),
                warm_start_sha256=warm_hash,latent_labels=str(a.latent_labels),label_audit=label_audit,
                latent_loss_weight=1.,physical_loss_weight=10.,confidence_residual_scale=.001,
                optimizer_initialization='fresh AdamW after loading warm-start weights',
                optimization_target='Training-only solved latents and frozen predictor physical targets')
        write('config.json',config)
        for path in [Path(__file__),ROOT/'src/fiber_soccer/target_guided.py',ROOT/'src/fiber_soccer/guidance_data.py',ROOT/'src/fiber_soccer/latent_supervision.py',ROOT/'src/fiber_soccer/planner.py',ROOT/'src/fiber_soccer/planner_policy.py']:
            (a.output/path.name).write_bytes(path.read_bytes())
        def weighted(error):return (error.square()*weights).sum(-1)/weights.sum()
        def evaluate():
            result={}
            with torch.no_grad():
                for skill,(x,target,teacher) in validation.items():
                    errors=[];mapping=[];prediction=[];norms=[]
                    for begin in range(0,len(x),256):
                        xb=x[begin:begin+256].to(a.device);yb=target[begin:begin+256].to(a.device)
                        desired=predictor.bounded_targets(xb);z=model.act(xb);decoded=physical(xb,z)
                        errors.append(weighted(decoded-yb).cpu());mapping.append(weighted(decoded-desired).cpu())
                        prediction.append(weighted(desired-yb).cpu());norms.append(z.norm(dim=-1).cpu())
                    error,mapping_error,prediction_error=torch.cat(errors),torch.cat(mapping),torch.cat(prediction)
                    assert torch.allclose(torch.cat(norms),torch.full((len(x),),16.),atol=3e-6)
                    groups={}
                    for name,mask in [('all',torch.ones(len(x),dtype=torch.bool)),('teacher_executed',teacher.bool()),('student_executed',~teacher.bool())]:
                        if mask.any():groups[name]=dict(states=int(mask.sum()),decoded_expert_target_mse=float(error[mask].mean()),
                            decoded_prediction_mse=float(mapping_error[mask].mean()),predictor_expert_target_mse=float(prediction_error[mask].mean()))
                    result[skill]=groups
            return result
        def metrics(step,loss=None):
            result=dict(update=step,elapsed=time.monotonic()-started,loss=loss,validation=evaluate())
            with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            print(json.dumps(result),flush=True);state('training',update=step)
            return result
        initial=metrics(0)
        optimizer=torch.optim.AdamW(model.decoder.parameters(),lr=2e-4,weight_decay=1e-4)
        generator=torch.Generator().manual_seed(a.seed)
        for step in range(1,a.updates+1):
            batches=[];supervised=[]
            for skill,(x,z,targets,contacts) in cached.items():
                ids=torch.cat((torch.randint(len(x),(48,),generator=generator),contacts[torch.randint(len(contacts),(16,),generator=generator)]))
                batches.append((x[ids],z[ids],targets[ids]))
                if labels:supervised.append((labels[skill]['z'][ids],confidence(labels[skill]['final_mapping_mse'][ids])))
            x,z,desired=[torch.cat([b[i] for b in batches]).to(a.device) for i in range(3)]
            latent=model.decode(x,z,desired);physical_error=weighted(physical(x,latent)-desired)
            if labels:
                solved,reliability=[torch.cat([b[i] for b in supervised]).to(a.device) for i in range(2)]
                loss=distillation_loss((latent-solved).square().mean(-1),physical_error,reliability,64)
            else:loss=physical_error.mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1,error_if_nonfinite=True);optimizer.step()
            if step%1000==0 or step==a.updates:final=metrics(step,float(loss.detach()))
        for key,value in header['planner'].items():assert torch.equal(value,prior.state_dict()[key].cpu()),key
        for key,value in h['model'].items():assert torch.equal(value,predictor.state_dict()[key].cpu()),key
        assert all(p.grad is None for p in actor.parameters())
        assert all(p.grad is None for p in prior.parameters()) and all(p.grad is None for p in predictor.parameters())
        payload=target_guided_payload(header,model,config);torch.save(payload,a.output/'final.pt')
        rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state(a.device) if torch.device(a.device).type=='cuda' else None
        old=SoccerPolicy(a.base,a.device);after=torch.random.get_rng_state();cuda_after=torch.cuda.get_rng_state(a.device) if cuda_rng is not None else None
        torch.random.set_rng_state(rng)
        if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,a.device)
        deployed=SoccerPolicy(a.output/'final.pt',a.device)
        assert torch.equal(after,torch.random.get_rng_state())
        if cuda_rng is not None:assert torch.equal(cuda_after,torch.cuda.get_rng_state(a.device))
        with torch.no_grad():
            output=deployed(probes)
            assert torch.equal(output['latent'],model.act(probes))
            assert torch.equal(output['joint_targets'],physical(probes,model.act(probes)))
        assert sha(a.base)==parent_hash and sha(a.predictor)==predictor_hash
        complete=dict(updates=a.updates,smoke_test=a.smoke_test,checkpoint_sha256=sha(a.output/'final.pt'),
            initial=initial,final=final,prior_and_predictor_unchanged=True,actor_frozen=True,
            deployable_roundtrip=True,loading_rng_matches_r3=True,zero_initialization_matches_r3=True,
            scope='Offline latent mapping and export validation, not closed-loop performance.')
        if a.warm_start:
            assert sha(a.warm_start)==warm_hash
            complete.update(warm_start_matches_source=True,training_only_labels_verified=True,
                zero_initialization_check_scope='Constructor before loading warm-start weights')
        write('complete.json',complete);state('completed')
    except Exception as error:
        state('failed',error=repr(error));raise


if __name__=='__main__':main()
