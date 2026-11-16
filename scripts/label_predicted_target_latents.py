"""Create training-only latent labels for internally predicted physical targets.

All source inputs are the first three quarters of recorded environment lanes.
The original expert targets and validation states are not optimization targets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from fiber_soccer.guidance_data import load_guidance_data
from fiber_soccer.planner_policy import SoccerPolicy
from fiber_soccer.latent_oracle import fit_latent,SOLVER_VERSION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();assert a.batch_size>0
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);started=time.monotonic()
    def write(name,data):(a.output/name).write_text(json.dumps(data,indent=2)+'\n')
    def state(stage,**details):write('status.json',dict(stage=stage,elapsed=time.monotonic()-started,**details))
    try:
        state('loading_training_split')
        source_hash=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
        header=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        assert header['planner_arch']=='target_guided_latent_v1' and not header['config']['smoke_test']
        paths=[Path(record['path']) for record in header['config']['data']]
        training,validation,provenance=load_guidance_data(paths)
        del validation
        assert [(r['path'],r['sha256']) for r in provenance]==[(r['path'],r['sha256']) for r in header['config']['data']]
        policy=SoccerPolicy(header,a.device)
        weights=torch.ones(29,device=a.device);weights[:12]=2;root_weight=(weights/weights.mean()).sqrt()
        def physical(raw):return policy.default+(6*raw).clamp(-6,6)*policy.scale
        def error(raw,target):return ((physical(raw)-target).square()*weights).sum(-1)/weights.sum()
        metadata=dict(purpose='train_inverse_from_predicted_targets_v1',checkpoint=str(a.checkpoint),
            source_checkpoint_sha256=source_hash,predictor_sha256=header['config']['predictor_sha256'],
            base_sha256=header['config']['base_sha256'],data=provenance,iterations=64,learning_rate=.05,
            batch_size=a.batch_size,solver_version=SOLVER_VERSION,gradient_reduction='sum_per_state',
            leg_weight=2,split='first_three_quarters_of_each_source_environment_set',
            expert_targets_used=False,validation_states_used=False,
            source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write('config.json',metadata)
        (a.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        reports={}
        for skill,(x,unused_expert_targets,contact) in training.items():
            state('labeling',skill=skill,states=len(x),completed=0)
            state_hash=hashlib.sha256(x.numpy().tobytes()).hexdigest()
            solved_all=[];targets_all=[];before_all=[];after_all=[]
            for begin in range(0,len(x),a.batch_size):
                xb=x[begin:begin+a.batch_size].to(a.device)
                with torch.no_grad():
                    initial=policy.planner.act(xb)
                    desired=policy.planner.target_predictor.bounded_targets(xb)
                    before=error(policy.actor(xb[:,:64],xb[:,64:93],xb[:,93:930],initial),desired)
                solved,info=fit_latent(policy.actor,(xb[:,:64],xb[:,64:93],xb[:,93:930]),desired*root_weight,
                    initial,lambda raw:physical(raw)*root_weight,iterations=64)
                with torch.no_grad():after=error(policy.actor(xb[:,:64],xb[:,64:93],xb[:,93:930],solved),desired)
                assert torch.isfinite(solved).all() and torch.isfinite(after).all()
                assert torch.allclose(solved.norm(dim=-1),torch.full((len(xb),),16.,device=a.device),atol=3e-6)
                solved_all.append(solved.cpu());targets_all.append(desired.cpu());before_all.append(before.cpu());after_all.append(after.cpu())
                if begin//a.batch_size%16==0:
                    done=begin+len(xb);state('labeling',skill=skill,states=len(x),completed=done)
                    print(json.dumps(dict(skill=skill,completed=done,total=len(x),elapsed=time.monotonic()-started)),flush=True)
            before=torch.cat(before_all);after=torch.cat(after_all)
            destination=a.output/f'{skill}.pt';temporary=destination.with_suffix('.tmp')
            torch.save(dict(metadata=metadata,skill=skill,x_sha256=state_hash,z=torch.cat(solved_all),
                desired_targets=torch.cat(targets_all),contact=contact,initial_mapping_mse=before,final_mapping_mse=after),temporary)
            temporary.replace(destination)
            reports[skill]=dict(states=len(x),state_sha256=state_hash,label_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                initial_mapping_mse=float(before.mean()),final_mapping_mse=float(after.mean()),maximum_mapping_mse=float(after.max()))
            write('partial.json',reports)
        assert all(p.grad is None for p in policy.parameters())
        assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==source_hash
        write('complete.json',dict(metadata=metadata,skills=reports,elapsed=time.monotonic()-started,
            scope='Training-only solver labels for predicted targets. No simulator performance measurement.'))
        state('completed')
    except Exception as error:
        state('failed',error=repr(error));raise


if __name__=='__main__':main()
