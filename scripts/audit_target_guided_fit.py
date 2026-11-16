"""Inspect learned mapper fit on fixed training samples."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.guidance_data import load_guidance_data
from fiber_soccer.planner_policy import SoccerPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(4)
    header=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    training,_,_=load_guidance_data([Path(d['path']) for d in header['config']['data']])
    policy=SoccerPolicy(header,'cuda:0');weights=torch.ones(29,device='cuda:0');weights[:12]=2
    result={};g=torch.Generator().manual_seed(816022)
    with torch.no_grad():
        for skill,(x,_,contact) in training.items():
            labels=torch.load(a.labels/f'{skill}.pt',map_location='cpu',weights_only=True)
            assert labels['x_sha256']==hashlib.sha256(x.numpy().tobytes()).hexdigest()
            contacts=contact.nonzero().flatten()
            samples={'train_uniform':torch.randperm(len(x),generator=g)[:1024],
                     'train_contact':contacts[torch.randperm(len(contacts),generator=g)[:1024]]}
            result[skill]={}
            for group,ids in samples.items():
                errors=[];latent_errors=[];corrections=[];activations=[[],[]]
                for rows in ids.split(256):
                    xb=x[rows].cuda();desired=policy.planner.target_predictor.bounded_targets(xb)
                    prior=policy.planner.prior.act(xb);z=policy.planner.decode(xb,prior,desired)
                    raw=policy.actor(xb[:,:64],xb[:,64:93],xb[:,93:930],z)
                    actual=policy.default+(6*raw).clamp(-6,6)*policy.scale
                    errors.append(((actual-desired).square()*weights).sum(-1).cpu()/weights.sum().cpu())
                    latent_errors.append((z.cpu()-labels['z'][rows]).square().mean(-1))
                    h=torch.cat((prior,policy.planner.prior.normalize_observation(xb),
                        ((desired-policy.planner.target_predictor.default)/policy.planner.target_predictor.scale).clamp(-1,1)),-1)
                    for index,layer in enumerate(policy.planner.decoder):
                        h=layer(h)
                        if index in [1,3]:activations[index//2].append(h.cpu())
                    corrections.append(h.cpu())
                c=torch.cat(corrections)
                result[skill][group]=dict(states=len(ids),mapping_mse=float(torch.cat(errors).mean()),
                    latent_label_mse=float(torch.cat(latent_errors).mean()),
                    correction_rms=float(c.square().mean().sqrt()),correction_state_std=float(c.std(0).mean()),
                    hidden=[dict(near_zero_fraction=float((torch.cat(v).abs()<1e-4).float().mean()),
                        mean=float(torch.cat(v).mean()),std=float(torch.cat(v).std())) for v in activations])
    a.output.write_text(json.dumps(dict(checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        groups=result,scope='Fixed training samples only; not closed-loop performance or fresh validation.'),indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
