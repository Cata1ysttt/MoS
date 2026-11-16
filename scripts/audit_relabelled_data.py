"""Check that relabelling changes supervision without changing recorded states."""
import argparse
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--before',type=Path,required=True)
p.add_argument('--after',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(4)
result={}
for skill in ['pass','shoot']:
    before=torch.load(a.before/f'{skill}_trajectory.pt',map_location='cpu',weights_only=True)
    after=torch.load(a.after/f'{skill}_trajectory.pt',map_location='cpu',weights_only=True)
    assert len(before['rows'])==len(after['rows'])
    for field in ['seed','num_envs','dt','original_trajectory','actor_sha256']:
        assert before['metadata'][field]==after['metadata'][field],field
    maximum_prior_difference=0.
    errors=[[],[]];latent_changes=[]
    for old,new in zip(before['rows'],after['rows']):
        for key in ['expert_observation','target','episode','episode_tick','teacher_executed','z','previous_label_z']:
            assert torch.equal(old[key],new[key]),key
        assert torch.equal(old['x'][:,:944],new['x'][:,:944])
        assert torch.equal(old['x'][:,-6:],new['x'][:,-6:])
        difference=(old['x'][:,944:1200]-new['x'][:,944:1200]).abs().max()
        maximum_prior_difference=max(maximum_prior_difference,float(difference))
        assert maximum_prior_difference<1e-5,'Backward input changed beyond numerical tolerance'
        for values,row in zip(errors,[old,new]):values.append(row['label_action_mse'])
        latent_changes.append((old['label_z']-new['label_z']).square().mean(-1))
    result[skill]=dict(frames=len(before['rows']),states=sum(len(r['x']) for r in before['rows']),
        recorded_inputs_targets_and_executed_latents_identical=True,
        backward_prior_max_abs_difference=maximum_prior_difference,
        old_label_action_mse=float(torch.cat(errors[0]).mean()),new_label_action_mse=float(torch.cat(errors[1]).mean()),
        supervised_latent_change_mse=float(torch.cat(latent_changes).mean()),
        new_gradient_reduction=after['metadata']['semantic_gradient_reduction'])
a.output.write_text(json.dumps(dict(before=str(a.before),after=str(a.after),results=result),indent=2))
print(json.dumps(result,indent=2))
