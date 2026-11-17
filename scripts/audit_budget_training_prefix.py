"""Verify the 100-iteration prefix while the fixed long training continues."""
import argparse
import json
from pathlib import Path
import torch
from audit_new_ball_history_sensitivity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(4)
    def rows(path):
        result=[]
        for line in path.read_text().splitlines():
            try:result.append(json.loads(line))
            except json.JSONDecodeError:break
        return result
    current=rows(a.run/'metrics.jsonl');old=rows(a.reference/'metrics.jsonl')
    assert len(old)==100 and len(current)>=101,'Wait until iteration 100 export has completed'
    assert all({k:v for k,v in left.items() if k!='elapsed'}=={k:v for k,v in right.items() if k!='elapsed'}
               for left,right in zip(old,current[:100]))
    snapshot=torch.load(a.run/'latest.pt',map_location='cpu',weights_only=False)
    assert snapshot['updates']==100,'The live latest checkpoint has passed the prefix window; do not label it iteration 100'
    reference=torch.load(a.reference/'final.pt',map_location='cpu',weights_only=False)
    assert reference['updates']==100
    counts={}
    for name,left,right in [('planner',snapshot['planner'],reference['planner']),
            ('adapter',snapshot['post_pass_latent_adapter']['model'],reference['post_pass_latent_adapter']['model'])]:
        assert left.keys()==right.keys()
        for key in left:assert torch.equal(left[key],right[key]),(name,key)
        counts[name]=len(left)
    saved=a.run/'prefix_100_audit.pt';assert not saved.exists();torch.save(snapshot,saved)
    result=dict(compared_iterations=100,metrics_exact_except_elapsed=True,planner_and_adapter_tensors_exact=True,
        tensor_counts=counts,prefix_snapshot_sha256=sha(saved),reference_checkpoint_sha256=sha(a.reference/'final.pt'),
        prefix_environment_steps=current[99]['environment_steps'],prefix_actor_updates=sum(r['actor_updates'] for r in current[:100]),
        prefix_critic_updates=sum(r['critic_updates'] for r in current[:100]),
        scope='Independent prefix reproducibility check. Snapshot is an audit artifact, not a selected performance checkpoint. The 1000-iteration final checkpoint remains the predeclared evaluation candidate; training was not stopped or resumed from this snapshot.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
