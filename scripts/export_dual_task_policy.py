"""Export a versioned single-policy bundle and verify both command branches."""
import argparse,hashlib,json
from pathlib import Path
import torch
from fiber_soccer.dual_task_latent import load_dual_task_policy
from fiber_soccer.dual_task_deployment import FORMAT,load_soccer_policy
p=argparse.ArgumentParser();p.add_argument('--tag',default='planner_dual_task_reward_r3');a=p.parse_args()
root=Path(__file__).resolve().parents[1];run=root/'runs'/a.tag
source=run/'final.pt';dest=run/'unified_policy.pt';assert not dest.exists()
header=torch.load(source,map_location='cpu',weights_only=False)
payload=dict(format=FORMAT,policy=header,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             skills=['pass','shoot'],inference_expert_required=False)
torch.save(payload,dest)
torch.set_num_threads(4)
reference=load_dual_task_policy(source,'cuda:0');deployed=load_soccer_policy(dest,'cuda:0')
x=torch.load(run/'initial_state.pt',map_location='cuda:0',weights_only=True)['raw_observation'][:8]
for skill,kicked in [(0,0),(0,1),(1,0),(1,1),(0,1)]:
    probe=x.clone();probe[:,-2:]=0;probe[:,-2+skill]=1;probe[:,940]=kicked
    expected=reference(probe);actual=deployed(probe)
    assert all(torch.equal(expected[k],actual[k]) for k in expected)
from fiber_soccer.planner_policy import load_soccer_policy as legacy
try:legacy(payload,'cpu')
except (KeyError,ValueError):pass
else:raise AssertionError('Legacy loader must not silently ignore the new heads')
result=dict(bundle_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_sha256=payload['source_sha256'],
            both_skills_and_recovery_equal=True,legacy_loader_rejects_bundle=True)
(root/'reports'/f'{a.tag}_deployment_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
