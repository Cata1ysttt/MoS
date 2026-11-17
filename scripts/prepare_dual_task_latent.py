"""Create a zero residual checkpoint and verify both task branches on real states."""
import json
from pathlib import Path
import torch
from fiber_soccer.dual_task_latent import DualTaskResidual, load_dual_task_policy, DualTaskPhase
from fiber_soccer.planner_policy import load_soccer_policy

root=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
source=root/'runs/planner_new_ball_task_history_r1/diagnostic.pt'
output=root/'runs/planner_dual_task_zero_r1'
output.mkdir(exist_ok=False)
header=torch.load(source,map_location='cpu',weights_only=False)
original=load_soccer_policy(header,'cuda:0')
layer=DualTaskResidual(original.planner.obs_mean.cpu(),original.planner.obs_std.cpu())
header['dual_task_adapter']=dict(version=1,width=layer.width,model=layer.state_dict())
torch.save(header,output/'diagnostic.pt')
policy=load_dual_task_policy(output/'diagnostic.pt','cuda:0')
x=torch.load(root/'runs/planner_post_pass_latent_ppo_low_noise_r1/initial_state.pt',map_location='cuda:0',weights_only=True)['raw_observation'][:8]
checks=0
for skill,kicked in [(0,0),(0,1),(1,0),(1,0),(1,1),(0,1)]:
    probe=x.clone();probe[:,-2:]=0;probe[:,-2+skill]=1;probe[:,940]=kicked
    a=original(probe);b=policy(probe)
    assert all(torch.equal(a[k],b[k]) for k in a), (skill,kicked)
    checks+=len(x)
policy.task_adapter.requires_grad_(True)
for skill in range(2):
    policy.reset();probe=x.clone();probe[:,-2:]=0;probe[:,-2+skill]=1;probe[:,940]=0
    policy(probe)
    z=policy.task_adapter(policy.last_adapter_input,policy.last_base_latent)
    z[:,0].sum().backward()
    assert policy.task_adapter.heads[skill].net[-1].weight.grad.abs().sum()>0
    assert policy.task_adapter.heads[1-skill].net[-1].weight.grad.abs().sum()==0
    policy.task_adapter.zero_grad()
phase=DualTaskPhase(2,'cpu')
phase.previous_distance.fill_(1)
phase.begin_step()
phase.observe(torch.tensor([[2.,0.],[2.,0.]]),torch.tensor([[1.,0.],[1.,0.]]),torch.tensor([2.,2.]),torch.tensor([True,False]),torch.ones(2,dtype=torch.bool),torch.ones(2,dtype=torch.bool),torch.ones(2,dtype=torch.long))
r=phase.rewards(torch.ones(2),torch.ones(2),torch.zeros(2),torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),.02)
assert r[0,9]==15 and r[1,9]==0
phase.begin_step()
r=phase.rewards(torch.ones(2),torch.ones(2),torch.zeros(2),torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),.02)
assert not r[:,9].any(), 'Do not reward the same pass repeatedly'
result=dict(zero_equivalence_samples=checks,both_heads_receive_gradients=True,pass_event_reward_once=True)
(output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
