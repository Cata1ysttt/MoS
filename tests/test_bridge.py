from types import SimpleNamespace
import torch
from fiber_soccer.bridge import Bridge

def test_joint_order_and_physical_clipping_roundtrip():
 b=Bridge.__new__(Bridge)
 b.order=[2,0,1];b.inverse=torch.argsort(torch.tensor(b.order))
 b.term=SimpleNamespace(_scale=torch.tensor([[.2,.3,.4]]),_offset=torch.tensor([[.1,-.2,.3]]),cfg=SimpleNamespace(clip=None))
 raw=torch.tensor([[1.,-2.,3.]])
 physical=b.expert_targets(raw)
 torch.testing.assert_close(b.env_action(physical),raw)
 b.term.cfg.clip={};b.term._clip=torch.tensor([[[-.5,.5],[-.5,.5],[-.5,.5]]])
 assert b.expert_targets(torch.ones(1,3)*100).max()<=.5

def test_history_is_group_major_and_reset_local():
 b=Bridge.__new__(Bridge);b.last=torch.zeros(2,29);b.history=torch.ones(2,10,93)
 b.ids=list(range(29));b.default=torch.zeros(29)
 b.robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(2,29),joint_vel=torch.zeros(2,29),root_ang_vel_b=torch.zeros(2,3),projected_gravity_b=torch.tensor([[0.,0.,-1.]]*2)))
 b.advance(torch.zeros(2,29),torch.tensor([True,False]))
 _,_,h=b.observe()
 assert h[0].count_nonzero()==0 and torch.all(h[1]==1)
