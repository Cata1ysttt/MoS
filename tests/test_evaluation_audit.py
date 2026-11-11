"""Keep candidate target clipping and joint conversion observable."""
import torch
from types import SimpleNamespace
from fiber_soccer.bridge import Bridge

def test_unreachable_target_has_nonzero_roundtrip_error():
 b=Bridge.__new__(Bridge);b.order=[0,1];b.inverse=torch.tensor([0,1])
 b.term=SimpleNamespace(_scale=torch.tensor([[.1,.1]]),_offset=torch.zeros(1,2),cfg=SimpleNamespace(clip=None))
 desired=torch.tensor([[2.,.2]])
 actual=b.expert_targets(b.env_action(desired))
 torch.testing.assert_close(actual,torch.tensor([[1.,.2]]))
 assert (actual-desired).abs().amax()==1
