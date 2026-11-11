import torch
from fiber_soccer.pass_measurement import first_kick_samples


def test_measurement_gates_and_units():
    v=torch.tensor([[2.,0.],[0.,2.],[.1,0.],[2.,0.],[2.,0.]])
    d=torch.tensor([[3.,0.]]).expand(5,-1)
    mask,values=first_kick_samples(v,d,torch.ones(5)*2,torch.tensor([1,1,1,0,1]).bool(),torch.tensor([1,1,1,1,0]).bool())
    assert mask.tolist()==[True,True,False,False,False]
    torch.testing.assert_close(values[:2,2],torch.tensor([0.,90.]))
    torch.testing.assert_close(values[:2,4],torch.tensor([0.,1.]))
