import torch
from fiber_soccer.latent_oracle import fit_latent


def test_action_inversion_improves_targets_without_changing_actor():
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.eye(29,256),requires_grad=False)
        def forward(self,state,z):
            return z@self.weight.T+state
    actor=Actor();original=actor.weight.clone()
    previous=torch.zeros(2,256);previous[:,-1]=16
    target=torch.full((2,29),.2)
    z,report=fit_latent(actor,(torch.zeros(2,29),),target,previous,lambda x:x,iterations=80)
    assert report['final_mse']<report['initial_mse']*.01
    torch.testing.assert_close(z.norm(dim=-1),torch.full((2,),16.))
    assert actor.weight.grad is None and torch.equal(actor.weight,original)
