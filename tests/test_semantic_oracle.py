import torch
from fiber_soccer.semantic_oracle import fit_regularized_latent


def test_regularized_label_keeps_norm_improves_joint_objective_and_freezes_actor():
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.eye(2),requires_grad=False)
        def forward(self,s,last,h,z):return z[:,:2]@self.weight/16
    actor=Actor();initial=torch.zeros(1,256);initial[:,0]=16
    prior=torch.zeros_like(initial);prior[:,1]=16
    target=torch.tensor([[1.,0.]])
    z,audit=fit_regularized_latent(actor,(None,None,None),target,initial,prior,lambda a:a,weight=.1,iterations=30)
    torch.testing.assert_close(z.norm(dim=-1),torch.tensor([16.]))
    initial_objective=.1*(initial-prior).square().mean()
    assert audit['objective']<float(initial_objective)
    assert audit['semantic_mse']<float((initial-prior).square().mean())
    assert actor.weight.grad is None and not z.requires_grad


def test_independent_regularized_latent_solve_does_not_depend_on_batch_size():
    class Actor(torch.nn.Module):
        def forward(self, state, z):
            return z[:,:4]*.01
    actor=Actor()
    initial=torch.zeros(1,256);initial[:,-1]=16
    prior=torch.zeros_like(initial);prior[:,0]=16
    target=torch.full((1,4),.12)
    one,_=fit_regularized_latent(actor,(None,),target,initial,prior,lambda x:x,weight=.01,iterations=40)
    many,_=fit_regularized_latent(actor,(None,),target.repeat(256,1),initial.repeat(256,1),prior.repeat(256,1),lambda x:x,weight=.01,iterations=40)
    torch.testing.assert_close(many,one.expand_as(many),atol=1e-5,rtol=1e-5)
