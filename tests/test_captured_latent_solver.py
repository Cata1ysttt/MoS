import pytest
import torch
from fiber_soccer.captured_latent_solver import fixed_adam,CapturedLatentSolver
from fiber_soccer.latent_oracle import fit_latent


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(256,29,bias=False).requires_grad_(False)
    def forward(self,state,z):return self.layer(z).tanh()


def test_fixed_loop_preserves_reference_best_and_stop_masks():
    torch.manual_seed(816031);actor=Actor();state=(torch.zeros(4,1),)
    z=torch.nn.functional.normalize(torch.randn(4,256),dim=-1)*16
    desired=torch.zeros(4,29)
    with torch.no_grad():desired[0]=actor(state[0],z[:1])[0]
    before={k:v.clone() for k,v in actor.state_dict().items()}
    ref,info=fit_latent(actor,state,desired,z,lambda x:x,iterations=64)
    actual,initial,error,updates,finite=fixed_adam(actor,state,desired,z,lambda x:x,64)
    assert finite and updates[0]==0 and updates.max()==info['max_state_updates']
    torch.testing.assert_close(actual,ref,rtol=1e-4,atol=3e-6)
    torch.testing.assert_close(actual[0],z[0])
    assert float(error.mean())==pytest.approx(info['final_mse'],rel=1e-4,abs=1e-8)
    assert all(p.grad is None for p in actor.parameters())
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())


def test_captured_solver_rejects_trainable_actor_and_cpu():
    actor=Actor()
    with pytest.raises(ValueError):CapturedLatentSolver(actor,lambda x:x,0)
    solver=CapturedLatentSolver(actor,lambda x:x)
    with pytest.raises(ValueError):solver.solve((torch.zeros(1,1),),torch.zeros(1,29),torch.zeros(1,256))
    actor.requires_grad_(True)
    with pytest.raises(ValueError):CapturedLatentSolver(actor,lambda x:x)
