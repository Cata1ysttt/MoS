import pytest
import torch
from fiber_soccer.predicted_target_solver import refine


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.05),requires_grad=False)
    def forward(self,state,action,history,z):return z[:,:29]*self.weight


@pytest.mark.parametrize('mode',[torch.no_grad,torch.inference_mode])
def test_internal_target_refinement_with_frozen_actor(mode):
    actor=Actor();before=actor.weight.clone();rng=torch.random.get_rng_state()
    with mode():
        x=torch.zeros(2,2406);z=torch.zeros(2,256);z[:,32]=16
        desired=torch.ones(2,29)*.1;default=torch.zeros(29);scale=torch.ones(29)
        solved,info=refine(actor,x,z,desired,default,scale,64)
    assert info['final_mse']<info['initial_mse']*.01
    assert torch.allclose(solved.norm(dim=-1),torch.full((2,),16.),atol=2e-6)
    assert not solved.requires_grad and actor.weight.grad is None and torch.equal(actor.weight,before)
    assert torch.equal(torch.random.get_rng_state(),rng)
    # Independent per-state gradient scaling preserves each result across batch sizes.
    single,_=refine(actor,x[:1],z[:1],desired[:1],default,scale,64)
    torch.testing.assert_close(single,solved[:1])


def test_reject_trainable_actor_and_bad_step_count():
    actor=Actor();args=[torch.zeros(1,2406),torch.zeros(1,256),torch.zeros(1,29),torch.zeros(29),torch.ones(29)]
    with pytest.raises(ValueError):refine(actor,*args,0)
    actor.requires_grad_(True)
    with pytest.raises(ValueError):refine(actor,*args,4)
