import pytest
import torch
from fiber_soccer.lpp import project,slerp,alignment_loss,objective

def test_sphere_and_endpoints():
 a,b=project(torch.randn(4,256)),project(torch.randn(4,256))
 torch.testing.assert_close(slerp(a,b,0),a)
 torch.testing.assert_close(slerp(a,b,1),b)
 torch.testing.assert_close(slerp(a,b,.4).norm(dim=-1),torch.full((4,),16.))
 assert torch.isfinite(project(torch.zeros(1,256))).all()

def test_terminal_mask_and_units():
 target=torch.zeros(2,5,29);prediction=torch.ones_like(target);prediction[:,3:]=999
 mask=torch.tensor([[1,1,1,0,0.]]*2)
 assert alignment_loss(target,prediction,mask)==1

def test_missing_encoder_is_error():
 with pytest.raises(ValueError,match='official B'):
  objective(torch.zeros(1,5,29),torch.zeros(1,5,29),torch.ones(1,5),torch.zeros(1,256),mode='full_lpp')


def test_execution_mode_preserves_hold_endpoint_and_gradient():
    from fiber_soccer.lpp import execute_latent, project
    previous=project(torch.randn(2,256))
    endpoint=torch.randn(2,256,requires_grad=True)
    for alpha in [.2,.6,1.]:
        torch.testing.assert_close(execute_latent(previous,endpoint,alpha,'hold'),project(endpoint))
    execute_latent(previous,endpoint,.2,'hold')[:,0].sum().backward()
    assert endpoint.grad is not None and torch.isfinite(endpoint.grad).all()
    torch.testing.assert_close(execute_latent(previous,endpoint,1.,'slerp'),project(endpoint),atol=1e-5,rtol=1e-5)
