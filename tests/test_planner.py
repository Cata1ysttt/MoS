import torch
from fiber_soccer.planner import LatentPlanner,load_planner,SCHEMA,OBS_DIM


def test_planner_shared_call_handles_mixed_skills_and_normalizes_latent():
    torch.manual_seed(7)
    model=LatentPlanner(horizon=3,width=32).eval()
    x=torch.randn(4,OBS_DIM);x[:,-2:]=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    with torch.no_grad():
        z=model(x)
        assert z.shape==(4,3,256)
        torch.testing.assert_close(z.norm(dim=-1),torch.full((4,3),16.))
        torch.testing.assert_close(model.act(x),z[:,0])
        for i in range(4):torch.testing.assert_close(model(x[i:i+1]),z[i:i+1],atol=1e-5,rtol=1e-5)
    ck=dict(planner=model.state_dict(),observation_schema=SCHEMA,horizon=3,width=32)
    loaded=load_planner(ck,'cpu')
    torch.testing.assert_close(loaded(x),z)


def test_planner_rejects_incompatible_observation_contract():
    import pytest
    with pytest.raises(ValueError,match='schema'):
        load_planner(dict(observation_schema='legacy_padding'),'cpu')


def test_future_labels_never_cross_reset_and_keep_alignment():
    from fiber_soccer.planner import trajectory_windows
    x=torch.arange(6.).reshape(6,1,1)
    episode=torch.tensor([[0],[0],[1],[1],[1],[2]])
    current,future,mask,target=trajectory_windows(x,x+100,episode,x+200,3)
    assert current[:,0,0].tolist()==[0,1,2,3]
    assert future[1,0,:,0].tolist()==[101,102,103]
    assert mask[:,0].tolist()==[[True,True,False],[True,False,False],[True,True,True],[True,True,False]]
    assert target[:,0,0].tolist()==[200,201,202,203]
