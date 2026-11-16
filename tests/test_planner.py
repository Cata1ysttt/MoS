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


def test_new_schema_checkpoint_preserves_phase_and_anchor_input_width():
    from fiber_soccer.planner import SCHEMA_V2
    model=LatentPlanner(horizon=2,width=32,schema=SCHEMA_V2)
    x=torch.zeros(2,950);x[0,939:944]=torch.tensor([1.,0.,.2,-.1,.3]);x[:,-1]=1
    restored=load_planner(dict(planner=model.state_dict(),observation_schema=SCHEMA_V2,horizon=2,width=32),'cpu')
    torch.testing.assert_close(restored(x),model(x))
    assert restored.schema==SCHEMA_V2 and restored.obs_dim==950


def test_backward_residual_planner_starts_at_prior_and_keeps_old_schema_loadable():
    from fiber_soccer.planner import SCHEMA_V3
    from fiber_soccer.lpp import project
    m=LatentPlanner(horizon=4,width=32,schema=SCHEMA_V3)
    x=torch.randn(2,1206);prior=project(torch.randn(2,256));x[:,-262:-6]=prior
    torch.testing.assert_close(m(x),prior[:,None].expand(-1,4,-1))
    ck=dict(planner=m.state_dict(),observation_schema=SCHEMA_V3,horizon=4,width=32)
    torch.testing.assert_close(load_planner(ck,'cpu')(x),m(x))


def test_label_confidence_weights_correct_future_frame_without_crossing_reset():
    from fiber_soccer.planner import trajectory_windows
    x = torch.arange(5.).reshape(5, 1, 1)
    episodes = torch.tensor([[0], [0], [1], [1], [1]])
    confidence = torch.tensor([[1.], [.5], [.2], [.8], [.9]])
    _, _, weights, _ = trajectory_windows(x, x, episodes, x, 3, confidence)
    torch.testing.assert_close(weights[:, 0], torch.tensor([[1., .5, 0.], [.5, 0., 0.], [.2, .8, .9]]))
    import pytest
    with pytest.raises(ValueError, match="finite"):
        trajectory_windows(x, x, episodes, x, 3, torch.full_like(confidence, float('nan')))


def test_label_confidence_does_not_change_pass_shoot_task_balance():
    from fiber_soccer.planner import balanced_skill_mean
    values = torch.tensor([1., 1., 10., 10.], requires_grad=True)
    weights = torch.tensor([1., 1., .01, .01])
    loss = balanced_skill_mean(values, weights)
    torch.testing.assert_close(loss, torch.tensor(5.5))
    loss.backward()
    torch.testing.assert_close(values.grad, torch.full((4,), .25))
