import json
import pytest
import torch
from fiber_soccer import planner_policy
from fiber_soccer.planner import SCHEMA_V4
from fiber_soccer.predicted_target_solver import PredictedTargetSolverPolicy


class Predictor(torch.nn.Module):
    def bounded_targets(self,x):
        return (x[:,-1:]*.1+x[:,-3:-2]*.03+x[:,-6:-5]*.02).expand(-1,29)


class Prior(torch.nn.Module):
    def act(self,x):
        z=x.new_zeros((len(x),256));z[:,40]=16
        return z


class Planner(torch.nn.Module):
    obs_dim=2406;schema=SCHEMA_V4
    def __init__(self):
        super().__init__();self.target_predictor=Predictor();self.prior=Prior()
    def act(self,x):return self.prior.act(x)
    def decode(self,x,prior,desired):return prior


class Actor(torch.nn.Module):
    def __init__(self,unused):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.05))
    def forward(self,state,last,history,z):return self.weight*z[:,:29]


@pytest.fixture
def runtime(monkeypatch,tmp_path):
    (tmp_path/'actor_meta.json').write_text(json.dumps(dict(joint_names=list(range(29)),action_rescale=[1.]*29)))
    (tmp_path/'pickup_scene.json').write_text(json.dumps(dict(default_q=[0.]*29)))
    monkeypatch.setattr(planner_policy,'ASSETS',tmp_path)
    monkeypatch.setattr(planner_policy,'load_planner',lambda checkpoint,device:Planner())
    monkeypatch.setattr(planner_policy,'load_native',lambda device:None)
    monkeypatch.setattr(planner_policy,'NativeActor',Actor)
    return dict(planner_arch='target_guided_latent_v1',predicted_target_solver_steps=4)


def test_factory_selects_declared_policy_and_rejects_silent_fallback(runtime,tmp_path):
    with pytest.raises(ValueError,match='load_soccer_policy'):planner_policy.SoccerPolicy(runtime)
    path=tmp_path/'policy.pt';torch.save(runtime,path)
    assert isinstance(planner_policy.load_soccer_policy(path),PredictedTargetSolverPolicy)
    ordinary={k:v for k,v in runtime.items() if k!='predicted_target_solver_steps'}
    assert type(planner_policy.load_soccer_policy(ordinary)) is planner_policy.SoccerPolicy
    with pytest.raises(ValueError):planner_policy.load_soccer_policy(dict(runtime,predicted_target_solver_steps=0))
    with pytest.raises(ValueError):planner_policy.load_soccer_policy(dict(ordinary,predicted_target_solver_backend='cuda_graph'))
    with pytest.raises(ValueError):planner_policy.load_soccer_policy(dict(ordinary,predicted_target_refinement_scope='initial_pass'))


def test_same_command_interface_changes_internal_targets_without_mutating_inputs(runtime):
    policy=planner_policy.load_soccer_policy(runtime)
    x=torch.zeros(2,2406);x[:,-2]=1;before=x.clone()
    skill=torch.tensor([0,1]);direction=torch.tensor([[1.,0.,0.],[0.,1.,0.]]);speed=torch.tensor([1.,2.])
    actual=policy(x,skill=skill,target_direction=direction,target_speed=speed)
    conditioned=x.clone();conditioned[:,-2:]=torch.eye(2)
    conditioned[:,-6:-3]=direction;conditioned[:,-3]=speed
    explicit=policy(conditioned)
    assert torch.equal(x,before)
    assert torch.equal(actual['latent'],explicit['latent'])
    assert torch.equal(actual['joint_targets'],explicit['joint_targets'])
    assert not torch.equal(actual['latent'],policy(x)['latent'])
    assert all(p.grad is None for p in policy.parameters())
    with pytest.raises(ValueError):policy(x,skill=2)
    with pytest.raises(ValueError):policy(x[:,:10])


def test_factory_can_load_refinement_inside_inference_context(runtime):
    with torch.inference_mode():
        policy=planner_policy.load_soccer_policy(runtime)
        x=torch.zeros(2,2406);x[:,-1]=1
        result=policy(x,target_speed=2.)
        assert torch.isfinite(result['joint_targets']).all()
    assert not policy.actor.weight.is_inference()
    assert policy.actor.weight.grad is None


def test_scoped_refinement_preserves_pass_and_uses_shared_prior_after_shoot(runtime):
    source=planner_policy.load_soccer_policy(runtime)
    candidate=planner_policy.load_soccer_policy(dict(runtime,predicted_target_refinement_scope='initial_pass'))
    x=torch.zeros(2,2406);x[:,-2]=1;x[:,-3]=2
    assert torch.equal(candidate(x)['latent'],source(x)['latent'])
    mixed=x.clone();mixed[0,-2:]=torch.tensor([0.,1.])
    actual=candidate(mixed);reference=source(mixed)
    assert torch.equal(actual['latent'][0],candidate.planner.prior.act(mixed)[0])
    assert torch.equal(actual['latent'][1],reference['latent'][1])
    recovered=candidate(x)
    assert torch.equal(recovered['latent'][0],candidate.planner.prior.act(x)[0])
    candidate.reset(torch.tensor([True,False]))
    assert torch.equal(candidate(x)['latent'],source(x)['latent'])
    candidate(x,skill='shoot');result=candidate(x)
    assert torch.equal(result['latent'],candidate.planner.prior.act(x))
    assert candidate.solver_records[-1]['refined_environments']==0
    assert candidate.solver_records[-1]['final_mse'] is None
    candidate.reset();assert torch.equal(candidate(x)['latent'],source(x)['latent'])


def test_initial_goal_recovery_preserves_kicks_and_only_changes_shoot_origin_recovery(runtime):
    metadata=dict(runtime,predicted_target_refinement_scope='initial_pass')
    source=planner_policy.load_soccer_policy(metadata)
    candidate=planner_policy.load_soccer_policy(dict(metadata,standing_recovery_mode='initial_backward_goal'))
    x=torch.zeros(2,2406);x[:,944]=16;x[:,-2]=1;x[:,-3]=2
    saved=x.clone()
    with torch.inference_mode():
        for contact,skill in [(False,'pass'),(True,'pass'),(False,'shoot'),(True,'shoot')]:
            x[:,940]=float(contact)
            a=candidate(x,skill=skill);b=source(x,skill=skill)
            assert all(torch.equal(a[k],b[k]) for k in b)
        # Goal stays the reset goal, even though the current backward feature changes.
        x[:,944]=0;x[:,945]=16
        a=candidate(x,skill=torch.tensor([0,1]));b=source(x,skill=torch.tensor([0,1]))
        assert torch.equal(a['latent'][0],saved[0,944:1200])
        assert all(torch.equal(a[k][1],b[k][1]) for k in b)
        expected=candidate.actor(x[:,:64],x[:,64:93],x[:,93:930],saved[:,944:1200])
        assert torch.equal(a['action_normalized'][0],expected[0])
        assert candidate.solver_records[-1]['standing_recovery_environments']==1
    # Reset is legal outside inference_mode and clears only the selected lane.
    candidate.reset(torch.tensor([True,False]));source.reset(torch.tensor([True,False]))
    a=candidate(x,skill='pass');b=source(x,skill='pass')
    assert torch.equal(a['latent'][0],b['latent'][0])
    assert torch.equal(a['latent'][1],saved[1,944:1200])
    assert torch.equal(candidate.initial_goal.goal[0],x[0,944:1200])
    assert not candidate.initial_goal.state_dict()
    candidate.reset();assert candidate.initial_goal.goal is None


def test_initial_goal_factory_rejects_invalid_metadata_and_direct_bypass(runtime):
    metadata=dict(runtime,predicted_target_refinement_scope='initial_pass',standing_recovery_mode='initial_backward_goal')
    with pytest.raises(ValueError,match='load_soccer_policy'):planner_policy.SoccerPolicy(metadata)
    with pytest.raises(ValueError,match='load_soccer_policy'):PredictedTargetSolverPolicy(metadata)
    with pytest.raises(ValueError):planner_policy.load_soccer_policy(dict(metadata,standing_recovery_mode='unknown'))
    with pytest.raises(ValueError):planner_policy.load_soccer_policy(dict(metadata,predicted_target_refinement_scope='all'))
