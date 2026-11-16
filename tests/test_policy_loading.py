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
