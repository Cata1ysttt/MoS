import copy
import pytest
import torch
from fiber_soccer.ppo_resume import restore_learning_state,assert_tree_equal,validate_resume,MATCHED_OPTIONS


def test_resume_preserves_next_optimizer_update_and_exploration():
    torch.manual_seed(41)
    model=torch.nn.Linear(4,3);critic=torch.nn.Linear(4,1)
    opt=torch.optim.Adam(model.parameters(),lr=.002);vopt=torch.optim.Adam(critic.parameters(),lr=.003)
    generator=torch.Generator().manual_seed(76)
    x=torch.randn(8,4);target=torch.randn(8,3)
    def step(m,v,o,vo):
        o.zero_grad();(m(x)-target).square().mean().backward();o.step()
        vo.zero_grad();(v(x)-target[:,:1]).square().mean().backward();vo.step()
    for _ in range(3):step(model,critic,opt,vopt)
    state=copy.deepcopy(dict(critic=critic.state_dict(),optimizer=opt.state_dict(),critic_optimizer=vopt.state_dict(),generator_state=generator.get_state()))
    payload=copy.deepcopy(dict(planner=model.state_dict()))
    resumed=torch.nn.Linear(4,3);rv=torch.nn.Linear(4,1)
    ro=torch.optim.Adam(resumed.parameters());rvo=torch.optim.Adam(rv.parameters());rg=torch.Generator()
    result=restore_learning_state(resumed,rv,ro,rvo,rg,payload,state,901)
    assert result['exploration_rng_restored'] and not result['simulator_restored']
    assert torch.equal(torch.rand(10,generator=generator),torch.rand(10,generator=rg))
    step(model,critic,opt,vopt);step(resumed,rv,ro,rvo)
    assert_tree_equal(model.state_dict(),resumed.state_dict())
    assert_tree_equal(critic.state_dict(),rv.state_dict())
    assert_tree_equal(opt.state_dict(),ro.state_dict())


def test_manifest_rejects_changed_base_objective_and_smoke_status():
    current={key:1 for key in MATCHED_OPTIONS};current['smoke_test']=False
    config=dict(current,training='passing_latent_ppo_v1',parent_sha256='base')
    complete=dict(iterations=5,total_iterations=10,total_environment_steps=100,
        smoke_test=False,shared_tensors_unchanged=True,deployable_roundtrip=True)
    assert validate_resume(config,complete,current,'base')==dict(iterations=10,environment_steps=100)
    with pytest.raises(ValueError,match='base'):validate_resume(config,complete,current,'other')
    with pytest.raises(ValueError,match='reference_reward_scale'):
        validate_resume(config,complete,dict(current,reference_reward_scale=0),'base')
    with pytest.raises(ValueError,match='smoke_test'):
        validate_resume(config,complete,dict(current,smoke_test=True),'base')


def test_resume_matches_training_scene_and_accepts_legacy_passing_default():
    current={key:1 for key in MATCHED_OPTIONS}
    current.update(smoke_test=False,training_scene='pass')
    config=dict(current,training='passing_latent_ppo_v1',parent_sha256='base')
    del config['training_scene']
    complete=dict(iterations=5,smoke_test=False,shared_tensors_unchanged=True,
                  deployable_roundtrip=True)
    assert validate_resume(config,complete,current,'base')['iterations']==5
    with pytest.raises(ValueError,match='training_scene'):
        validate_resume(config,complete,dict(current,training_scene='shoot'),'base')
