import pytest
import torch
from fiber_soccer.post_pass_settle import PostPassSettleGate
from fiber_soccer.planner_policy import load_soccer_policy


def test_ten_step_followthrough_contact_origin_and_handoff():
    gate=PostPassSettleGate(10);skills=torch.tensor([0,1,0]);contact=torch.tensor([False,False,False])
    with torch.inference_mode():
        assert not gate(contact,skills).any()
        contact[:]=True
        for _ in range(10):assert not gate(contact,skills).any()
        assert gate(contact,skills).tolist()==[True,False,True]
        skills[:]=1;contact[:]=False
        assert not gate(contact,skills).any()  # New shooting ball.
        contact[:]=True;gate(contact,skills);skills[:]=0
        for _ in range(12):assert not gate(contact,skills).any()  # Shooting recovery is not passing contact.
    gate.reset();skills[:]=0
    for _ in range(11):result=gate(contact,skills)
    assert result.all()
    gate.reset(torch.tensor([True,False,False]))
    assert gate(contact,skills).tolist()==[False,True,True]
    contact[1]=False;assert gate(contact,skills).tolist()==[False,False,True]
    assert not gate.state_dict()


def test_invalid_settle_configuration():
    for value in [0,-1,True,1.5]:
        with pytest.raises(ValueError):PostPassSettleGate(value)
    with pytest.raises(ValueError,match='delay'):load_soccer_policy({'post_pass_settle_delay_steps':True})
    with pytest.raises(ValueError,match='combine'):load_soccer_policy({'post_pass_settle_delay_steps':10,'post_pass_latent_adapter':{}})


def test_settle_audit_uses_live_actions_and_includes_failure_step(tmp_path,monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    scripts=Path(__file__).parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('settle_execution_audit',scripts/'run_post_pass_settle_study.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    summary=dict(steps=20,post_pass_settle_delay_steps=10,
        contact_observations=[dict(stage=0,environment=0,tick=0),dict(stage=0,environment=1,tick=1)],
        switches=[dict(tick=15,environments=[0])],first_resets=[dict(environment=1,tick=11)],
        post_pass_settle_active_control_steps=6)
    expected=[int(10<=tick<15)+int(tick==11) for tick in range(20)]
    records=[dict(post_pass_settle_live_environments=value,post_pass_settle_environments=value+int(tick>=12))
        for tick,value in enumerate(expected)]
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    path=tmp_path/'predicted_target_solver.json';path.write_text(json.dumps(records))
    audit=module.settle_audit(tmp_path)
    assert audit['active_control_steps']==6 and audit['first_activation_tick']==10
    assert audit['active_steps_per_environment']=={'0':5,'1':1}
    records[15]['post_pass_settle_live_environments']=1;path.write_text(json.dumps(records))
    with pytest.raises(AssertionError,match='Live settling'):module.settle_audit(tmp_path)


def test_valid_kick_waits_for_ball_speed_then_latches_and_resets():
    gate=PostPassSettleGate(2,trigger='valid_kick')
    skills=torch.tensor([0,0,1]);contact=torch.tensor([True,False,True])
    speed=torch.tensor([.249,1.,1.])
    with torch.inference_mode():
        # Neither weak contact, a fast incoming ball, nor shooting enables settling.
        for _ in range(20):assert not gate(contact,skills,speed).any()
        speed[0]=.25
        assert not gate(contact,skills,speed).any()  # Valid-event age zero.
        speed[0]=0
        assert not gate(contact,skills,speed).any()  # Latched despite deceleration.
        assert gate(contact,skills,speed).tolist()==[True,False,False]
    gate.reset(torch.tensor([True,False,False]))
    assert not gate(contact,skills,speed).any()
    speed[0]=.3
    for _ in range(3):active=gate(contact,skills,speed)
    assert active.tolist()==[True,False,False]
    contact[0]=False;assert not gate(contact,skills,speed).any()
    contact[0]=True;speed[0]=0
    for _ in range(5):assert not gate(contact,skills,speed).any()
    skills[2]=0
    for _ in range(5):assert not gate(contact,skills,speed).any()  # Shoot-origin contact stays excluded.
    gate.reset();assert not gate.state_dict()


def test_valid_kick_threshold_matches_measurement():
    from fiber_soccer.pass_measurement import first_kick_samples
    speeds=torch.tensor([0.,.249,.25,.251,1.])
    kicked=torch.ones(5,dtype=torch.bool);skills=torch.zeros(5,dtype=torch.long)
    velocity=torch.stack((speeds,torch.zeros_like(speeds)),dim=-1)
    valid,_=first_kick_samples(velocity,torch.ones_like(velocity),torch.ones_like(speeds),kicked,kicked)
    gate=PostPassSettleGate(1,'valid_kick')
    gate(kicked,skills,speeds)
    assert torch.equal(gate(kicked,skills,speeds),valid)
    for speed in [None,torch.ones(4),torch.tensor([0.,0.,0.,0.,float('nan')]),-torch.ones(5)]:
        with pytest.raises(ValueError):gate(kicked,skills,speed)
    with pytest.raises(ValueError):PostPassSettleGate(10,'unknown')
    with pytest.raises(ValueError,match='delay'):load_soccer_policy({'post_pass_settle_trigger':'valid_kick'})


def test_valid_kick_settle_audit_ignores_weak_contact(tmp_path,monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    scripts=Path(__file__).parents[1]/'scripts';monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('valid_settle_audit',scripts/'run_post_pass_settle_study.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    summary=dict(steps=25,post_pass_settle_delay_steps=10,post_pass_settle_trigger='valid_kick',
        contact_observations=[dict(stage=0,environment=0,tick=0),dict(stage=0,environment=1,tick=0)],
        pass_impacts={'0':dict(tick=5)},switches=[dict(tick=20,environments=[0])],first_resets=[],
        post_pass_settle_active_control_steps=5)
    records=[dict(post_pass_settle_live_environments=int(15<=t<20),post_pass_settle_environments=int(15<=t<20)) for t in range(25)]
    (tmp_path/'summary.json').write_text(json.dumps(summary));p=tmp_path/'predicted_target_solver.json';p.write_text(json.dumps(records))
    result=module.settle_audit(tmp_path)
    assert result['active_steps_per_environment']=={'0':5} and result['first_activation_tick']==15
    records[10]['post_pass_settle_live_environments']=1;p.write_text(json.dumps(records))
    with pytest.raises(AssertionError,match='Live settling'):module.settle_audit(tmp_path)
