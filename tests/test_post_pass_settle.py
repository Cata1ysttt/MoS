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
