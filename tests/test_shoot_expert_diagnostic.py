import numpy as np
import pytest
import torch
from fiber_soccer.shoot_expert_diagnostic import ShootingExpertDiagnostic


class Session:
    def __init__(self):self.inputs=[]
    def run(self,outputs,inputs):
        assert outputs==['actions'] and inputs['obs'].shape==(1,1030) and inputs['rma_obs'].shape==(1,150)
        assert inputs['time_step'].shape==(1,1) and inputs['time_step'].dtype==np.float32
        self.inputs.append({k:v.copy() for k,v in inputs.items()})
        return [np.full((1,29),7.+len(self.inputs),dtype=np.float32)]


def obs():
    x=torch.arange(3*2406,dtype=torch.float32).reshape(3,2406)
    x[:,940]=0;x[:,-2:]=torch.tensor([1.,0.])
    return x


def test_raw_expert_called_only_for_shoot_rows_and_not_clamped():
    session=Session();controller=ShootingExpertDiagnostic(session=session)
    x=obs();x[[0,2],-2:]=torch.tensor([0.,1.]);before=x.clone()
    action=controller.act(x)
    assert len(session.inputs)==2 and torch.equal(x,before)
    assert torch.all(action[0]==8) and torch.all(action[1]==0) and torch.all(action[2]==9)
    np.testing.assert_array_equal(session.inputs[0]['obs'],x[0:1,1200:2230].numpy())
    np.testing.assert_array_equal(session.inputs[1]['rma_obs'],x[2:3,2230:2380].numpy())


def test_full_history_advances_during_pass_and_preserves_native_shoot():
    session=Session();controller=ShootingExpertDiagnostic(session=session,history_mode='initialize_shoot_ball_task')
    x=obs()
    with torch.inference_mode():
        controller.act(x);x[1,940]=1;controller.act(x)
        assert not session.inputs
        x[:,940]=0;x[:,-2:]=torch.tensor([0.,1.]);before=x.clone();controller.act(x)
        assert torch.equal(x,before)
        for i in [0,2]:np.testing.assert_array_equal(session.inputs[i]['obs'],x[i:i+1,1200:2230].numpy())
        assert controller.records[-1]['new_ball_history_environments']==1
        assert controller.records[-1]['new_ball_task_history_environments']==1
        expected=x[1,933:936].numpy()
        np.testing.assert_array_equal(session.inputs[1]['obs'][:,930:960].reshape(10,3),np.tile(expected,(10,1)))
        np.testing.assert_array_equal(session.inputs[1]['rma_obs'].reshape(50,3),np.tile(np.r_[expected[:2],1.],(50,1)))
    controller.reset(torch.tensor([False,True,False]));controller.act(x)
    assert controller.records[-1]['new_ball_task_history_environments']==0
    controller.reset();controller.act(x)
    assert controller.records[-1]['new_ball_history_environments']==0


def test_diagnostic_rejects_ambiguous_configuration_and_bad_action():
    with pytest.raises(ValueError):ShootingExpertDiagnostic()
    with pytest.raises(ValueError):ShootingExpertDiagnostic('unused',session=Session())
    with pytest.raises(ValueError):ShootingExpertDiagnostic(session=Session(),history_mode='unknown')
    controller=ShootingExpertDiagnostic(session=Session())
    with pytest.raises(ValueError):controller.act(obs().double())
    controller.session.run=lambda *args:[np.full((1,29),np.nan,dtype=np.float32)]
    x=obs();x[:,-2:]=torch.tensor([0.,1.])
    with pytest.raises(ValueError):controller.act(x)
