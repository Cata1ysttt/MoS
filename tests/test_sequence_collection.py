import pytest
import torch
from fiber_soccer.sequence_collection import SequenceShootingRecorder


def test_passive_recorder_keeps_only_alive_shooting_slots_and_exact_inputs():
    recorder=SequenceShootingRecorder(4,list(reversed(range(29))))
    x=torch.randn(4,2406);x[:,-2:]=torch.tensor([[1.,0.],[0.,1.],[0.,1.],[0.,1.]])
    z=torch.randn(4,256);alive=torch.tensor([True,True,False,True])
    scale=torch.arange(116.).reshape(4,29);offset=scale+2
    original=x.clone();rng=torch.random.get_rng_state()
    recorder.record(7,x,z,alive,scale,offset)
    assert torch.equal(rng,torch.random.get_rng_state()) and torch.equal(x,original)
    x.fill_(999);z.fill_(999);scale.fill_(999);offset.fill_(999)
    data=recorder.export(dict(seed=123,purpose='diagnostic'))
    assert data['environment'].tolist()==[1,3] and data['tick'].tolist()==[7,7]
    assert torch.equal(data['x'],original[[1,3]])
    assert data['action_scale'][0].tolist()==list(reversed(range(29,58)))
    assert not data['metadata']['expert_loaded'] and not data['metadata']['expert_actions_executed']
    with pytest.raises(ValueError,match='strictly increase'):
        recorder.record(7,original,z,alive,scale,offset)


def test_recorder_ignores_passing_and_cannot_export_empty_collection():
    recorder=SequenceShootingRecorder(2,list(range(29)))
    x=torch.zeros(2,2406);x[:,-2]=1
    recorder.record(0,x,torch.zeros(2,256),torch.ones(2,dtype=torch.bool),1.,0.)
    with pytest.raises(ValueError,match='No alive shooting'):
        recorder.export({})
