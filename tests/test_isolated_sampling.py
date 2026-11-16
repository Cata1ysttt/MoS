import pytest
import torch
from fiber_soccer.isolated_sampling import event_seed,isolated_torch_rng


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_samples_ignore_call_order_and_restore_generators(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(123)
    cpu_before=torch.random.get_rng_state()
    gpu_before=torch.cuda.get_rng_state() if device=='cuda' else None
    def draw(environment):
        with isolated_torch_rng(event_seed(93,environment),device):
            return torch.rand(12,device=device),torch.rand(3)
    original={i:draw(i) for i in [0,1,2]}
    assert torch.equal(cpu_before,torch.random.get_rng_state())
    if gpu_before is not None:assert torch.equal(gpu_before,torch.cuda.get_rng_state())
    for i in [2,0,1]:
        torch.rand(100+i,device=device)
        for old,new in zip(original[i],draw(i)):assert torch.equal(old,new)
    before=torch.random.get_rng_state()
    before_gpu=torch.cuda.get_rng_state() if device=='cuda' else None
    with pytest.raises(RuntimeError):
        with isolated_torch_rng(712,device):
            torch.rand(9,device=device)
            raise RuntimeError('sampling failed')
    assert torch.equal(before,torch.random.get_rng_state())
    if before_gpu is not None:assert torch.equal(before_gpu,torch.cuda.get_rng_state())


def test_seed_is_stable_and_distinguishes_environment_and_event():
    assert event_seed(93,7)==event_seed(93,7)
    assert len({event_seed(93,i,j) for i in range(16) for j in range(3)})==48
    assert event_seed(93,7)!=event_seed(94,7)
    with pytest.raises(ValueError):event_seed(93,-1)
