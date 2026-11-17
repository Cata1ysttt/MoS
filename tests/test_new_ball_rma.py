import pytest
import torch
from fiber_soccer.new_ball_rma import NewBallRmaHistory,initialize_new_ball_history,START,END


def observation(n=2):
    x=torch.arange(n*2406,dtype=torch.float32).reshape(n,2406)
    x[:,940]=0;x[:,-2:]=torch.tensor([1.,0.]);x[:,933:935]=torch.tensor([2.,-.4])
    return x


def test_only_pre_object_samples_replaced_and_body_history_preserved():
    x=observation(4);before=x.clone();age=torch.tensor([0,1,49,50]);anchor=torch.tensor([[2.,3.,1.]]).expand(4,-1)
    y=initialize_new_ball_history(x,age,anchor)
    assert torch.equal(x,before)
    assert torch.equal(y[:,:START],x[:,:START]) and torch.equal(y[:,END:],x[:,END:])
    old=x[:,START:END].reshape(4,50,3);new=y[:,START:END].reshape(4,50,3)
    for i,count in enumerate([50,49,1,0]):
        assert torch.equal(new[i,:count],anchor[i].expand(count,-1))
        assert torch.equal(new[i,count:],old[i,count:])


def test_native_tasks_and_skill_change_without_new_ball_are_unchanged():
    module=NewBallRmaHistory();x=observation()
    assert torch.equal(module(x),x)
    x[:,940]=1;assert torch.equal(module(x),x)
    x[:,-2:]=torch.tensor([0.,1.])  # Same kicked ball is not a new track.
    assert torch.equal(module(x),x)
    module.reset();assert torch.equal(module(x),x)  # Native shooting, including contact.
    x[:,940]=0;assert torch.equal(module(x),x)


def test_new_ball_lifecycle_mixed_batch_and_inference_reset():
    module=NewBallRmaHistory();x=observation()
    with torch.inference_mode():
        module(x);x[0,940]=1;module(x)
        x[:,940]=0;x[:,-2:]=torch.tensor([0.,1.]);y=module(x)
        assert not torch.equal(y[0,START:END],x[0,START:END])
        assert torch.equal(y[1],x[1])
        anchor=torch.tensor([2.,-.4,1.])
        assert torch.equal(y[0,START:END].reshape(50,3),anchor.expand(50,-1))
    # Real later measurements must be retained; old padding expires after 50 calls.
    for age in range(1,51):
        x[:,933:935]+=1
        y=module(x)
        if age<50:
            assert torch.equal(y[0,START:END].reshape(50,3)[:50-age],anchor.expand(50-age,-1))
        else:assert torch.equal(y,x)
    module.reset(torch.tensor([True,False]));assert torch.equal(module(x),x)
    assert not module.state_dict()
    with pytest.raises(ValueError):module(observation(1))
    module.reset();module(observation(1))
