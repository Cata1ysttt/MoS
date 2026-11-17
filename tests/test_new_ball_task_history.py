import torch
from fiber_soccer.new_ball_task_history import NewBallTaskHistory,initialize_new_ball_task_history,TERMS


def obs(n=2):
    x=torch.arange(n*2406,dtype=torch.float32).reshape(n,2406)
    x[:,940]=0;x[:,-2:]=torch.tensor([1.,0.])
    return x


def test_only_old_ball_task_terms_replaced():
    x=obs(4);original=x.clone();age=torch.tensor([0,1,9,10])
    anchors=torch.arange(40,dtype=torch.float32).reshape(4,10)
    y=initialize_new_ball_task_history(x,age,anchors)
    assert torch.equal(x,original)
    assert torch.equal(y[:,:2130],x[:,:2130]) and torch.equal(y[:,2230:],x[:,2230:])
    for start,end,a,b in TERMS:
        old=x[:,start:end].reshape(4,10,b-a);new=y[:,start:end].reshape(4,10,b-a)
        for i,count in enumerate([10,9,1,0]):
            assert torch.equal(new[i,:count],anchors[i,a:b].expand(count,-1))
            assert torch.equal(new[i,count:],old[i,count:])


def test_combined_track_native_mixed_batch_expiration_and_reset():
    module=NewBallTaskHistory();x=obs()
    with torch.inference_mode():
        assert torch.equal(module(x),x)
        x[0,940]=1;assert torch.equal(module(x),x)
        x[:,-2:]=torch.tensor([0.,1.]);assert torch.equal(module(x),x)  # Same kicked ball.
        x[:,-2:]=torch.tensor([1.,0.]);module(x)
        x[:,-2:]=torch.tensor([0.,1.]);x[:,940]=0
        before=x.clone();y=module(x)
        assert torch.equal(x,before) and torch.equal(y[1],x[1])
        assert torch.equal(y[:,:2130],x[:,:2130]) and torch.equal(y[:,2380:],x[:,2380:])
        anchor=torch.cat((x[0,933:939],x[0,2400:2404])).clone()
        for start,end,a,b in TERMS:assert torch.equal(y[0,start:end].reshape(10,b-a),anchor[a:b].expand(10,-1))
    for age in range(1,51):
        x[:,933:939]+=1;x[:,2403]+=1;y=module(x)
        for start,end,a,b in TERMS:
            new=y[0,start:end].reshape(10,b-a);old=x[0,start:end].reshape(10,b-a)
            count=max(0,10-age)
            assert torch.equal(new[:count],anchor[a:b].expand(count,-1))
            assert torch.equal(new[count:],old[count:])
        if age==10:assert not torch.equal(y[0,2230:2380],x[0,2230:2380])
        if age==50:assert torch.equal(y,x)
    module.reset(torch.tensor([True,False]));assert torch.equal(module(x),x)
    module.reset();assert module.initial_task_measurement is None
    assert torch.equal(module(x),x)  # Native first shooting call.
    assert not module.state_dict()
