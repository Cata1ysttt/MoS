import torch
from fiber_soccer.diagnostic_replay import load_replay, action_loss


def test_reserved_and_inactive_states_excluded(tmp_path):
    rows=[]
    for _ in range(2):
        x=torch.arange(10.).reshape(10,1)
        rows.append(dict(active=torch.tensor([True,False]+[True]*8),projector_obs=x,
                         expert_targets=x,bfm_inputs=(x,x,x)))
    path=tmp_path/'data.pt';torch.save(rows,path)
    result=load_replay(path,'cpu')
    assert result[0].shape==(2,1)
    assert result[0].eq(0).all()


def test_action_loss_backpropagates_only_projector():
    p=torch.nn.Linear(1,1,bias=False)
    actor=torch.nn.Linear(1,1,bias=False).requires_grad_(False)
    def forward(a,b,c,z):return actor(z)+a
    data=tuple(torch.ones(8,1) for _ in range(5))
    action_loss(p,forward,data,lambda a:a).backward()
    assert p.weight.grad is not None
    assert actor.weight.grad is None
