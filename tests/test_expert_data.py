import torch
from fiber_soccer.expert_data import expert_windows


def test_windows_exclude_reset_transitions_and_reset_previous_latent():
    rows=[]
    for tick in range(10):
        rows.append(dict(tick=tick,projector_obs=torch.full((2,3),tick),
                         done=torch.tensor([tick==2,False]),kick_count=torch.zeros(2),
                         bfm_inputs=(torch.zeros(2,4),)*3,targets=torch.zeros(2,29)))
    rows[2]['kick_count'][0]=1
    data=expert_windows(rows)
    assert data['mask'][0].tolist()==[True,True,True,False,False]
    assert data['previous_valid'].tolist()==[False,False,False,True]
    assert data['contact'][0] and not data['contact'][2]
    assert data['env_ids'].tolist()==[0,1,0,1]
    assert data['previous_obs'][3,0]==0 and data['obs'][3,0]==5


def test_single_step_replay_uses_current_observation_and_resets_history():
    from fiber_soccer.expert_data import replay_error
    from fiber_soccer.lpp import project
    torch.manual_seed(9)
    observations=torch.randn(4,2,256)
    rows=[dict(tick=t,projector_obs=observations[t],done=torch.tensor([t==1,False]),
               kick_count=torch.zeros(2),bfm_inputs=(torch.zeros(2,4),)*3,
               targets=project(observations[t])[:,:29]) for t in range(4)]
    data=expert_windows(rows,interval=1)
    assert data['mask'].shape==(8,1)
    assert data['mask'].all()
    assert not data['previous_valid'][4] and data['previous_valid'][5]
    def actor(a,b,c,z):return z[:,:29]
    error=replay_error(project,actor,data,torch.arange(8),lambda x:x,project(torch.ones(1,256)))
    torch.testing.assert_close(error,torch.zeros(8),atol=1e-10,rtol=0)
