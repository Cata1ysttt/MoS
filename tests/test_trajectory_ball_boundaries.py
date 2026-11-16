import pytest
import torch
from fiber_soccer.planner import trajectory_event_ids,trajectory_windows,SCHEMA_DIMS,SCHEMA_V4


def test_future_labels_stop_at_ball_respawn_without_robot_reset():
    x=torch.zeros(6,2,SCHEMA_DIMS[SCHEMA_V4]);episode=torch.zeros(6,2,dtype=torch.long)
    x[:,:,-3]=2.;x[:,:,939]=1
    x[1:3,0,940]=1  # Ball kicked, then a new ball appears at tick 3.
    x[3:,0,939]=0
    before=x.clone()
    events=trajectory_event_ids(x,episode)
    z=torch.randn(6,2,256);targets=torch.randn(6,2,29)
    old=trajectory_windows(x,z,episode,targets,3)[2]
    new=trajectory_windows(x,z,events,targets,3)[2]
    assert old[2,0].tolist()==[True,True,True]
    assert new[2,0].tolist()==[True,False,False]
    assert new[:,1].all() and new[:,:,0].all()
    torch.testing.assert_close(x,before,rtol=0,atol=0)


def test_recorded_event_ids_handle_unobservable_respawn_and_speed_retargeting():
    x=torch.zeros(6,2,1206);episode=torch.zeros(6,2,dtype=torch.long)
    ball=torch.zeros_like(episode);ball[2:,0]=1
    episode[4:,1]=1
    x[:,:,-3]=2.;x[3:,1,-3]=3.
    events=trajectory_event_ids(x,episode,ball)
    assert events[:,0].tolist()==[0,0,1,1,1,1]
    assert events[:,1].tolist()==[0,0,0,1,2,2]
    with pytest.raises(ValueError):trajectory_event_ids(x,episode,ball[:-1])
