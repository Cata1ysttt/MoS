import torch
from fiber_soccer.planner_history import CausalHistory, causal_history


def test_causal_history_matches_online_updates_and_resets_per_lane():
    features = torch.arange(20.).reshape(5, 2, 2)
    episode = torch.tensor([[0,0],[0,0],[0,0],[1,0],[1,0]])
    built = causal_history(features, episode, length=3)
    live = CausalHistory(features[0], length=3)
    torch.testing.assert_close(built[0], live.flat)
    for t in range(1,5):
        torch.testing.assert_close(built[t], live.push(features[t], episode[t] != episode[t-1]))
    torch.testing.assert_close(built[3,0], features[3,0].repeat(3))
    torch.testing.assert_close(built[3,1], features[[3,2,1],1].flatten())
    changed = features.clone()
    changed[3:] = 999
    torch.testing.assert_close(causal_history(changed,episode,3)[:3], built[:3])
