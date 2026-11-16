"""Causal task-feature history shared by offline reconstruction and online state."""
import torch


class CausalHistory:
    def __init__(self, first_frame, length=50):
        if length < 1 or first_frame.ndim != 2:
            raise ValueError('History requires a batch of features and positive length')
        self.value = first_frame[:, None].repeat(1, length, 1).clone()

    @property
    def flat(self):
        return self.value.flatten(1)

    def push(self, frame, reset):
        if frame.shape != self.value[:, 0].shape or reset.shape != frame.shape[:1]:
            raise ValueError('History batch or reset shape mismatch')
        self.value = torch.cat((frame[:, None], self.value[:, :-1]), 1)
        self.value[reset] = frame[reset, None]
        return self.flat


def causal_history(features, episode, length=50):
    """Newest frame first; repeat the first available frame at each reset.

    No future state is used, and histories never cross an episode boundary.
    """
    if features.ndim != 3 or features.shape[:2] != episode.shape or not len(features):
        raise ValueError('Expected aligned time/environment features and episode IDs')
    history = CausalHistory(features[0], length)
    rows = [history.flat.clone()]
    for tick in range(1, len(features)):
        rows.append(history.push(features[tick], episode[tick] != episode[tick-1]).clone())
    return torch.stack(rows)
