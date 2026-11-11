"""Training-only replay of physical expert targets on oracle-visited states."""
import torch


def load_replay(path, device, reserved_envs=8):
    rows = torch.load(path, map_location='cpu', weights_only=True)
    mask = torch.stack([r['active'] for r in rows]).bool()
    if mask.shape[1] <= reserved_envs:
        raise ValueError('Insufficient environments for replay split')
    mask[:, -reserved_envs:] = False
    if not mask.any():
        raise ValueError('No active replay samples')
    values = [torch.stack([r['projector_obs'] for r in rows]),
              torch.stack([r['expert_targets'] for r in rows])]
    values += [torch.stack([r['bfm_inputs'][k] for r in rows]) for k in range(3)]
    result = tuple(v[mask].to(device) for v in values)
    if not all(torch.isfinite(v).all() for v in result):
        raise ValueError('Nonfinite replay data')
    return result


def action_loss(projector, actor, data, to_targets, batch_size=64):
    x, y, *states = data
    ids = torch.randint(len(x), (batch_size,), device=x.device)
    prediction = to_targets(actor(*(v[ids] for v in states), projector(x[ids])))
    return (prediction-y[ids]).square().mean()
