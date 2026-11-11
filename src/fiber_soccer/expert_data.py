"""Build aligned windows and preserve resets and independent environment splits."""
import torch
from .lpp import execute_latent


def expert_windows(rows, interval=5):
    if interval not in (1,5):
        raise ValueError('Supported latent intervals are 1 and 5')
    if len(rows) < interval or any(row['tick'] != i for i, row in enumerate(rows)):
        raise ValueError('Expected a contiguous dense trajectory')
    obs = torch.stack([r['projector_obs'] for r in rows])
    done = torch.stack([r['done'] for r in rows])
    count = torch.stack([r['kick_count'] for r in rows])
    event = torch.zeros_like(done)
    event[1:] = (count[1:] > count[:-1]) & ~done[:-1]
    # Retrospective contact neighborhoods are sampling labels, never policy inputs.
    contact = event.clone()
    for shift in range(1, min(21, len(rows))):
        same_episode = torch.stack([~done[k:len(done)-shift+k] for k in range(shift)]).all(0)
        contact[:-shift] |= event[shift:] & same_episode
        contact[shift:] |= event[:-shift] & same_episode
    pieces = []
    for start in range(0, len(rows)-interval+1, interval):
        chunk = rows[start:start+interval]
        mask = torch.ones_like(done[start:start+interval])
        for h in range(1, interval):
            mask[h] = mask[h-1] & ~done[start+h-1]
        previous_valid = (~done[start-interval:start]).all(0) if start >= interval else torch.zeros_like(done[0])
        pieces.append(dict(
            obs=obs[start], previous_obs=obs[max(0, start-interval)], previous_valid=previous_valid,
            states=tuple(torch.stack([r['bfm_inputs'][k] for r in chunk], 1) for k in range(3)),
            targets=torch.stack([r['targets'] for r in chunk], 1), mask=mask.T,
            contact=(contact[start:start+interval] & mask).any(0),
            env_ids=torch.arange(len(done[0])), ticks=torch.full((len(done[0]),), start)))
    result={k:torch.cat([p[k] for p in pieces]) for k in pieces[0] if k!='states'}
    result['states']=tuple(torch.cat([p['states'][k] for p in pieces]) for k in range(3))
    return result


def replay_error(projector, actor, data, indices, to_targets, stand, execution='slerp'):
    """Teacher-state action alignment; previous policy output is a fixed endpoint."""
    with torch.no_grad():
        previous=projector(data['previous_obs'][indices])
        previous=torch.where(data['previous_valid'][indices,None],previous,stand)
    endpoint=projector(data['obs'][indices])
    predictions=[]
    interval=data['targets'].shape[1]
    for h in range(interval):
        command=execute_latent(previous,endpoint,(h+1)/interval,execution)
        predictions.append(to_targets(actor(*(v[indices,h] for v in data['states']),command)))
    error=(torch.stack(predictions,1)-data['targets'][indices]).square().mean(-1)
    mask=data['mask'][indices]
    return (error*mask).sum(1)/mask.sum(1).clamp_min(1)
