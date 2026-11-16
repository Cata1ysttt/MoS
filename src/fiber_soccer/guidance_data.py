"""Historical trajectory splits shared by target prediction and latent mapping."""
import hashlib
import torch
from .planner import with_task_history,trajectory_event_ids,observed_contact_window,SCHEMA_V3,SCHEMA_V4


def load_guidance_data(paths,smoke=False):
    parts={skill:[] for skill in ['pass','shoot']}
    validation={skill:[] for skill in parts};provenance=[]
    for path in paths:
        assert (path.parent/'complete.json').exists()
        data=torch.load(path,map_location='cpu',weights_only=True);m=data['metadata'];rows=data['rows'];skill=m['skill']
        assert skill in parts and m['seed'] not in {811314,815701,815702,815801,815802,815803,815804}
        x=torch.stack([r['x'] for r in rows]);target=torch.stack([r['target'] for r in rows])
        if m['schema']==SCHEMA_V3:x=with_task_history(x,torch.stack([r['expert_observation'] for r in rows]))
        else:assert m['schema']==SCHEMA_V4
        assert x.shape[-1]==2406 and target.shape==(*x.shape[:2],29)
        assert torch.isfinite(x).all() and torch.isfinite(target).all()
        assert (x[:,:,-2:].argmax(-1)==int(skill=='shoot')).all()
        episode=torch.stack([r['episode'] for r in rows])
        present=['ball_episode' in r for r in rows];assert all(present) or not any(present)
        ball=torch.stack([r['ball_episode'] for r in rows]) if all(present) else None
        contact=observed_contact_window(x,trajectory_event_ids(x,episode,ball),before=10,after=0)&(x[:,:,940]<.5)
        n=x.shape[1];cut=n*3//4;assert 0<cut<n
        tx=x[:,:cut].flatten(0,1);ty=target[:,:cut].flatten(0,1);tc=contact[:,:cut].flatten()
        if smoke:
            ids=torch.cat((torch.arange(min(256,len(tx))),tc.nonzero().flatten()[:128])).unique()
            tx,ty,tc=tx[ids],ty[ids],tc[ids]
        parts[skill].append((tx,ty,tc))
        mask=contact[:,cut:];teacher=torch.stack([r['teacher_executed'] for r in rows])[:,cut:]
        validation[skill].append((x[:,cut:][mask],target[:,cut:][mask],teacher[mask]))
        provenance.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               metadata=m,training_lanes=[0,cut],validation_lanes=[cut,n]))
    training={skill:tuple(torch.cat([entry[i] for entry in values]) for i in range(3)) for skill,values in parts.items()}
    validation={skill:tuple(torch.cat([entry[i] for entry in values]) for i in range(3)) for skill,values in validation.items()}
    assert all(len(v[0])>0 and v[2].any() for v in training.values())
    assert all(len(v[0])>0 for v in validation.values())
    return training,validation,provenance
