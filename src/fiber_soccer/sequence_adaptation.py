"""Data boundaries for shooting adaptation on autonomous post-pass states."""
import torch
from .planner import SCHEMA_V4
from .sequence_collection import shooting_action_targets


def split_sequence_labels(data, base_sha256, expert_sha256):
    m=data['metadata'];x=data['x'];environment=data['environment']
    if m['purpose']!='training':
        raise ValueError('Diagnostic/evaluation states must not enter adaptation')
    if m['schema']!=SCHEMA_V4 or m['format']!='flat_sequence_shoot_states_v1':
        raise ValueError('Unexpected sequence data schema')
    if m['seed'] in (811314,812801,812904,812905,813201,815101,815102):
        raise ValueError('Reserved evaluation seed cannot enter adaptation')
    if m['checkpoint_sha256']!=base_sha256 or m['expert_sha256']!=expert_sha256:
        raise ValueError('Collection policy or labeling expert identity mismatch')
    if m['expert_loaded'] or m['expert_actions_executed'] or not m['offline_expert_loaded']:
        raise ValueError('Require autonomous collection with offline expert labels')
    if m['physical_target_mapping']!='raw_action_clamp_10_then_affine_no_physical_clip' or m['raw_action_clip']!=10:
        raise ValueError('Labels must include the source environment raw-action clamp')
    n=len(x)
    if x.shape!=(n,2406) or data['target'].shape!=(n,29) or environment.shape!=(n,):
        raise ValueError('Invalid sequence tensors')
    if not ((environment>=0)&(environment<m['num_envs'])).all() or not (x[:,-2:].argmax(-1)==1).all():
        raise ValueError('Invalid environment or nonshooting state')
    expected=shooting_action_targets(data['expert_raw_action'],data['action_scale'],data['action_offset'],m['joint_order'])
    if not torch.equal(expected,data['target']):
        raise ValueError('Saved targets do not match the source action contract')
    if not torch.isfinite(x).all() or not torch.isfinite(data['target']).all():
        raise ValueError('Nonfinite input or target')
    split=m['num_envs']*3//4
    train=(environment<split).nonzero().flatten()
    valid=(environment>=split).nonzero().flatten()
    if not len(train) or not len(valid):
        raise ValueError('Both disjoint environment splits must contain states')
    return train,valid


class EnvironmentSampler:
    """Uniform environment, then uniform state: long surviving lanes do not dominate."""
    def __init__(self, environment, indices):
        if len(indices)==0:
            raise ValueError('Cannot sample an empty split')
        order=environment[indices].argsort(stable=True)
        self.indices=indices[order]
        _,self.counts=environment[self.indices].unique_consecutive(return_counts=True)
        self.starts=torch.cat((torch.zeros(1,dtype=torch.long),self.counts.cumsum(0)[:-1]))

    def sample(self, size, generator):
        group=torch.randint(len(self.counts),(size,),generator=generator)
        within=(torch.rand(size,generator=generator)*self.counts[group]).long()
        return self.indices[self.starts[group]+within]
