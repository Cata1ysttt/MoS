"""Passive shooting-state collection after a real passing handoff.

No expert or latent optimizer is loaded here. Labels can be computed offline
from the saved, exact shooting observation history and action affine mapping.
"""
import torch


class SequenceShootingRecorder:
    def __init__(self, num_envs, joint_order):
        self.num_envs = num_envs
        self.joint_order = list(joint_order)
        if sorted(self.joint_order) != list(range(29)):
            raise ValueError('Expected a complete 29-joint action permutation')
        self.batches = []

    def record(self, tick, observation, latent, alive, action_scale, action_offset):
        if observation.shape != (self.num_envs, 2406) or latent.shape != (self.num_envs, 256):
            raise ValueError('Expected v4 observations and one executed latent per environment')
        if alive.shape != (self.num_envs,) or alive.dtype != torch.bool:
            raise ValueError('Expected one alive flag per environment')
        selected = alive & (observation[:, -2:].argmax(-1) == 1)
        ids = selected.nonzero().flatten()
        if not len(ids):
            return
        if self.batches and tick <= self.batches[-1]['tick'][0].item():
            raise ValueError('Recording ticks must strictly increase')
        def affine(value):
            return torch.as_tensor(value, device=observation.device, dtype=observation.dtype).expand(
                self.num_envs, 29)[ids][:, self.joint_order].detach().cpu().clone()
        self.batches.append(dict(
            x=observation[ids].detach().cpu().clone(),
            z=latent[ids].detach().cpu().clone(),
            environment=ids.detach().cpu().clone(),
            tick=torch.full((len(ids),), tick, dtype=torch.long),
            action_scale=affine(action_scale), action_offset=affine(action_offset)))

    def export(self, metadata):
        if not self.batches:
            raise ValueError('No alive shooting states were observed')
        fields = {key: torch.cat([batch[key] for batch in self.batches]) for key in self.batches[0]}
        if not all(torch.isfinite(fields[key]).all() for key in ['x','z','action_scale','action_offset']):
            raise ValueError('Nonfinite collected states')
        return dict(metadata=dict(metadata, schema='soccer_state_v4_task_history_2406',
                    format='flat_sequence_shoot_states_v1', num_envs=self.num_envs,
                    joint_order=self.joint_order, expert_loaded=False, expert_actions_executed=False,
                    sampling='Every control step while alive and commanded to shoot after passing handoff'), **fields)
