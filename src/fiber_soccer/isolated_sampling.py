"""Independent Torch randomness for a specified environment and task event."""
from contextlib import contextmanager
import hashlib


def event_seed(seed, environment, event=0):
    if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in (seed,environment,event)):
        raise ValueError('Seed, environment and event must be nonnegative integers')
    value=f'soccer-handoff-v1:{seed}:{environment}:{event}'.encode()
    return int.from_bytes(hashlib.blake2b(value,digest_size=8).digest(),'little') & ((1<<63)-1)


@contextmanager
def isolated_torch_rng(seed, device):
    """Restore CPU and the selected CUDA generator, including on exceptions.

This does not isolate the simulator's own RNG or arbitrary Python/NumPy RNGs.
Do not use torch.manual_seed here: it also seeds other CUDA devices.
"""
    import torch
    device=torch.device(device)
    if device.type not in ('cpu','cuda'):raise ValueError('Only CPU and CUDA are supported')
    devices=[] if device.type=='cpu' else [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(devices[0]):torch.cuda.manual_seed(seed)
        yield
