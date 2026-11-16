"""Measure adapter objective competition on held-out recorded states, without updates."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from fiber_soccer.native_bfm import NativeActor, load_native
from fiber_soccer.paths import ASSETS
from fiber_soccer.planner import (
    SCHEMA_V4, load_planner, observed_contact_window, trajectory_event_ids,
    trajectory_windows, with_task_history,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        parser.error('Batch count and size must be positive')
    assert not args.output.exists(), 'Preserve previous audit output'
    assert (args.checkpoint.parent/'complete.json').exists()
    torch.set_num_threads(4)
    torch.manual_seed(812155)
    checksum = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    header = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = load_planner(header, args.device)
    assert model.schema == SCHEMA_V4 and model.adapter is not None
    parameters = list(model.adapter.parameters())
    actor = NativeActor(load_native(args.device))
    default = torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'], device=args.device)
    scale = torch.tensor(json.loads((ASSETS/'actor_meta.json').read_text())['action_rescale'], device=args.device)*1.5
    joint_weight = torch.ones(29, device=args.device)
    joint_weight[:12] = 2
    pools = {}
    for path in args.data:
        assert (path.parent/'complete.json').exists()
        data = torch.load(path, map_location='cpu', weights_only=True)
        assert data['metadata']['skill'] == ('pass', 'shoot')[model.adapter_skill]
        rows = data['rows']
        cut = len(rows[0]['x'])*3//4
        assert cut < len(rows[0]['x'])
        x = torch.stack([r['x'][cut:] for r in rows])
        x = with_task_history(x, torch.stack([r['expert_observation'][cut:] for r in rows]))
        episode = torch.stack([r['episode'][cut:] for r in rows])
        present = ['ball_episode' in r for r in rows]
        assert all(present) or not any(present)
        ball = torch.stack([r['ball_episode'][cut:] for r in rows]) if all(present) else None
        events = trajectory_event_ids(x, episode, ball)
        labels = torch.stack([r['label_z'][cut:] for r in rows])
        target = torch.stack([r['target'][cut:] for r in rows])
        values = trajectory_windows(x, labels, events, target, model.horizon)
        contact = observed_contact_window(x, events, before=10, after=0)
        contact = contact[:len(values[0])] & (x[:len(values[0]), :, 940] < .5)
        for name, mask in [('uniform', torch.ones_like(contact)), ('precontact', contact)]:
            selected = tuple(v[mask] for v in values)
            pools[name] = tuple(torch.cat((old, new)) for old, new in zip(pools[name], selected)) if name in pools else selected
    results = {}
    for name, values in pools.items():
        assert len(values[0]), f'No states in {name}'
        measurements = []
        for _ in range(args.batches):
            ids = torch.randint(len(values[0]), (args.batch_size,))
            x, target, mask, expert = [v[ids].to(args.device) for v in values]
            noise = torch.randn_like(target)
            time = torch.rand(len(x), device=args.device)
            mixed = (1-time[:, None, None])*noise+time[:, None, None]*target
            prediction = model.endpoint(mixed, time, x)
            flow = ((prediction-target).square().mean(-1)*mask).sum()/mask.sum()
            sampled = model.act(x)
            action = actor(x[:, :64], x[:, 64:93], x[:, 93:930], sampled)
            error = (default+action.clamp(-1, 1)*scale-expert).square()
            action_loss = (error*joint_weight).sum(-1).div(joint_weight.sum()).mean()
            gradients = []
            for loss in (flow, action_loss*100):
                gradients.append(torch.cat([g.flatten() for g in torch.autograd.grad(loss, parameters)]).double())
            gf, ga = gradients
            fn, an = gf.norm(), ga.norm()
            dot = gf.dot(ga)
            measurements.append(dict(flow_loss=float(flow.detach()), action_mse=float(action_loss.detach()),
                flow_gradient_norm=float(fn), weighted_action_gradient_norm=float(an),
                cosine=float(dot/(fn*an)) if fn > 0 and an > 0 else None,
                weighted_action_descent_dot_combined_gradient=float(ga.dot(ga+gf))))
        results[name] = dict(available_states=len(values[0]), batches=measurements,
            negative_cosine_batches=sum(v['cosine'] is not None and v['cosine'] < 0 for v in measurements),
            combined_gradient_opposes_action_descent_batches=sum(v['weighted_action_descent_dot_combined_gradient'] < 0 for v in measurements))
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checksum
    assert all(p.grad is None for p in actor.parameters())
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=checksum,
        data=list(map(str, args.data)), seed=812155, batch_size=args.batch_size,
        action_weight=100, leg_weight=2, results=results,
        scope='Last quarter of recorded environments; offline adapter gradients only; no parameter updates. Negative cosine does not by itself establish a closed-loop failure cause.')
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
