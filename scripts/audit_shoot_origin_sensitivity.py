"""Measure explicit root-anchor sensitivity on historical shooting states.

This changes only planner fields 941:944. It is an input ablation, not a
coordinate-consistent simulator transformation or an executed controller.
"""
import argparse
import json
from pathlib import Path
import torch
from audit_new_ball_history_sensitivity import sha, stats
from fiber_soccer.planner_policy import load_soccer_policy
from fiber_soccer.new_ball_rma import initialize_new_ball_history
from fiber_soccer.new_ball_task_history import initialize_new_ball_task_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    source = args.collection / 'shoot_states.pt'
    data = torch.load(source, map_location='cpu', weights_only=True)
    summary = json.loads((args.collection / 'summary.json').read_text())
    starts = {i: v['tick'] for v in summary['switches'] for i in v['environments']}
    x = data['x']; env = data['environment'].long(); tick = data['tick'].long()
    assert x.shape[1] == 2406 and (x[:, -2:].argmax(-1) == 1).all()
    assert set(env.tolist()) == set(starts)
    ages = torch.empty(len(x), dtype=torch.long)
    ball = torch.empty(len(x), 3); task = torch.empty(len(x), 10)
    origins = torch.empty(len(x), 3)
    groups = {'handoff': [], 'first_close_after_history_flush': []}
    for i, start in starts.items():
        ids = (env == i).nonzero().flatten()
        assert int(tick[ids[0]]) == start
        first = x[ids[0]]
        ages[ids] = tick[ids] - start
        ball[ids, :2] = first[933:935]; ball[ids, 2] = 1
        task[ids] = torch.cat((first[933:939], first[2400:2404]))
        origins[ids] = first[941:944]
        groups['handoff'].append(int(ids[0]))
        close = ids[(ages[ids] >= 50) & (x[ids, 933:935].norm(dim=-1) <= .7) & (x[ids, 940] < .5)]
        if len(close): groups['first_close_after_history_flush'].append(int(close[0]))
    digest = sha(args.checkpoint)
    policy = load_soccer_policy(args.checkpoint, 'cuda:0')
    assert policy.refinement_scope == 'initial_pass'
    assert policy.new_ball_history_mode == 'initialize_shoot_ball_task'
    results = {}
    with torch.inference_mode():
        for name, indices in groups.items():
            values = {mode: {'latent_l2': [], 'joint_rms_rad': [], 'maximum_joint_change_rad': []}
                      for mode in ('rebase_xy', 'rebase_heading', 'rebase_xy_heading')}
            for start in range(0, len(indices), 64):
                ids = torch.tensor(indices[start:start+64])
                raw = x[ids].cuda(); age = ages[ids].cuda(); origin = origins[ids].cuda()
                full = initialize_new_ball_history(raw, age, ball[ids].cuda())
                full = initialize_new_ball_task_history(full, age, task[ids].cuda())
                z = policy.planner.prior.act(full)
                def targets(latent):
                    action = policy.actor(full[:, :64], full[:, 64:93], full[:, 93:930], latent)
                    return policy.default + (6 * action).clamp(-6, 6) * policy.scale
                reference = targets(z)
                for mode, metrics in values.items():
                    modified = full.clone()
                    if mode in ('rebase_xy', 'rebase_xy_heading'):
                        modified[:, 941:943] -= origin[:, :2]
                    if mode in ('rebase_heading', 'rebase_xy_heading'):
                        angle = modified[:, 943] - origin[:, 2]
                        modified[:, 943] = torch.atan2(angle.sin(), angle.cos())
                    assert torch.equal(modified[:, :941], full[:, :941])
                    assert torch.equal(modified[:, 944:], full[:, 944:])
                    nz = policy.planner.prior.act(modified)
                    delta = targets(nz) - reference
                    measured = dict(latent_l2=(nz-z).norm(dim=-1),
                                    joint_rms_rad=delta.square().mean(-1).sqrt(),
                                    maximum_joint_change_rad=delta.abs().amax(-1))
                    for key, value in measured.items():
                        assert torch.isfinite(value).all()
                        metrics[key].append(value.cpu())
            results[name] = dict(states=len(indices), metrics={mode: {key: stats(torch.cat(v))
                for key, v in metrics.items()} for mode, metrics in values.items()})
    assert sha(args.checkpoint) == digest
    first_ids = torch.tensor(groups['handoff'])
    output = dict(checkpoint_sha256=digest, collection_sha256=sha(source), groups=results,
        handoff_root_xy_norm_m=stats(origins[first_ids, :2].norm(dim=-1)),
        handoff_absolute_heading_rad=stats(origins[first_ids, 2].abs()),
        actions_executed=False, expert_loaded=False, weights_modified=False,
        other_features_exact=True,
        scope='Input-only ablation of explicit root XY and heading anchors on historical r3 states with corrected ball histories. Body features, backward latent and task histories remain fixed. This is not a complete coordinate transformation, a valid invariance test, or evidence of closed-loop improvement.')
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
