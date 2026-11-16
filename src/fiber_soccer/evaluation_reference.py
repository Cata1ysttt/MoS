"""Load completed paired controls with explicit reuse provenance."""
import hashlib
import json
from pathlib import Path


def load_reference_evaluations(runs, tag, seed, num_envs, baseline_sha256, expert_checkpoints):
    if not tag.replace('_', '').isalnum():
        raise ValueError('Invalid reference tag')
    runs = Path(runs)
    status = json.loads((runs/(tag+'_status.json')).read_text())
    protocol = json.loads((runs/(tag+'_protocol.json')).read_text())
    path = runs/(tag+'_results.json')
    encoded = path.read_bytes()
    results = json.loads(encoded)
    if status.get('status') != 'completed' or status.get('results') != results or status.get('protocol') != protocol:
        raise ValueError('Reference must have matching completed status, protocol and results')
    if protocol['seed'] != seed or protocol['num_envs'] != num_envs:
        raise ValueError('Reference seed or environment count differs')
    if protocol['checkpoint_sha256']==baseline_sha256:
        reference_role='planner'
    elif protocol.get('baseline_checkpoint_sha256')==baseline_sha256:
        reference_role='baseline'
    else:
        raise ValueError('Requested baseline differs from reference planners')
    selected = {}
    for skill in ['pass', 'shoot']:
        selected[skill] = {'baseline': results[skill][reference_role], 'expert': results[skill]['expert']}
        for controller, value in selected[skill].items():
            expected = 'expert' if controller == 'expert' else 'unified_latent_planner'
            if value['seed'] != seed or value['requested'] != num_envs or value['completed'] != num_envs:
                raise ValueError('Reference rollout incomplete or unpaired')
            if value['controller'] != expected or value['expert_assisted_oracle']:
                raise ValueError('Reference controller or assistance differs')
        if selected[skill]['expert']['checkpoint'] != expert_checkpoints[skill]:
            raise ValueError('Reference expert differs from configured expert')
        if selected[skill]['baseline']['initial_state_sha256'] != selected[skill]['expert']['initial_state_sha256']:
            raise ValueError('Reference initial physical states differ')
    provenance = dict(reference_evaluations_tag=tag, reference_planner_role=reference_role, reference_results_sha256=hashlib.sha256(encoded).hexdigest(),
        validation_seed_previously_evaluated=True,
        reference_runs={skill:{'baseline':protocol.get('reference_runs',{}).get(skill,{}).get('baseline',str(runs/(tag+'_'+skill+'_baseline'))) if reference_role=='baseline' else str(runs/(tag+'_'+skill+'_planner')),
                              'expert':protocol.get('reference_runs',{}).get(skill,{}).get('expert',str(runs/(tag+'_'+skill+'_expert')))} for skill in selected})
    return selected, provenance
