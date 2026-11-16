"""Local shooting-expert sensitivity to additional configured proprioceptive noise.

Inputs already contain observation noise. This is a sensitivity experiment, not
an estimate of the irreducible error of a clean-state policy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

p = argparse.ArgumentParser()
p.add_argument('--evaluations', type=Path, nargs='+', required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
session = ort.InferenceSession(str(a.model), sess_options=options, providers=['CPUExecutionProvider'])
meta = session.get_modelmeta().custom_metadata_map
names = meta['observation_names'].split(',')
assert names == ['base_lin_vel','base_ang_vel','joint_pos','joint_vel','actions','ball_pos','ball_vel','ball_target_dir','ball_target_speed']
assert all(float(x) == 10 for x in meta['observation_history_lengths'].split(','))
scale = np.asarray([float(x) for x in meta['action_scale'].split(',')],np.float32)
# Verified against the retained KickingObservationsCfg used by the shooting export.
noise_terms = {'base_lin_vel':(slice(0,30),.5), 'base_ang_vel':(slice(30,60),.2),
               'joint_pos':(slice(60,350),.01), 'joint_vel':(slice(350,640),.5)}
result = dict(note=__doc__,physical_scale='Rounded export action_scale; before additional physical joint clipping',evaluations={})
for path in a.evaluations:
    records = torch.load(path/'action_diagnostics.pt',map_location='cpu',weights_only=True)
    values = torch.cat([r['projector_obs'][r['active']] for r in records if 10 <= r['tick'] <= 100])
    assert values.shape[-1] == 1180
    values = values[torch.linspace(0,len(values)-1,min(128,len(values))).long()].numpy()
    groups = {}
    for case,terms in [('joint_noise',['joint_pos','joint_vel']),('all_proprioception',list(noise_terms))]:
        rng = np.random.default_rng(812400)
        variances = []
        for observation in values:
            samples = []
            for draw in range(16):
                obs = observation[:1030].copy()
                for term in terms:
                    segment, bound = noise_terms[term]
                    obs[segment] += rng.uniform(-bound,bound,obs[segment].shape).astype(np.float32)
                action = session.run(['actions'],{'obs':obs[None], 'rma_obs':observation[1030:][None],
                                                  'time_step':np.zeros((1,1),np.float32)})[0][0]
                samples.append(np.clip(action,-10,10)*scale)
            variances.append(np.asarray(samples).var(0).mean())
        variances = np.asarray(variances)
        groups[case] = dict(states=len(values),draws_per_state=16,mean_action_variance=float(variances.mean()),
                            median_action_rms=float(np.sqrt(np.median(variances))),p90_action_rms=float(np.sqrt(np.quantile(variances,.9))))
    result['evaluations'][path.name] = groups
print(json.dumps(result,indent=2))
a.output.write_text(json.dumps(result,indent=2))
