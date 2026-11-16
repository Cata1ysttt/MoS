"""Offline action-regression diagnostic on the same states/labels as a flow planner.

This checkpoint is not a latent policy and cannot establish task success.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from torch import nn
from fiber_soccer.lpp import Block
from fiber_soccer.native_bfm import NativeActor, load_native
from fiber_soccer.paths import ASSETS
from fiber_soccer.planner import load_planner, observed_contact_window, balanced_skill_mean
from fiber_soccer.planner_history import causal_history

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, nargs='+', required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
information = p.add_mutually_exclusive_group()
information.add_argument('--expert-observation', action='store_true', help='Offline information-sufficiency diagnostic with exact saved expert inputs')
information.add_argument('--state-input-control', action='store_true', help='Same 1200-wide added path, using only already-available state features')
p.add_argument('--ball-history', action='store_true', help='Causal 50-frame ball XY/launch-state history ablation')
p.add_argument('--updates', type=int, default=20000)
p.add_argument('--device', default='cuda:0')
a = p.parse_args()
extra_dim = (150 if a.ball_history else 0)+(1200 if a.expert_observation or a.state_input_control else 0)
a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.manual_seed(812350)
reference_ck = torch.load(a.reference, map_location='cpu', weights_only=False)
reference = load_planner(reference_ck, a.device).eval().requires_grad_(False)
assert reference_ck['planner_arch'] == 'flow_endpoint_v1'
actor = NativeActor(load_native(a.device))
scale = torch.tensor(json.loads((ASSETS/'actor_meta.json').read_text())['action_rescale'], device=a.device)*1.5
default = torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'], device=a.device)
joint_weight = torch.ones(29, device=a.device)
joint_weight[:12] = 2

class ActionRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('obs_mean', reference.obs_mean.clone())
        self.register_buffer('obs_std', reference.obs_std.clone())
        self.condition = copy.deepcopy(reference.condition)
        self.command_encoder = copy.deepcopy(reference.command_encoder)
        width = reference.width
        self.head = nn.Sequential(Block(width), Block(width), Block(width), nn.LayerNorm(width), nn.Linear(width, 29))
        self.history_encoder = None
        if extra_dim:
            # Zero-initialize the new input path without changing the base
            # initialization or random stream used for minibatch sampling.
            with torch.random.fork_rng(devices=[]):
                self.history_encoder = nn.Linear(extra_dim, width, bias=False)
                nn.init.zeros_(self.history_encoder.weight)
            self.register_buffer('history_mean', torch.zeros(extra_dim, device=a.device))
            self.register_buffer('history_std', torch.ones(extra_dim, device=a.device))
        self.requires_grad_(True)

    def forward(self, x):
        base = x[:,:reference.obs_dim]
        normalized = ((base-self.obs_mean)/self.obs_std.clamp_min(.05)).clamp(-20,20)
        context = self.condition(normalized)
        if self.command_encoder is not None:
            context = context+self.command_encoder(normalized[:,-6:])
        if self.history_encoder is not None:
            history = ((x[:,reference.obs_dim:]-self.history_mean)/self.history_std.clamp_min(.05)).clamp(-20,20)
            context = context+self.history_encoder(history)
        return self.head(context).tanh()

model = ActionRegressor().to(a.device)
train, valid = {}, {}
contact_flags = {id(train): {}, id(valid): {}}
provenance = []
for path in a.data:
    data = torch.load(path, map_location='cpu', weights_only=True)
    meta, rows = data['metadata'], data['rows']
    assert meta['schema'] == reference.schema and meta['seed'] not in (811314,812801)
    provenance.append(meta)
    x = torch.stack([r['x'] for r in rows])
    z = torch.stack([r.get('label_z',r['z']) for r in rows])
    target = torch.stack([r['target'] for r in rows])
    episode = torch.stack([r['episode'] for r in rows])
    contact = observed_contact_window(x, episode)
    extras = []
    if a.ball_history:
        features = torch.cat((x[:,:,933:935], x[:,:,939:940]),-1)
        extras.append(causal_history(features,episode,50))
    if a.state_input_control:
        assert x.shape[-1] == 1206, 'State-copy control is defined for the 1206-dimensional schema'
        extras.append(x[:,:,:1200].clone())
    if a.expert_observation:
        assert all('expert_observation' in row for row in rows), 'Exact expert observations were not saved'
        expert_inputs = torch.stack([row['expert_observation'] for row in rows])
        assert expert_inputs.shape[-1] <= 1200
        extras.append(torch.nn.functional.pad(expert_inputs,(0,1200-expert_inputs.shape[-1])))
    if extras:
        x = torch.cat((x,*extras),-1)
    length = len(x)-reference.horizon+1
    x, z, target, contact = x[:length], z[:length], target[:length], contact[:length]
    shape = x.shape[:2]
    flat, labels, targets = x.flatten(0,1), z.flatten(0,1), target.flatten(0,1)
    actions, confidence = [], []
    with torch.no_grad():
        for begin in range(0,len(flat),1024):
            xb, zb = flat[begin:begin+1024].to(a.device), labels[begin:begin+1024].to(a.device)
            action = actor(xb[:,:64],xb[:,64:93],xb[:,93:930],zb).clamp(-1,1)
            error = (default+action*scale-targets[begin:begin+1024].to(a.device)).square().mean(-1)
            actions.append(action.cpu())
            confidence.append((1/(1+error/.01)).cpu())
    actions = torch.cat(actions).reshape(*shape,29)
    confidence = torch.cat(confidence).reshape(shape)
    split = max(1,shape[1]*3//4)
    for pool, section in [(train,slice(None,split)),(valid,slice(split,None))]:
        skill = meta['skill']
        values = tuple(v[:,section].flatten(0,1) for v in (x,actions,confidence))
        if skill in pool:
            values = tuple(torch.cat((old,new)) for old,new in zip(pool[skill],values))
        pool[skill] = values
        flags = contact[:,section].flatten()
        previous = contact_flags[id(pool)].get(skill)
        contact_flags[id(pool)][skill] = torch.cat((previous,flags)) if previous is not None else flags
assert set(train) == {'pass','shoot'}
if extra_dim:
    history = torch.cat([v[0][:,reference.obs_dim:] for v in train.values()])
    model.history_mean.copy_(history.mean(0).to(a.device))
    model.history_std.copy_(history.std(0).clamp_min(.05).to(a.device))
starts = {id(pool): {skill:torch.where(v[0][:,93:930].abs().sum(-1)<1e-8)[0] for skill,v in pool.items()} for pool in (train,valid)}
contacts = {key: {skill:torch.where(v)[0] for skill,v in flags.items()} for key,flags in contact_flags.items()}

def batch(pool, size):
    selected = []
    for skill in ['pass','shoot']:
        values = pool[skill]
        reset_ids, contact_ids = starts[id(pool)][skill], contacts[id(pool)][skill]
        nr = size//4 if len(reset_ids) else 0
        nc = size//4 if pool is train and len(contact_ids) else 0
        ids = torch.randint(len(values[0]),(size-nr-nc,))
        if nr:
            ids = torch.cat((ids,reset_ids[torch.randint(len(reset_ids),(nr,))]))
        if nc:
            ids = torch.cat((ids,contact_ids[torch.randint(len(contact_ids),(nc,))]))
        selected.append(tuple(v[ids].to(a.device) for v in values))
    return tuple(torch.cat([v[i] for v in selected]) for i in range(3))

def physical_error(prediction, target):
    return (((prediction-target)*scale).square()*joint_weight).sum(-1)/joint_weight.sum()

def phase_errors(x, errors, weights):
    """Expose pre-contact fit separately from preparation and recovery."""
    result = {}
    for skill, xb, eb, wb in zip(['pass','shoot'], x.chunk(2), errors.chunk(2), weights.chunk(2)):
        launched, kicked = xb[:,939] > .5, xb[:,940] > .5
        masks = {'before_launch': ~launched & ~kicked,
                 'launched_before_contact': launched & ~kicked, 'after_contact': kicked}
        result[skill] = {}
        for name, mask in masks.items():
            result[skill][name] = dict(states=int(mask.sum()),
                action_mse=float((eb[mask]*wb[mask]).sum()/wb[mask].sum()) if mask.any() else None)
    return result

# Identical validation state sampling to the flow trainer, without flow's random
# endpoint interpolation. The compared action sampler itself uses fixed noise.
with torch.random.fork_rng(devices=[torch.device(a.device).index or 0] if a.device.startswith('cuda') else []):
    torch.manual_seed(812151)
    validation = batch(valid,256)
with torch.no_grad():
    vx, target, weight = validation
    predicted_z = reference(vx[:,:reference.obs_dim])[:,0]
    reference_action = actor(vx[:,:64],vx[:,64:93],vx[:,93:930],predicted_z).clamp(-1,1)
    reference_per_state = physical_error(reference_action,target)
    reference_error = balanced_skill_mean(reference_per_state,weight)
    reference_phase_errors = phase_errors(vx,reference_per_state,weight)
config = dict(method='offline_direct_action_regression_diagnostic',latent_policy=False,
              expert_observation_input=a.expert_observation,state_input_control=a.state_input_control,
              extra_input_dim=extra_dim,trainable_parameters=sum(v.numel() for v in model.parameters()),
              extra_ball_history_frames=50 if a.ball_history else 0,ball_history_fields=['relative_xy','launch_started'],
              history_order='newest_first',history_reset='repeat_first_frame',
              reference=str(a.reference),reference_action_mse=float(reference_error),
              reference_phase_errors=reference_phase_errors,
              schema=reference.schema,width=reference.width,data=provenance,
              warm_started_condition_encoder=True,reset_sample_fraction=.25,
              contact_sample_fraction=.25,label_error_scale=.01,leg_weight=2,
              seed=812350,validation_seed=812151,
              note='Same state/decoded-label physical action comparison; not autonomous task success')
(a.output/'config.json').write_text(json.dumps(config,indent=2))
print('REFERENCE',float(reference_error),flush=True)
optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
best, start = float('inf'), time.monotonic()
for update in range(1,a.updates+1):
    model.train()
    x, target, weight = batch(train,64)
    loss = balanced_skill_mean(physical_error(model(x),target),weight)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1)
    optimizer.step()
    if update == 1 or update%500 == 0:
        model.eval()
        with torch.no_grad():
            x, target, weight = validation
            error = physical_error(model(x),target)
            score = balanced_skill_mean(error,weight)
            task_scores = [(e*w).sum()/w.sum() for e,w in zip(error.chunk(2),weight.chunk(2))]
        row = dict(update=update,train_action_mse=float(loss.detach()),validation_action_mse=float(score),
                   pass_validation_action_mse=float(task_scores[0]),shoot_validation_action_mse=float(task_scores[1]),
                   frozen_flow_reference_action_mse=float(reference_error),
                   phase_errors=phase_errors(x,error,weight),elapsed=time.monotonic()-start)
        print(row,flush=True)
        with (a.output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        payload = dict(action_model=model.state_dict(),diagnostic_only=True,config=config,update=update)
        torch.save(payload,a.output/'latest.pt')
        if float(score)<best:
            best=float(score)
            torch.save(payload,a.output/'best.pt')
assert all(p.grad is None for p in actor.parameters())
assert all(p.grad is None for p in reference.parameters())
(a.output/'complete.json').write_text(json.dumps(dict(updates=a.updates,best_validation_action_mse=best,
                                                     flow_reference_action_mse=float(reference_error),
                                                     diagnostic_only=True),indent=2))
