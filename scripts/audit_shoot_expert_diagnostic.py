"""Validate original ONNX input routing without running a simulator or changing policy."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import torch
from fiber_soccer.shoot_expert_diagnostic import ShootingExpertDiagnostic
from fiber_soccer.new_ball_rma import initialize_new_ball_history
from fiber_soccer.new_ball_task_history import initialize_new_ball_task_history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    graph=onnx.load(a.checkpoint).graph
    assert not any(attr.type in (onnx.AttributeProto.GRAPH,onnx.AttributeProto.GRAPHS)
        for node in graph.node for attr in node.attribute),'Nested graphs require a separate dependency audit'
    producers={output:node for node in graph.node for output in node.output}
    todo=['actions'];seen=set();leaves=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        seen.add(name)
        if name in producers:todo.extend(producers[name].input)
        else:leaves.add(name)
    dependencies=sorted(leaves & {value.name for value in graph.input})
    assert 'time_step' not in dependencies,'Fixed-zero time input is not justified for this expert'

    raw=ShootingExpertDiagnostic(a.checkpoint)
    full=ShootingExpertDiagnostic(history_mode='initialize_shoot_ball_task',session=raw.session)
    native=torch.load(a.native/'action_diagnostics.pt',map_location='cpu',weights_only=True)[0]['projector_obs']
    x=torch.zeros(len(native),2406);x[:,1200:2380]=native;x[:,-1]=1
    def manual(obs,frame=0):
        return torch.from_numpy(np.concatenate([raw.session.run(['actions'],dict(obs=row[None,1200:2230].numpy(),
            rma_obs=row[None,2230:2380].numpy(),time_step=np.full((1,1),frame,np.float32)))[0] for row in obs]))
    expected=manual(x);assert torch.equal(raw.act(x),expected) and torch.equal(full.act(x),expected)
    assert torch.equal(manual(x,frame=1),expected)
    assert full.records[-1]['new_ball_history_environments']==0
    data=torch.load(a.collection/'shoot_states.pt',map_location='cpu',weights_only=True)
    ids=[int((data['environment']==i).nonzero()[0]) for i in data['environment'].unique().tolist()[:16]]
    x=data['x'][ids].clone();full.reset();x[:,-2:]=torch.tensor([1.,0.]);x[:,940]=0
    full.act(x);x[:,940]=1;full.act(x)
    x[:,940]=0;x[:,-2:]=torch.tensor([0.,1.]);before=x.clone()
    actual=full.act(x);age=torch.zeros(len(x),dtype=torch.long)
    anchor=torch.cat((x[:,933:935],torch.ones(len(x),1)),dim=-1)
    task=torch.cat((x[:,933:939],x[:,2400:2404]),dim=-1)
    transformed=initialize_new_ball_task_history(initialize_new_ball_history(x,age,anchor),age,task)
    assert torch.equal(actual,manual(transformed)) and torch.equal(x,before)
    assert torch.isfinite(actual).all() and actual.shape==(len(x),29)
    assert full.records[-1]['new_ball_task_history_environments']==len(x)
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==digest
    report=dict(action_input_dependencies=dependencies,actions_independent_of_time_step=True,
        native_frame_zero_and_one_actions_exact=True,expert_sha256=digest,native_states=len(native),handoff_states=len(x),native_raw_actions_exact=True,
        full_history_actions_equal_manual_onnx=True,input_not_mutated=True,raw_environment_action_order_preserved=True,
        closed_loop_actions_executed=False,
        scope='Original expert ONNX routing audit only. Historical states with synthetic phase flags; no actions executed and no expert/student performance conclusion.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
