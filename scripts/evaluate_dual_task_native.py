"""Use the unchanged native evaluation logic with the explicit dual-task loader."""
import argparse
from pathlib import Path
import runpy
import sys
import torch
from fiber_soccer import planner_policy
from fiber_soccer.dual_task_latent import DualTaskPolicy

p=argparse.ArgumentParser(add_help=False);p.add_argument('--skill',choices=['pass','shoot'],required=True)
a,rest=p.parse_known_args()
original=planner_policy.load_soccer_policy

def load(checkpoint,device='cpu',**kwargs):
    header=torch.load(checkpoint,map_location='cpu',weights_only=False) if isinstance(checkpoint,(str,Path)) else checkpoint
    if 'dual_task_adapter' in header:
        with torch.inference_mode(False):return DualTaskPolicy(header,device,**kwargs)
    return original(header,device,**kwargs)

planner_policy.load_soccer_policy=load
entry=Path(__file__).with_name('evaluate_selected_pass_expert.py' if a.skill=='pass' else 'probe_shoot_expert.py')
sys.argv=[str(entry),*rest]
runpy.run_path(str(entry),run_name='__main__')
