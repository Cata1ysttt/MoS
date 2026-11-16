"""Capture official preprocessed reference for identical-reference Isaac tests.

Run with UFO on PYTHONPATH and its existing mjlab runtime. Uses the official
motion loader rather than guessing interpolation or quaternion conventions.
"""
import argparse,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--model-folder',required=True);p.add_argument('--data-path',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise RuntimeError('Refuse overwrite reference')
from ufo.inference import tracking_inference as tracking
original=tracking.get_backward_observation

def capture(*args,**kwargs):
    backward,reference=original(*args,**kwargs)
    payload={k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in reference.items()}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(reference=payload,source='Official UFO get_backward_observation',motion_id=7),a.output)
    return backward,reference
tracking.get_backward_observation=capture
sys.argv=['tracking','--model-folder',a.model_folder,'--data-path',a.data_path,'--motion-list','7','--max-steps','1','--disable-dr','--disable-obs-noise','--export-onnx','false']
tracking.main()
