"""Check paired UFO ONNX exports and differentiability before simulator integration."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
import onnx
from onnx2torch import convert
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,default=Path('assets/ufo/g1_policy'));p.add_argument('--output',type=Path,default=Path('runs/ufo_export_audit.json'));a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(17);rng=np.random.default_rng(17);report={}
for name in ['FBcprAuxModel','backward_encoder']:
 path=a.assets/'exported'/f'{name}.onnx';session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']);graph=onnx.load(str(path))
 # ONNX represents omitted optional Clip max by an empty trailing name.
 # Remove only omitted trailing arguments for onnx2torch compatibility.
 for node in graph.graph.node:
  if node.op_type=='Clip':
   while node.input and node.input[-1]=='':node.input.pop()
 model=convert(graph).eval().requires_grad_(False)
 inputs={}
 for item in session.get_inputs():
  shape=[8 if isinstance(v,str) or v is None else v for v in item.shape];shape[0]=8
  value=rng.normal(0,.1,shape).astype(np.float32)
  if 'z'==item.name:value=value/np.linalg.norm(value,axis=-1,keepdims=True)*16
  if item.name=='actor_obs':value[:,-256:]=value[:,-256:]/np.linalg.norm(value[:,-256:],axis=-1,keepdims=True)*16
  inputs[item.name]=value
 tensors=[torch.tensor(v,requires_grad=(k in ('z','actor_obs'))) for k,v in inputs.items()]
 output=model(*tensors);output=output[0] if isinstance(output,tuple) else output
 expected=session.run(None,inputs)[0]
 error=float(np.max(np.abs(output.detach().numpy()-expected)))
 assert np.isfinite(expected).all() and error<1e-4,(name,error)
 entry=dict(inputs={k:list(v.shape) for k,v in inputs.items()},output_shape=list(expected.shape),max_abs_conversion_error=error)
 if any(t.requires_grad for t in tensors):
  output.square().mean().backward();entry['latent_gradient_finite']=all(bool(torch.isfinite(t.grad).all()) for t in tensors if t.requires_grad);assert entry['latent_gradient_finite']
 if name=='backward_encoder':entry['output_norms']=np.linalg.norm(expected,axis=-1).tolist()
 report[name]=entry
report['scope']='Synthetic input conversion and gradient check only; not real-state contract or passing validation.'
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
