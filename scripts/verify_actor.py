import json,hashlib
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from onnx2torch import convert
root=Path(__file__).resolve().parents[1];torch.set_num_threads(2);torch.manual_seed(42)
model=convert(str(root/'assets/actor.onnx')).eval().requires_grad_(False)
options=ort.SessionOptions();options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session=ort.InferenceSession(str(root/'assets/actor.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
maximum=0.;violations=0
for i in range(1000):
    state=torch.randn(1,64)*.2;state[:,58:61]=torch.tensor([0.,0.,-1.])
    inputs=[state,torch.randn(1,29)*.2,torch.randn(1,837)*.2,torch.randn(1,256)]
    inputs[-1]=torch.nn.functional.normalize(inputs[-1],dim=-1)*16
    with torch.no_grad():actual=model(*inputs).numpy()
    expected=session.run(None,{n:x.numpy() for n,x in zip(['state','last_action','history_actor','z'],inputs)})[0]
    violations+=int(not np.allclose(actual,expected,atol=1e-5,rtol=1e-4))
    maximum=max(maximum,float(abs(actual-expected).max()))
z=inputs[-1].requires_grad_(True);direction=torch.randn_like(z);direction/=direction.norm()
y=model(*inputs[:-1],z).square().sum();grad=torch.autograd.grad(y,z)[0];eps=.002
fd=(model(*inputs[:-1],z.detach()+eps*direction).square().sum()-model(*inputs[:-1],z.detach()-eps*direction).square().sum())/(2*eps)
analytical=(grad*direction).sum();torch.testing.assert_close(fd,analytical,atol=.002,rtol=.03)
assert grad.norm()>0 and all(p.grad is None for p in model.parameters())
report=dict(samples=1000,violating_samples=violations,passed=violations==0,max_abs_error=maximum,gradient=float(analytical),finite_difference=float(fd),actor_sha256=hashlib.sha256((root/'assets/actor.onnx').read_bytes()).hexdigest())
(root/'assets/verification.json').write_text(json.dumps(report,indent=2));print(report)
# Trusted local conversion artifact, not an official PyTorch checkpoint.
if violations:raise RuntimeError(f'Actor equivalence gate failed: {violations}/1000 samples; see verification.json')
torch.save(model,root/'assets/actor_converted.pt')
