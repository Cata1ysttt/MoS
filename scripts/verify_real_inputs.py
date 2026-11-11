import torch,numpy as np,onnxruntime as ort,json
from pathlib import Path
from onnx2torch import convert
root=Path(__file__).resolve().parents[1];torch.set_num_threads(2);torch.manual_seed(42)
m=convert(str(root/'assets/actor.onnx')).eval().requires_grad_(False)
s=ort.InferenceSession(str(root/'assets/actor.onnx'),providers=['CPUExecutionProvider'])
samples=torch.load(root/'runs/expert_smoke_v5/actor_inputs.pt',weights_only=True)
mx=0.;bad=0;n=0
for inputs in samples:
 for row in range(len(inputs[0])):
  x=[v[row:row+1] for v in inputs];z=torch.nn.functional.normalize(torch.randn(1,256),dim=-1)*16
  with torch.no_grad():a=m(*x,z).numpy()
  b=s.run(None,{k:v.numpy() for k,v in zip(['state','last_action','history_actor','z'],[*x,z])})[0]
  mx=max(mx,float(abs(a-b).max()));bad+=int(not np.allclose(a,b,atol=1e-5,rtol=1e-4));n+=1
report=dict(samples=n,strict_violations=bad,max_abs_error=mx);print(report)
(root/'assets/real_verification.json').write_text(json.dumps(report,indent=2))
