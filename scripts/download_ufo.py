"""Download the paired public UFO G1 models with release checksum verification."""
import argparse, hashlib, json
from pathlib import Path
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('assets/ufo/g1_policy'));p.add_argument('--full',action='store_true');a=p.parse_args()
base='https://huggingface.co/xuewang/ufo-g1-policy/resolve/main/g1_policy/'
a.output.mkdir(parents=True,exist_ok=True)
manifest=json.load(urllib.request.urlopen(base+'release_manifest.json',timeout=60))
(a.output/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
files=['config.json','exported/FBcprAuxModel.onnx','exported/backward_encoder.onnx','exported/FBcprAuxModel.meta.json','exported/backward_encoder.meta.json','exported/FBcprAuxModel.deploy.yaml','goal_inference/goal_reaching.pkl','reward_inference/reward_locomotion.pkl']
if a.full:files+=['checkpoint/model/config.json','checkpoint/model/init_kwargs.json','checkpoint/model/model.safetensors','checkpoint/train_status.json','training_config.json']
for name in files:
 path=a.output/name;path.parent.mkdir(parents=True,exist_ok=True);expected=manifest['files'][name]
 if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256']:continue
 tmp=path.with_suffix(path.suffix+'.download')
 with urllib.request.urlopen(base+name,timeout=120) as source,tmp.open('wb') as target:
  while block:=source.read(1024*1024):target.write(block)
 if tmp.stat().st_size!=expected['bytes'] or hashlib.sha256(tmp.read_bytes()).hexdigest()!=expected['sha256']:raise RuntimeError(f'Checksum failed: {name}')
 tmp.replace(path);print('Verified',name,flush=True)
