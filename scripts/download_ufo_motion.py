"""Fetch official full G1 motion set for native simulator verification."""
import hashlib,urllib.request
from pathlib import Path
p=Path('/home/jiongye/UFO/ufo/data/lafan_29dof.pkl');expected='f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11'
if not p.exists():
 tmp=p.with_suffix('.download')
 with urllib.request.urlopen('https://huggingface.co/datasets/xuewang/UFO-MotionData/resolve/main/g1/lafan_29dof.pkl',timeout=120) as src,tmp.open('wb') as dst:
  while b:=src.read(1024*1024):dst.write(b)
 assert hashlib.sha256(tmp.read_bytes()).hexdigest()==expected
 tmp.replace(p)
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
print('Official motion checksum verified',flush=True)
