"""Resource paths shared by MoS training and evaluation."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get('MOS_ASSETS_DIR', str(ROOT / 'assets')))
