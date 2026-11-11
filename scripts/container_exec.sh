#!/usr/bin/env bash
set -euo pipefail
# Defaults reuse the existing local runtime; override for another installation.
container="${MOS_CONTAINER:-fiber-soccer-runtime}"
mos_root="${MOS_CONTAINER_ROOT:-/workspace/fiber-soccer}"
rolling_root="${MOS_ROLLINGBALL_ROOT:-/workspace/g1-rollingball}"
python_bin="${MOS_ISAAC_PYTHON:-/isaac-sim/python.sh}"
docker exec -w "$rolling_root" \
  -e "PYTHONPATH=$mos_root/src:$rolling_root/source/whole_body_tracking:$rolling_root/source/isaaclab:$rolling_root/source/isaaclab_tasks:$rolling_root/source/isaaclab_assets:$rolling_root/source/isaaclab_rl" \
  -e "MOS_ASSETS_DIR=${MOS_ASSETS_DIR:-$mos_root/assets}" \
  -e "SWANLAB_PROJECT=${SWANLAB_PROJECT:-MoS}" \
  -e "SWANLAB_WORKSPACE=${SWANLAB_WORKSPACE:-}" \
  "$container" "$python_bin" "$@"
