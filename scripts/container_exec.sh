#!/usr/bin/env bash
set -euo pipefail
# Defaults reuse the existing local runtime; override for another installation.
container="${MOS_CONTAINER:-fiber-soccer-runtime}"
mos_root="${MOS_CONTAINER_ROOT:-/workspace/fiber-soccer}"
rolling_root="${MOS_ROLLINGBALL_ROOT:-/workspace/g1-rollingball}"
python_bin="${MOS_ISAAC_PYTHON:-/isaac-sim/python.sh}"
cuda_env=()
if [[ -n "${MOS_CUDA_VISIBLE_DEVICES:-}" ]]; then
  cuda_env=(-e "CUDA_VISIBLE_DEVICES=$MOS_CUDA_VISIBLE_DEVICES")
fi
command=(docker exec -w "$rolling_root" "${cuda_env[@]}" \
  -e "PYTHONPATH=$mos_root/src:$rolling_root/source/whole_body_tracking:$rolling_root/source/isaaclab:$rolling_root/source/isaaclab_tasks:$rolling_root/source/isaaclab_assets:$rolling_root/source/isaaclab_rl" \
  -e "MOS_ASSETS_DIR=${MOS_ASSETS_DIR:-$mos_root/assets}" \
  -e "MOS_SWANLAB_PROJECT=${MOS_SWANLAB_PROJECT:-MoS}" \
  -e "MOS_SWANLAB_WORKSPACE=${MOS_SWANLAB_WORKSPACE:-}" \
  "$container" "$python_bin" "$@")
# Native-model training can overlap; keep Isaac Sim instances sequential.
for argument in "$@"; do
  if [[ "$argument" == "--headless" ]]; then
    host_root="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
    mkdir -p "$host_root/runs"
    exec flock --exclusive "$host_root/runs/isaac_sim.lock" "${command[@]}"
  fi
done
exec "${command[@]}"
