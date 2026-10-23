#!/usr/bin/env bash
# Native MuJoCo viewer on WSLg with NVIDIA OpenGL acceleration.
# Usage: bash scripts/play_native_wsl.sh <TASK_ID> --checkpoint-file <model.pt> --num-envs 1
# All arguments are forwarded to mjlab's play command; do not pass --viewer.
# These environment variables apply only to this process and its children.
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$project_dir"

if command -v uv >/dev/null 2>&1; then
    uv_bin=$(command -v uv)
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
else
    echo 'uv not found. Install uv or add it to PATH.' >&2
    exit 127
fi

export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export MUJOCO_GL=glfw

exec "$uv_bin" run --locked --no-sync play "$@" --viewer native
