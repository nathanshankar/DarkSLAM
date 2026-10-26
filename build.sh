#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
torch_prefix="$("$python_bin" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S "$project_dir" -B "$project_dir/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$torch_prefix" "$@"
cmake --build "$project_dir/build" --parallel "${JOBS:-2}"
