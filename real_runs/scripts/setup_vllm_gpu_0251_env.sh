#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$script_dir/../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
venv_dir="${VLLM_GPU_0251_VENV:-$root/.venv-vllm-gpu-0251}"
torch_package="${VLLM_GPU_0251_TORCH_PACKAGE:-torch==2.11.0}"
vllm_package="${VLLM_GPU_0251_PACKAGE:-vllm[bench]==0.25.1}"
transformers_package="${VLLM_GPU_0251_TRANSFORMERS_PACKAGE:-transformers==5.13.1}"
hf_hub_package="${VLLM_GPU_0251_HF_HUB_PACKAGE:-huggingface-hub==1.23.0}"

if [[ ! -x "$venv_dir/bin/python" ]]; then
  "$python_bin" -m venv "$venv_dir"
fi

"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install \
  "$torch_package" \
  "$vllm_package" \
  "$transformers_package" \
  "$hf_hub_package"
"$venv_dir/bin/python" -m pip check
"$venv_dir/bin/python" - <<'PY'
import torch
import transformers
import vllm

print("vllm", vllm.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("transformers", transformers.__version__)
PY