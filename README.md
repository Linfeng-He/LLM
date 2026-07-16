# vLLM benchmark runs

## Setup

These commands assume Ubuntu/Debian with `apt`. Run repo commands from this
checkout root; the checkout does not need to live at a fixed absolute path.

### Repo Paths

Set these in every new shell before setup or running experiments.

```bash
cd /path/to/inference
export INFERENCE_ROOT="$(pwd -P)"
export VLLM_REF="${VLLM_REF:-v0.10.2}"
export VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-$INFERENCE_ROOT/vllm_source_0102}"
export CPU_TORCH_INDEX_URL="${CPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
export GPU_TORCH_INDEX_URL="${GPU_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
export VLLM_GPU_PACKAGE="${VLLM_GPU_PACKAGE:-vllm[bench]==0.10.2}"
export VLLM_TRANSFORMERS_PACKAGE="${VLLM_TRANSFORMERS_PACKAGE:-transformers==4.55.4}"
export VLLM_HF_HUB_PACKAGE="${VLLM_HF_HUB_PACKAGE:-huggingface-hub<1.0}"
export VLLM_GPU_0251_TORCH_PACKAGE="${VLLM_GPU_0251_TORCH_PACKAGE:-torch==2.11.0}"
export VLLM_GPU_0251_PACKAGE="${VLLM_GPU_0251_PACKAGE:-vllm[bench]==0.25.1}"
export VLLM_GPU_0251_TRANSFORMERS_PACKAGE="${VLLM_GPU_0251_TRANSFORMERS_PACKAGE:-transformers==5.13.1}"
export VLLM_GPU_0251_HF_HUB_PACKAGE="${VLLM_GPU_0251_HF_HUB_PACKAGE:-huggingface-hub==1.23.0}"
export NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-580}"
```

### System Tools

Install build tools, Python venv support, NUMA tools, PCI inspection tools, and
Linux `perf`.

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential curl git ca-certificates pkg-config jq \
  python3 python3-venv python3-dev \
  cmake ninja-build numactl libnuma-dev libtcmalloc-minimal4 \
  gcc-12 g++-12 pciutils dkms \
  linux-tools-common linux-tools-generic

sudo apt-get install -y --no-install-recommends "linux-headers-$(uname -r)" || \
  echo "Install the linux-headers package matching $(uname -r) before installing the NVIDIA driver."

sudo apt-get install -y --no-install-recommends "linux-tools-$(uname -r)" || \
  echo "Install the linux-tools package matching $(uname -r) if perf is not on PATH."

perf --version
lspci | grep -Ei 'nvidia|vga|3d|display'
```

If `linux-tools-$(uname -r)` is unavailable for a custom/cloud kernel, install
the matching package shown by `apt-cache search "linux-tools-$(uname -r)"`, or
use the provider package such as `linux-tools-aws`, `linux-tools-azure`, or
`linux-tools-gcp`.

For full-system profiling, Ubuntu may block some `perf` events for non-root
users. Relax it only on machines where that is acceptable:

```bash
sudo sysctl kernel.perf_event_paranoid=1
sudo sysctl kernel.kptr_restrict=0
```

### NVIDIA Driver And `nvidia-smi`

`nvidia-smi` comes from the NVIDIA driver utilities package. GPU experiments
need the proprietary NVIDIA kernel driver, `/dev/nvidiactl`, and CUDA-visible
GPUs. They will not work while the open-source `nouveau` driver owns the GPUs.
If the shell says `Command 'nvidia-smi' not found` and suggests
`nvidia-utils-*`, install the matching `nvidia-driver-*` package too; utilities
alone do not load the NVIDIA kernel driver.

Recommended Ubuntu path:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ubuntu-drivers-common
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

Pinned server-driver path for reproducible bare-metal or cluster setup:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  nvidia-driver-${NVIDIA_DRIVER_VERSION}-server \
  nvidia-utils-${NVIDIA_DRIVER_VERSION}-server
sudo reboot
```

If you want a non-server driver package, install the matching desktop packages
instead:

```bash
sudo apt-get install -y --no-install-recommends \
  nvidia-driver-${NVIDIA_DRIVER_VERSION} \
  nvidia-utils-${NVIDIA_DRIVER_VERSION}
sudo reboot
```

If `lsmod | grep '^nouveau'` shows `nouveau`, blacklist it before the reboot:

```bash
printf 'blacklist nouveau\noptions nouveau modeset=0\n' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
sudo reboot
```

After reboot, verify the driver and CUDA visibility:

```bash
nvidia-smi
ls -l /dev/nvidia*
```

Secure Boot can prevent unsigned NVIDIA kernel modules from loading. Disable
Secure Boot or enroll/sign the module if `nvidia-smi` still cannot communicate
with the driver after installation.

### Python Environments

Create and activate the lightweight experiment-driver environment.

```bash
cd "$INFERENCE_ROOT"
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py

python3 -m venv .venv --without-pip
.venv/bin/python get-pip.py
source .venv/bin/activate
python -m pip install psutil matplotlib pandas
deactivate

[ -d "$VLLM_SOURCE_DIR" ] || git clone --branch "$VLLM_REF" --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_SOURCE_DIR"
```

Create and install the CPU vLLM environment.

```bash
cd "$INFERENCE_ROOT"
python3 -m venv .venv-vllm-cpu-0102 --without-pip
.venv-vllm-cpu-0102/bin/python get-pip.py
source .venv-vllm-cpu-0102/bin/activate
python -m pip install -r "$VLLM_SOURCE_DIR/requirements/cpu-build.txt" -r "$VLLM_SOURCE_DIR/requirements/cpu.txt" --extra-index-url "$CPU_TORCH_INDEX_URL"
cd "$VLLM_SOURCE_DIR"
PATH="$INFERENCE_ROOT/.venv-vllm-cpu-0102/bin:$PATH" VLLM_TARGET_DEVICE=cpu VLLM_CPU_DISABLE_AVX512=true VLLM_CPU_AVX512BF16=false VLLM_CPU_AVX512VNNI=false CC=gcc-12 CXX=g++-12 MAX_JOBS="${MAX_JOBS:-16}" python -m pip install --no-build-isolation .
cd "$INFERENCE_ROOT"
python -m pip install "$VLLM_TRANSFORMERS_PACKAGE" "$VLLM_HF_HUB_PACKAGE"
deactivate
```

Create and install the GPU vLLM environment.

```bash
cd "$INFERENCE_ROOT"
python3 -m venv .venv-vllm-gpu-0102 --without-pip
.venv-vllm-gpu-0102/bin/python get-pip.py
source .venv-vllm-gpu-0102/bin/activate
python -m pip install "$VLLM_GPU_PACKAGE" --extra-index-url "$GPU_TORCH_INDEX_URL"
python -m pip install "$VLLM_TRANSFORMERS_PACKAGE" "$VLLM_HF_HUB_PACKAGE"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("cuda_version", torch.version.cuda)
PY
deactivate
```

Create the vLLM 0.25.1 GPU environment used by the V1 A100 profiles. The
tracked installer pins the validated core environment: vLLM 0.25.1, PyTorch
2.11.0, Transformers 5.13.1, and Hugging Face Hub 1.23.0.

```bash
cd "$INFERENCE_ROOT"
bash real_runs/scripts/setup_vllm_gpu_0251_env.sh
```

The validated installation uses a CUDA 13.0 PyTorch build. Verify that the
installed NVIDIA driver supports that runtime before running the profiles.

GPU runs need a working NVIDIA driver (`nvidia-smi`), `/dev/nvidiactl`, and
CUDA-visible GPUs supported by the selected vLLM/PyTorch wheels. They are not
specific to V100S; set `GPU_TORCH_INDEX_URL` for the CUDA wheel family available
on the host, and edit `real_runs/config/gpu_modes.json` for the visible device
IDs and tensor-parallel sizes to test. The current metrics collector uses
`nvidia-smi dmon`, so AMD/Intel GPU runs require code changes before they are
valid.

Run profiles: `real_runs/config/run_profiles.json`.

## Small and medium CPU/GPU

```bash
cd /path/to/inference
source .venv/bin/activate
source real_runs/scripts/setup_run_env.sh small_medium
python -u real_runs/scripts/run_real_experiments.py 2>&1 | tee "$REAL_RUN_DIR/driver.log"
```

Expected runs: `18`.

## Medium GPU sustained

```bash
cd /path/to/inference
source .venv/bin/activate
source real_runs/scripts/setup_run_env.sh medium_gpu_sustained
python -u real_runs/scripts/run_real_experiments.py 2>&1 | tee "$REAL_RUN_DIR/driver.log"
```

Expected runs: `6`.

Outputs: `summary.json`, `all_results.csv`, `all_results.jsonl`, and per-run
`result.json`, `metrics/`, `logs/`, `figures/`.

## Qwen 7B on one A100 40GB

The A100 profiles select only `Qwen/Qwen2.5-Coder-7B-Instruct`. They use one
GPU, a 0.90 vLLM GPU-memory-utilization setting, a 32,768-token context, V0
`swap` preemption, and no benchmark or server timeout. The finite 64 GiB CPU KV
swap allocation is larger than the full requested KV footprint and is therefore
non-binding for these workloads.

Run the metric-collection smoke phase first:

```bash
cd /path/to/inference
source .venv/bin/activate
source real_runs/scripts/setup_run_env.sh a100_qwen7b_smoke
python -u real_runs/scripts/run_real_experiments.py 2>&1 | tee "$REAL_RUN_DIR/driver.log"
```

Run the 200-request ShareGPT and InstructCoder verification phase:

```bash
source real_runs/scripts/setup_run_env.sh a100_qwen7b_verify
python -u real_runs/scripts/run_real_experiments.py 2>&1 | tee "$REAL_RUN_DIR/driver.log"
```

Run the two 32-request, 30,000-output-token CPU KV-offload workloads:

```bash
source real_runs/scripts/setup_run_env.sh a100_qwen7b_offload
python -u real_runs/scripts/run_real_experiments.py 2>&1 | tee "$REAL_RUN_DIR/driver.log"
```

Results and the paper-method comparison are in
`real_runs/a100_qwen7b/final_report.md`.

## vLLM 0.25.1 A100 profiles

The newer V1 profiles use `.venv-vllm-gpu-0251`. Run the Qwen 7B smoke and
paper-validation profiles with:

```bash
cd /path/to/inference
source .venv/bin/activate
for profile in a100_qwen7b_v0251_smoke a100_qwen7b_v0251_verify; do
  source real_runs/scripts/setup_run_env.sh "$profile"
  python -u real_runs/scripts/run_real_experiments.py 2>&1 | tee "$REAL_RUN_DIR/driver.log"
done
```

Run the OPT-13B PagedAttention smoke and validation profiles with:

```bash
for profile in a100_opt13b_pagedattention_smoke a100_opt13b_pagedattention_verify; do
  source real_runs/scripts/setup_run_env.sh "$profile"
  python -u real_runs/scripts/run_real_experiments.py 2>&1 | tee "$REAL_RUN_DIR/driver.log"
done
```

Run the OPT-13B native CPU KV-offload matrix with:

```bash
for profile in \
  a100_opt13b_offload_smoke \
  a100_opt13b_offload_kv5_chat \
  a100_opt13b_offload_kv5_code \
  a100_opt13b_offload_kv10_chat \
  a100_opt13b_offload_kv10_code; do
  source real_runs/scripts/setup_run_env.sh "$profile"
  python -u real_runs/scripts/run_real_experiments.py 2>&1 | tee "$REAL_RUN_DIR/driver.log"
done
```

Downloaded datasets remain machine-local under `real_runs/datasets/`; the run
profiles and committed manifests record the required dataset names and paths.
