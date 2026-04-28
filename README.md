# vLLM benchmark runs

## Setup

```bash
cd /users/LH/inference

curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo apt-get update
sudo apt-get install -y --no-install-recommends cmake ninja-build numactl libnuma-dev libtcmalloc-minimal4 gcc-12 g++-12

python3 -m venv .venv --without-pip
.venv/bin/python get-pip.py
.venv/bin/python -m pip install psutil matplotlib pandas

[ -d vllm_source_0102 ] || git clone --branch v0.10.2 --depth 1 https://github.com/vllm-project/vllm.git vllm_source_0102

python3 -m venv .venv-vllm-cpu-0102 --without-pip
.venv-vllm-cpu-0102/bin/python get-pip.py
.venv-vllm-cpu-0102/bin/python -m pip install -r vllm_source_0102/requirements/cpu-build.txt -r vllm_source_0102/requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd vllm_source_0102
PATH="$PWD/../.venv-vllm-cpu-0102/bin:$PATH" VLLM_TARGET_DEVICE=cpu VLLM_CPU_DISABLE_AVX512=true VLLM_CPU_AVX512BF16=false VLLM_CPU_AVX512VNNI=false CC=gcc-12 CXX=g++-12 MAX_JOBS=16 ../.venv-vllm-cpu-0102/bin/python -m pip install --no-build-isolation .
cd ..

python3 -m venv .venv-vllm-gpu-0102 --without-pip
.venv-vllm-gpu-0102/bin/python get-pip.py
.venv-vllm-gpu-0102/bin/python -m pip install 'vllm==0.10.2' --extra-index-url https://download.pytorch.org/whl/cu128
```

GPU runs need `nvidia-smi`, `/dev/nvidiactl`, and CUDA-visible V100S cards.

Run profiles: `real_runs/config/run_profiles.json`.

## Small and medium CPU/GPU

```bash
cd /users/LH/inference
source real_runs/scripts/setup_run_env.sh small_medium
.venv/bin/python -u real_runs/scripts/run_real_experiments.py 2>&1 | tee "$REAL_RUN_DIR/driver.log"
```

Expected runs: `18`.

## Medium GPU sustained

```bash
cd /users/LH/inference
source real_runs/scripts/setup_run_env.sh medium_gpu_sustained
.venv/bin/python -u real_runs/scripts/run_real_experiments.py 2>&1 | tee "$REAL_RUN_DIR/driver.log"
```

Expected runs: `6`.

Outputs: `summary.json`, `all_results.csv`, `all_results.jsonl`, and per-run `result.json`, `metrics/`, `logs/`, `figures/`.
