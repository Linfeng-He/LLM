# Selected Sustained GPU Workload Parameters

Calibration target: medium model (`Qwen/Qwen2.5-Coder-7B-Instruct`) with vLLM, longer benchmark windows, nonempty telemetry, and high GPU memory pressure.

Recommended workload file:

`/users/LH/inference/real_runs/scripts/calibration_gpu_final_workloads.json`

Recommended GPU server setting:

`REAL_RUN_GPU_MEMORY_UTILIZATION=0.95`

The driver now launches vLLM with `--max-num-seqs` equal to the largest workload `max_num_seqs`, so the server can actually exercise concurrency 8 during these runs.

## Final Parameters

| workload | input_len | output_len | num_prompts | max_concurrency |
|---|---:|---:|---:|---:|
| chatbot | 512 | 1024 | 48 | 8 |
| code_generation | 2048 | 1024 | 48 | 8 |
| long_conversation | 6144 | 1024 | 24 | 4 |

## Final 2-GPU Validation

Result root: `/users/LH/inference/real_runs/calibration_medium_gpu_final`

| workload | status | wall duration | bench duration | max GPU FB | max PCIe RX+TX | validation |
|---|---:|---:|---:|---:|---:|---|
| chatbot | passed | 137.2s | 113.7s | 31.23 GiB | 10.85 GiB/s | no issues |
| code_generation | passed | 153.4s | 129.4s | 31.49 GiB | 11.72 GiB/s | no issues |
| long_conversation | passed | 161.3s | 135.4s | 31.35 GiB | 10.41 GiB/s | no issues |

## Earlier 1-GPU Scaling Check

Result root: `/users/LH/inference/real_runs/calibration_medium_gpu_sustained`

This used the same input/output/concurrency shape, but fewer prompts and `REAL_RUN_GPU_MEMORY_UTILIZATION=0.80`.

| workload | prompts | status | wall duration | max GPU FB | max PCIe RX+TX | validation |
|---|---:|---:|---:|---:|---:|---|
| chatbot | 32 | passed | 144.0s | 26.21 GiB | 3.35 GiB/s | no issues |
| code_generation | 32 | passed | 160.6s | 26.80 GiB | 3.97 GiB/s | no issues |
| long_conversation | 16 | passed | 171.8s | 26.80 GiB | 3.83 GiB/s | no issues |

The final prompt counts are 1.5x higher, so expected 1-GPU wall durations are roughly 216s, 241s, and 258s respectively, with higher memory pressure when using `REAL_RUN_GPU_MEMORY_UTILIZATION=0.95`.

## Run Command Pattern

```bash
REAL_RUN_WORKLOADS_FILE=real_runs/scripts/calibration_gpu_final_workloads.json \
REAL_RUN_GPU_MEMORY_UTILIZATION=0.95 \
REAL_RUN_GPU_MODES=gpu_1,gpu_2 \
REAL_RUN_TARGETS=gpu \
REAL_RUN_MODEL_SIZES=medium \
REAL_RUN_BENCH_TIMEOUT_S=0 \
REAL_RUN_SERVER_READY_TIMEOUT_S=0 \
.venv/bin/python -u real_runs/scripts/run_real_experiments.py
```
