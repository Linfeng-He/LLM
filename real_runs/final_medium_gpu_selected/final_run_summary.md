# Final Medium GPU Selected Run

Model: `Qwen/Qwen2.5-Coder-7B-Instruct`

GPU memory target: `REAL_RUN_GPU_MEMORY_UTILIZATION=0.95`

Workload file: `/users/LH/inference/real_runs/scripts/calibration_gpu_final_workloads.json`

| target | workload | status | duration | max GPU FB | max PCIe RX+TX | output tok/s | validation |
|---|---|---:|---:|---:|---:|---:|---|
| gpu_1 | final_chatbot | passed | 200.6s | 30.97 GiB | 4.20 GiB/s | 289.6 | no issues |
| gpu_1 | final_code_generation | passed | 225.7s | 31.56 GiB | 4.25 GiB/s | 253.3 | no issues |
| gpu_1 | final_long_conversation | passed | 241.2s | 31.56 GiB | 4.72 GiB/s | 118.4 | no issues |
| gpu_2 | final_chatbot | passed | 137.4s | 31.29 GiB | 11.01 GiB/s | 431.5 | no issues |
| gpu_2 | final_code_generation | passed | 153.8s | 31.61 GiB | 11.05 GiB/s | 378.6 | no issues |
| gpu_2 | final_long_conversation | passed | 160.8s | 31.37 GiB | 10.46 GiB/s | 181.2 | no issues |

All six runs produced result JSON, benchmark JSON, CPU metrics, GPU dmon metrics, raw GPU dmon logs, and timeline figures.
