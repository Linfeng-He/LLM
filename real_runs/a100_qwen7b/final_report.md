# Qwen 7B vLLM A100 KV-cache offload report

Date: 2026-07-15

## Scope

Only `Qwen/Qwen2.5-Coder-7B-Instruct` was run. The experiment used vLLM's
GPU-to-CPU KV-cache swapping. Model weights stayed on the GPU; this was not a
model-weight offload experiment.

All phases passed:

| Phase | Workloads | Passed | Failed |
| --- | ---: | ---: | ---: |
| Smoke | 2 | 2 | 0 |
| Verification | 2 | 2 | 0 |
| Over-capacity offload | 2 | 2 | 0 |

## System

- GPU: NVIDIA A100-SXM4-40GB, 40,960 MiB, compute capability 8.0
- GPU power limit: 400 W
- PCIe: Gen4 x16
- Driver: 580.159.03
- CPU: AMD EPYC 7J13, 30 exposed physical/logical cores, one NUMA node
- Host RAM: 216.26 GiB; OS swap: 0
- vLLM: 0.10.2, V0 engine
- PyTorch: 2.8.0+cu128; CUDA runtime: 12.8
- Transformers: 4.55.4
- Model snapshot: `c03e6d358207e414f1eca0bb1891e29f1db0e242`
- Model weights reported by vLLM: 14.25 GiB
- GPU KV pool reported by vLLM: 16.85 GiB
- GPU memory utilization setting: 0.90
- Preemption mode: `swap`
- CPU KV swap allocation: 64 GiB
- Benchmark and server timeouts: disabled

vLLM requires a finite CPU swap allocation. The configured 64 GiB exceeded the
approximately 51.3 GiB output-KV footprint requested by 32 x 30,000 Qwen tokens,
so it was non-binding for these workloads.

## Paper check

The closest published reference is Kwon et al., "Efficient Memory Management for
Large Language Model Serving with PagedAttention" (SOSP 2023,
https://arxiv.org/abs/2309.06180). Its evaluation uses A100 40GB GPUs,
ShareGPT-derived traffic, request throughput and normalized latency, and vLLM
GPU-to-CPU KV-block swapping. Section 4.5 describes swapping evicted KV blocks
to CPU RAM; Section 7.3 compares swapping with recomputation and identifies PCIe
bandwidth as the controlling cost.

The verification phase follows that methodology where possible: A100 40GB,
ShareGPT, Poisson arrivals at 2 requests/s, serving throughput/latency, and KV
occupancy. It is not an exact numeric reproduction because the paper used
OPT-13B and an older vLLM implementation, while this run uses Qwen 7B and vLLM
0.10.2. The correctness criterion is therefore behavioral: stable service below
capacity with no queue growth, preemption, or CPU KV use, followed by measurable
CPU KV use and PCIe traffic above capacity.

FlexGen (https://arxiv.org/abs/2303.06865) was also consulted for offload
measurement methodology: report throughput, latency, GPU/CPU memory capacity,
GPU utilization, and CPU-GPU I/O. FlexGen is not used as a numeric baseline
because it offloads weights/activations as well as KV state and targets a
different serving regime.

## Smoke

Both two-request smoke workloads passed with nonempty CPU, GPU, and vLLM metric
streams. GPU board power, integrated energy, utilization, memory, and PCIe
traffic were all nonzero.

## Verification

| Workload | Requests | Req/s | Total tok/s | Output tok/s | Mean TTFT | Mean TPOT | Peak running | Peak GPU KV | CPU KV | Preemptions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT chatbot | 200/200 | 1.851 | 739.28 | 336.09 | 41.48 ms | 14.59 ms | 12 | 1.42% | 0% | 0 |
| InstructCoder | 200/200 | 1.974 | 626.30 | 265.68 | 35.18 ms | 14.05 ms | 9 | 0.77% | 0% | 0 |

Both traces completed without waiting requests, swapped requests, or CPU KV use.
This is consistent with the paper's expected below-capacity regime.

## Over-capacity results

Each workload issued 32 concurrent requests with 30,000 output tokens and
`ignore_eos`, producing exactly 960,000 output tokens. The logical requested KV
state exceeded the 16.85 GiB GPU KV pool.

| Metric | ShareGPT chatbot | InstructCoder |
| --- | ---: | ---: |
| Successful requests | 32/32 | 32/32 |
| Benchmark duration | 1,871.48 s | 1,890.19 s |
| Output throughput | 512.96 tok/s | 507.89 tok/s |
| Mean TTFT | 644.04 ms | 380.79 ms |
| Mean TPOT | 45.21 ms | 46.13 ms |
| Peak GPU KV usage | 100.0% | 100.0% |
| Peak CPU KV usage | 30.2% (~19.33 GiB) | 30.4% (~19.46 GiB) |
| Peak swapped requests | 22 | 22 |
| Preemptions | 49 | 53 |
| Peak/average GPU SM | 100% / 81.20% | 100% / 80.24% |
| Peak/average board power | 404 / 252.38 W | 319 / 249.50 W |
| Integrated GPU energy | 588.20 kJ | 586.75 kJ |
| GPU energy/output token | 0.613 J | 0.611 J |
| Peak PCIe RX+TX | 5.13 GiB/s | 4.77 GiB/s |
| Integrated PCIe RX | 102.49 GiB | 104.61 GiB |
| Integrated PCIe TX | 51.57 GiB | 82.64 GiB |
| Integrated PCIe total | 154.07 GiB | 187.25 GiB |
| Peak server RSS | 115.18 GiB | 115.20 GiB |
| Peak server CPU | 113.57% | 113.46% |

These results directly establish the requested condition: GPU KV reached 100%,
requests were swapped, CPU KV became nonzero, and sustained bidirectional PCIe
traffic was observed while all requests completed.

## Measurement notes

- GPU energy is the trapezoidal integral of one-second `nvidia-smi` whole-board
  power samples. NVIDIA documents GA100 board-power accuracy as approximately
  +/-5 W. The driver does not expose a cumulative GA100 energy counter.
- CPU utilization, process CPU time, RSS/VMS, host RAM, disk/network counters,
  and temperatures were sampled once per second.
- CPU package power and energy are unavailable because this VM exposes no Linux
  powercap/RAPL counter or power PMU. Those values are recorded as unavailable,
  not zero-valued measurements.
- PCIe RX/TX comes from `nvidia-smi dmon` and is integrated over time. On this
  single-GPU run it represents aggregate host-GPU traffic, including KV swap and
  normal serving transfers; it cannot isolate KV bytes alone.

## Artifacts

- System snapshot: `offload/system_snapshot.json`
- Run manifest: `offload/run_manifest.json`
- Consolidated results: `offload/all_results.csv` and `offload/all_results.jsonl`
- Per-workload result: `offload/qwen7b/gpu_a100/<workload>/result.json`
- vLLM benchmark detail: `offload/qwen7b/gpu_a100/<workload>/results/bench_serve.json`
- CPU timeline: `offload/qwen7b/gpu_a100/<workload>/metrics/cpu_metrics.csv`
- GPU timeline: `offload/qwen7b/gpu_a100/<workload>/metrics/gpu_dmon.csv`
- vLLM counters: `offload/qwen7b/gpu_a100/<workload>/metrics/vllm_metrics.csv`
- Parsed engine slice: `offload/qwen7b/gpu_a100/<workload>/logs/vllm_engine.log`
- Timeline figure: `offload/qwen7b/gpu_a100/<workload>/figures/*.png`
