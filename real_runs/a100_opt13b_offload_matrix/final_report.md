# OPT-13B directional KV-offload matrix

Date: 2026-07-15

## Design

- Model: `facebook/opt-13b` at revision `e515202d...`, FP16, one A100 40GB.
- Applications: ShareGPT chat and InstructCoder code editing.
- Each application: 22 prompts x (251 input + 1,536 output tokens), exactly 29.994 GiB logical KV.
- Native vLLM 0.25.1 `OffloadingConnector` with `offload_prompt_only=false` and a 40 GiB CPU buffer.
- Each cell starts and stops a fresh server; no CPU/GPU cache state crosses cells.

## GPU budget calculation

Measured OPT-13B model memory was 23.94 GiB and measured non-KV runtime overhead was 1.11 GiB, for 25.05 GiB before the GPU KV pool.

| Requested GPU KV headroom | Executor target | A100 percentage | Feasible |
| ---: | ---: | ---: | :---: |
| +5 GiB | 30.05 GiB | 75.125% | Yes |
| +10 GiB | 35.05 GiB | 87.625% | Yes |
| +20 GiB | 45.05 GiB | 112.625% | No, omitted |

## Results

| Budget | App | Output tok/s | Mean TPOT | GPU→CPU KV | CPU→GPU KV | Load ops | 1s load bursts | Preemptions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| +5 GiB (75.125%) | Chat | 217.56 | 61.45 ms | 28.45 GiB | 27.76 GiB | 57 | 16 | 228 |
| +5 GiB (75.125%) | Code | 234.24 | 59.09 ms | 29.81 GiB | 37.00 GiB | 71 | 22 | 285 |
| +10 GiB (87.625%) | Chat | 348.12 | 41.76 ms | 28.45 GiB | 23.39 GiB | 35 | 8 | 291 |
| +10 GiB (87.625%) | Code | 377.57 | 42.27 ms | 29.81 GiB | 27.87 GiB | 39 | 10 | 196 |

## Direction and revisits

NVIDIA defines PCIe TX/RX from the GPU's perspective: TX is GPU-to-host and RX is host-to-GPU. In the connector, store is GPU-to-CPU and load is CPU-to-GPU. Connector counters are KV-specific and establish attribution; dmon is aggregate PCIe traffic and is used for the time shape.

CPU-to-GPU reloads occurred in every cell, so offloaded KV is revisited. Connector-reported load operations ranged from 35 to 71 per cell. At one-second resolution, these grouped into 8 to 22 observed bursts. The exact per-cell operation rate and seconds per burst are in `matrix_summary.csv`.

Increasing the GPU KV budget from +5 to +10 GiB raised chat throughput by 60.0% and code throughput by 61.2%. It reduced connector load-operation count by 38.6% for chat and 45.1% for code.

## Memory interpretation

The timeline figure excludes model weights and runtime allocations. Panel (a) shows only live KV bytes: CPU occupancy multiplied by the 40 GiB offload buffer and GPU occupancy multiplied by the actual GPU KV pool reported by vLLM. Panel (b) shows the same occupancy as percentages. Raw physical CPU RSS and GPU FB allocation remain available in each cell's metric CSV and in `combined_timeline_1s.csv`.

## Measurement limits

- `nvidia-smi dmon` PCIe values are GPU-centric MB/s over the previous 20 ms, sampled once per second. Short bursts can be aliased or missed.
- Hardware PCIe includes model/runtime transfers as well as KV traffic; it is not used alone to attribute bytes to KV offload.
- Connector load/store counters are authoritative for KV direction and totals, but operation count is connector-reported transfer operations, not individual KV blocks.
- Output throughput in tokens/s is the primary GPU serving-performance unit, consistent with vLLM/Punica-style systems papers. TTFT and TPOT are retained in the summary data.

## Artifacts

- Timeline figure: `figures/opt13b_offload_timelines.png` and `.pdf`
- Performance/traffic summary: `figures/opt13b_offload_summary.png` and `.pdf`
- Cell summary: `matrix_summary.csv` and `matrix_summary.json`
- Resampled timeline: `combined_timeline_1s.csv`
- Original one-second streams remain under each cell's `metrics/` directory.
