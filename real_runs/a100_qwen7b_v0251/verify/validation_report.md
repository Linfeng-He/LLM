# PagedAttention method validation

Date: 2026-07-15

## Reference

Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention, SOSP 2023 (https://arxiv.org/abs/2309.06180), Figure 12(a).
The paper point uses OPT-13B on one A100 40GB with ShareGPT-derived lengths and
Poisson arrivals. Paper values were extracted from the original vector asset
with SHA256 `c374ca5253ebac3368413abeba37427ad1e7e73773d302304a3f83ec4fc80226`.

## Matched protocol

- Hardware: one NVIDIA A100 40GB
- Workload: ShareGPT input/output lengths
- Arrival process: Poisson (`burstiness=1.0`)
- Configured arrival rate: 2.0 requests/s
- Metric: mean of per-request end-to-end latency divided by output tokens
- Precision: FP16
- GPU memory utilization setting: 0.90

The local run intentionally used no CPU KV connector, swap space, or offload
backend. This is a below-capacity methodology validation, not the later pressure
experiment.

## Local result

| Metric | Value |
| --- | ---: |
| Successful requests | 200/200 |
| Failed requests | 0 |
| Achieved request throughput | 1.865 req/s |
| Output throughput | 416.72 tok/s |
| Mean TTFT | 46.00 ms |
| Mean TPOT | 12.31 ms |
| Mean E2E latency | 2.783 s |
| Mean normalized latency | 0.01335 s/token |
| P99 normalized latency | 0.02175 s/token |
| Peak running requests | 10 |
| Peak waiting requests | 0 |
| Preemptions | 0 |
| Peak GPU KV usage (V1 log) | 0.9% |
| CPU KV offload | disabled |

## Interpretation

At 2.0 requests/s, the paper's OPT-13B vLLM curve is 1.125 s/token,
while this Qwen 7B vLLM 0.25.1 run is 0.01335 s/token. These absolute
numbers are not close: the local run is 84.2x lower.
That is expected because the model architecture/size and vLLM implementation
differ. The result that is reproduced is the serving behavior: all requests
complete, the waiting queue stays at zero, no preemption occurs, and the workload
remains below GPU KV capacity. This validates the experimental method and
instrumentation; it is not an exact numerical reproduction of the 2023 model.

## Artifacts

- Figure: `figures/pagedattention_method_validation.png`
- Vector figure: `figures/pagedattention_method_validation.pdf`
- Comparison data: `validation_comparison.csv`
- Machine-readable summary: `validation_summary.json`
