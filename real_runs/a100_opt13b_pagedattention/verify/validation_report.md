# OPT-13B PagedAttention validation

Date: 2026-07-15

## Published reference

Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention, SOSP 2023 (https://arxiv.org/abs/2309.06180), Figure 12(a).
The selected point is the paper's vLLM result at a configured request rate of
1.0 request/s: **0.03714 s/output-token**.

## Matched configuration

| Dimension | Paper | Local validation |
| --- | --- | --- |
| GPU | 1 x A100 40GB | 1 x A100-SXM4 40GB |
| Model | OPT-13B | `facebook/opt-13b` revision `e515202d...` |
| Precision | FP16 | FP16 |
| Context limit | 2,048 | 2,048 |
| Application | ShareGPT text generation | ShareGPT text generation |
| Arrivals | Poisson, 1.0 req/s | Poisson, 1.0 req/s |
| Metric | Mean E2E/output tokens | Mean E2E/output tokens |
| CPU KV offload | Not needed at low load | Disabled |

The paper used one-hour traces and its 2023 vLLM implementation. The local
validation deliberately uses a short 100-request trace and vLLM 0.25.1. The
ShareGPT dataset family and length-driven method match, but the random request
subset is not claimed to be byte-identical to the paper's unpublished trace.

## Result

| Metric | Value |
| --- | ---: |
| Successful requests | 100/100 |
| Failed requests | 0 |
| Local normalized latency | 0.02578 s/token |
| Paper normalized latency | 0.03714 s/token |
| Local difference from paper | -30.6% |
| Local P99 normalized latency | 0.03493 s/token |
| P99 difference from paper mean | -6.0% |
| Mean TTFT | 85.95 ms |
| Mean TPOT | 24.00 ms |
| Output throughput | 218.07 tok/s |
| Peak running requests | 11 |
| Peak waiting requests | 0 |
| Preemptions | 0 |
| Peak GPU KV usage | 32.3% |
| Model memory reported by vLLM | 23.94 GiB |
| GPU KV pool reported by vLLM | 10.95 GiB |
| CPU KV offload | disabled |

## Assessment

The local mean is **30.6% lower** than the published mean,
or 1.44x faster. Both lie in the same low-load
0.02-0.04 s/token regime, and the local P99 is only 6.0% below
the paper mean. This is a substantially closer validation than the Qwen-only
comparison because model, GPU, precision, application, offered rate, context,
and metric now match. It is still not an exact reproduction because the vLLM
versions and trace durations differ.

## Artifacts

- Figure: `figures/opt13b_pagedattention_validation.png`
- Vector figure: `figures/opt13b_pagedattention_validation.pdf`
- Comparison table: `validation_comparison.csv`
- Machine-readable summary: `validation_summary.json`
