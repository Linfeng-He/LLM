#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
VERIFY_DIR = ROOT / "real_runs" / "a100_opt13b_pagedattention" / "verify"
RUN_DIR = VERIFY_DIR / "opt13b" / "gpu_a100" / "sharegpt_1rps_short_trace"
REFERENCE_PATH = ROOT / "real_runs" / "paper_references" / "pagedattention_sosp2023_figure12a_vllm.json"
BENCH_PATH = RUN_DIR / "results" / "bench_serve.json"
RESULT_PATH = RUN_DIR / "result.json"
MANIFEST_PATH = VERIFY_DIR / "run_manifest.json"
SERVER_INFO_PATH = VERIFY_DIR / "opt13b" / "gpu_a100" / "server_info.json"
SERVER_LOG_PATH = VERIFY_DIR / "opt13b" / "gpu_a100" / "server.log"
OUTPUT_DIR = VERIFY_DIR / "figures"

PAPER_COLOR = "#8c510a"
LOCAL_COLOR = "#4c78a8"
LOCAL_FILL = "#d7e7f3"


def add_unified_style() -> None:
    figures_root = Path("/mnt/near-mem/rec/figures")
    if figures_root.exists() and str(figures_root) not in sys.path:
        sys.path.insert(0, str(figures_root))
    try:
        from unified_style import apply_unified_style

        apply_unified_style(plt, size=18)
    except Exception:
        plt.rcParams.update(
            {
                "font.size": 18,
                "font.weight": "bold",
                "axes.labelweight": "bold",
                "axes.titleweight": "bold",
            }
        )


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def calculate_local_latency(bench: dict) -> dict:
    normalized = []
    e2e = []
    for output_len, ttft, itls, error in zip(
        bench["output_lens"], bench["ttfts"], bench["itls"], bench["errors"]
    ):
        if error or output_len <= 0:
            continue
        latency = float(ttft) + sum(float(value) for value in itls)
        e2e.append(latency)
        normalized.append(latency / int(output_len))
    if not normalized:
        raise ValueError("benchmark contains no successful latency samples")
    return {
        "successful_requests": len(normalized),
        "mean_e2e_latency_s": sum(e2e) / len(e2e),
        "mean_normalized_latency_s_per_token": sum(normalized) / len(normalized),
        "p50_normalized_latency_s_per_token": percentile(normalized, 0.50),
        "p99_normalized_latency_s_per_token": percentile(normalized, 0.99),
        "mean_output_length_tokens": sum(bench["output_lens"]) / len(bench["output_lens"]),
    }


def extract_runtime_memory(server_log: str) -> dict:
    patterns = {
        "model_weight_gib": r"Model loading took ([0-9.]+) GiB",
        "gpu_kv_pool_gib": r"Available KV cache memory: ([0-9.]+) GiB",
        "gpu_kv_pool_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
    }
    values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, server_log)
        if not match:
            raise ValueError(f"missing runtime memory field: {key}")
        raw = match.group(1).replace(",", "")
        values[key] = int(raw) if key.endswith("tokens") else float(raw)
    return values


def validate_run(bench: dict, result: dict, manifest: dict, server_info: dict, server_log: str) -> None:
    command = server_info["server_command"]
    forbidden = {
        "--kv-offloading-size",
        "--kv-offloading-backend",
        "--swap-space",
        "--preemption-mode",
    }
    checks = {
        "passed result": result.get("status") == "passed",
        "exact model": result.get("model") == "facebook/opt-13b",
        "100 successful requests": bench.get("completed") == 100 and bench.get("failed") == 0,
        "ShareGPT Poisson 1 req/s": bench.get("request_rate") == 1.0 and bench.get("burstiness") == 1.0,
        "FP16 server": "--dtype" in command and command[command.index("--dtype") + 1] == "float16",
        "2048 context": "--max-model-len" in command and command[command.index("--max-model-len") + 1] == "2048",
        "90% GPU setting": manifest.get("gpu_memory_utilization") == "0.90",
        "zero waiting": result.get("max_requests_waiting") == 0,
        "zero preemption": result.get("preemptions_during_run") == 0,
        "offload disabled": manifest.get("gpu_kv_offloading_size_gb") is None
        and not any(flag in command for flag in forbidden),
        "no connector activity": "OffloadingConnector" not in server_log
        and "KV Transfer metrics" not in server_log,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("validation gates failed: " + ", ".join(failures))


def style_axis(axis) -> None:
    axis.grid(axis="y", linestyle="--", alpha=0.32, linewidth=1.0, zorder=0)
    axis.tick_params(axis="both", width=2.0, length=7, direction="out", labelsize=14)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(2.2)
    axis.spines["bottom"].set_linewidth(2.2)


def write_csv(path: Path, paper_value: float, local: dict) -> None:
    rows = [
        {
            "source": "PagedAttention SOSP 2023 Figure 12(a)",
            "model": "OPT-13B",
            "gpu": "A100 40GB",
            "workload": "ShareGPT, Poisson 1 req/s",
            "normalized_latency_s_per_token": paper_value,
            "statistic": "paper mean",
        },
        {
            "source": "local vLLM 0.25.1 validation",
            "model": "facebook/opt-13b",
            "gpu": "A100 40GB",
            "workload": "ShareGPT, Poisson 1 req/s, 100 requests",
            "normalized_latency_s_per_token": local["mean_normalized_latency_s_per_token"],
            "statistic": "local mean",
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    reference: dict,
    bench: dict,
    result: dict,
    local: dict,
    memory: dict,
    paper_value: float,
) -> None:
    local_value = local["mean_normalized_latency_s_per_token"]
    relative_gap = 100 * (local_value - paper_value) / paper_value
    p99_gap = 100 * (local["p99_normalized_latency_s_per_token"] - paper_value) / paper_value
    report = f"""# OPT-13B PagedAttention validation

Date: 2026-07-15

## Published reference

{reference['citation']} ({reference['paper_url']}), {reference['figure']}.
The selected point is the paper's vLLM result at a configured request rate of
1.0 request/s: **{paper_value:.5f} s/output-token**.

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
| Successful requests | {bench['completed']}/{bench['num_prompts']} |
| Failed requests | {bench['failed']} |
| Local normalized latency | {local_value:.5f} s/token |
| Paper normalized latency | {paper_value:.5f} s/token |
| Local difference from paper | {relative_gap:+.1f}% |
| Local P99 normalized latency | {local['p99_normalized_latency_s_per_token']:.5f} s/token |
| P99 difference from paper mean | {p99_gap:+.1f}% |
| Mean TTFT | {bench['mean_ttft_ms']:.2f} ms |
| Mean TPOT | {bench['mean_tpot_ms']:.2f} ms |
| Output throughput | {bench['output_throughput']:.2f} tok/s |
| Peak running requests | {result['max_requests_running']:.0f} |
| Peak waiting requests | {result['max_requests_waiting']:.0f} |
| Preemptions | {result['preemptions_during_run']:.0f} |
| Peak GPU KV usage | {result['max_gpu_kv_cache_usage_pct']:.1f}% |
| Model memory reported by vLLM | {memory['model_weight_gib']:.2f} GiB |
| GPU KV pool reported by vLLM | {memory['gpu_kv_pool_gib']:.2f} GiB |
| CPU KV offload | disabled |

## Assessment

The local mean is **{abs(relative_gap):.1f}% lower** than the published mean,
or {paper_value / local_value:.2f}x faster. Both lie in the same low-load
0.02-0.04 s/token regime, and the local P99 is only {abs(p99_gap):.1f}% below
the paper mean. This is a substantially closer validation than the Qwen-only
comparison because model, GPU, precision, application, offered rate, context,
and metric now match. It is still not an exact reproduction because the vLLM
versions and trace durations differ.

## Artifacts

- Figure: `figures/opt13b_pagedattention_validation.png`
- Vector figure: `figures/opt13b_pagedattention_validation.pdf`
- Comparison table: `validation_comparison.csv`
- Machine-readable summary: `validation_summary.json`
"""
    path.write_text(report)


def main() -> int:
    add_unified_style()
    reference = json.loads(REFERENCE_PATH.read_text())
    bench = json.loads(BENCH_PATH.read_text())
    result = json.loads(RESULT_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    server_info = json.loads(SERVER_INFO_PATH.read_text())
    server_log = SERVER_LOG_PATH.read_text(errors="replace")
    validate_run(bench, result, manifest, server_info, server_log)
    local = calculate_local_latency(bench)
    memory = extract_runtime_memory(server_log)
    paper_point = next(
        point
        for point in reference["points"]
        if point["request_rate_req_s"] == 1.0
    )
    paper_value = paper_point["normalized_latency_s_per_token"]
    local_value = local["mean_normalized_latency_s_per_token"]
    local_p50 = local["p50_normalized_latency_s_per_token"]
    local_p99 = local["p99_normalized_latency_s_per_token"]
    relative_gap = 100 * (local_value - paper_value) / paper_value

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), dpi=300)
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")

    curve_axis, bar_axis = axes
    paper_rates = [point["request_rate_req_s"] for point in reference["points"]]
    paper_latencies = [point["normalized_latency_s_per_token"] for point in reference["points"]]
    curve_axis.plot(
        paper_rates,
        paper_latencies,
        color=PAPER_COLOR,
        marker="o",
        markersize=7,
        markeredgecolor="black",
        markeredgewidth=0.8,
        linewidth=2.4,
        label="Paper vLLM curve",
        zorder=3,
    )
    curve_axis.errorbar(
        [1.0],
        [local_value],
        yerr=[[local_value - local_p50], [local_p99 - local_value]],
        color=LOCAL_COLOR,
        marker="*",
        markersize=17,
        markeredgecolor="black",
        markeredgewidth=0.9,
        elinewidth=2.0,
        capsize=5,
        linewidth=0,
        label="Our OPT-13B (P50-P99)",
        zorder=5,
    )
    curve_axis.set_yscale("log")
    curve_axis.set_xlim(0.95, 2.08)
    curve_axis.set_ylim(0.018, 1.7)
    curve_axis.set_xticks([1.0, 1.25, 1.5, 1.75, 2.0])
    curve_axis.set_xticklabels(["1.0", "1.25", "1.5", "1.75", "2.0"])
    curve_axis.set_yticks([0.02, 0.04, 0.1, 0.3, 1.0])
    curve_axis.set_yticklabels(["0.02", "0.04", "0.10", "0.30", "1.00"])
    curve_axis.set_xlabel("Offered request rate (req/s)", fontsize=16, fontweight="bold")
    curve_axis.set_ylabel("Normalized latency (s/token, log)", fontsize=16, fontweight="bold")
    curve_axis.set_title("(a) Published curve + our point", fontsize=17, fontweight="bold")
    curve_axis.annotate(
        f"Ours {local_value:.4f}\nPaper {paper_value:.4f}",
        xy=(1.0, local_value),
        xytext=(1.15, 0.024),
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.2},
        fontsize=11,
        fontweight="bold",
    )
    style_axis(curve_axis)

    bars = bar_axis.bar(
        [0, 1],
        [paper_value, local_value],
        width=0.58,
        color=[PAPER_COLOR, LOCAL_FILL],
        edgecolor="black",
        hatch=[None, "xx"],
        linewidth=0.9,
        zorder=3,
    )
    bar_axis.errorbar(
        [1],
        [local_value],
        yerr=[[local_value - local_p50], [local_p99 - local_value]],
        fmt="none",
        color=LOCAL_COLOR,
        elinewidth=2.0,
        capsize=6,
        zorder=5,
    )
    for bar, value in zip(bars, [paper_value, local_value]):
        bar_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0012,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )
    bar_axis.set_xticks([0, 1])
    bar_axis.set_xticklabels(["Paper\nOPT-13B", "Our run\nOPT-13B"], fontsize=14, fontweight="bold")
    bar_axis.set_ylim(0, 0.065)
    bar_axis.set_ylabel("Normalized latency (s/token)", fontsize=16, fontweight="bold")
    bar_axis.set_title("(b) Same-model result at 1 req/s", fontsize=17, fontweight="bold")
    bar_axis.text(
        0.5,
        0.94,
        f"Mean gap: {relative_gap:+.1f}% | Local P99: {local_p99:.4f}\n100/100 completed | Waiting: 0 | Offload: disabled",
        transform=bar_axis.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    style_axis(bar_axis)

    legend_handles = [
        Patch(facecolor=PAPER_COLOR, edgecolor="black", label="PagedAttention SOSP 2023"),
        Patch(facecolor=LOCAL_FILL, edgecolor="black", hatch="xx", label="Our vLLM 0.25.1 run"),
    ]
    fig.legend(
        handles=legend_handles,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=13,
        columnspacing=1.8,
    )
    fig.suptitle(
        "OPT-13B ShareGPT validation on one A100 40GB",
        fontsize=20,
        fontweight="bold",
        y=1.10,
    )
    fig.text(
        0.5,
        -0.01,
        "Matched: model, GPU, FP16, ShareGPT, Poisson 1 req/s, 2K context, metric. Different: vLLM version and trace duration.",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    png_path = OUTPUT_DIR / "opt13b_pagedattention_validation.png"
    pdf_path = OUTPUT_DIR / "opt13b_pagedattention_validation.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    csv_path = VERIFY_DIR / "validation_comparison.csv"
    summary_path = VERIFY_DIR / "validation_summary.json"
    report_path = VERIFY_DIR / "validation_report.md"
    write_csv(csv_path, paper_value, local)
    summary = {
        "reference": {
            "citation": reference["citation"],
            "paper_url": reference["paper_url"],
            "figure": reference["figure"],
            "model": "OPT-13B",
            "gpu": "1 x A100 40GB",
            "normalized_latency_at_1rps_s_per_token": paper_value,
            "asset_sha256": reference["source"]["asset_sha256"],
        },
        "local": {
            **local,
            "model": "facebook/opt-13b",
            "model_revision": "e515202d1e7750da62d245fbccb2723b9c1790f5",
            "vllm_version": "0.25.1",
            "configured_request_rate_req_s": bench["request_rate"],
            "achieved_request_throughput_req_s": bench["request_throughput"],
            "output_throughput_tokens_s": bench["output_throughput"],
            "mean_ttft_ms": bench["mean_ttft_ms"],
            "mean_tpot_ms": bench["mean_tpot_ms"],
            "peak_running_requests": result["max_requests_running"],
            "peak_waiting_requests": result["max_requests_waiting"],
            "preemptions": result["preemptions_during_run"],
            "peak_gpu_kv_usage_pct": result["max_gpu_kv_cache_usage_pct"],
            "cpu_kv_offload_enabled": False,
            **memory,
        },
        "comparison": {
            "local_relative_difference_pct": relative_gap,
            "paper_to_local_latency_ratio": paper_value / local_value,
            "local_p99_relative_difference_pct": 100 * (local_p99 - paper_value) / paper_value,
            "assessment": "same low-load latency regime; newer vLLM is faster",
        },
        "matched_dimensions": [
            "OPT-13B model",
            "one A100 40GB",
            "FP16",
            "ShareGPT text generation",
            "Poisson arrivals at 1 request/s",
            "2048-token context",
            "mean E2E latency divided by output tokens",
        ],
        "remaining_differences": [
            "vLLM 0.25.1 versus the paper's 2023 implementation",
            "100-request short trace versus one-hour paper trace",
            "ShareGPT request subset is not guaranteed byte-identical",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_report(report_path, reference, bench, result, local, memory, paper_value)

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())