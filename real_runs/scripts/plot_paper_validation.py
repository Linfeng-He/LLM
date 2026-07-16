#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VERIFY_DIR = ROOT / "real_runs" / "a100_qwen7b_v0251" / "verify"
RUN_DIR = VERIFY_DIR / "qwen7b" / "gpu_a100" / "paper_validation_sharegpt_2rps"
REFERENCE_PATH = ROOT / "real_runs" / "paper_references" / "pagedattention_sosp2023_figure12a_vllm.json"
BENCH_PATH = RUN_DIR / "results" / "bench_serve.json"
RESULT_PATH = RUN_DIR / "result.json"
MANIFEST_PATH = VERIFY_DIR / "run_manifest.json"
SERVER_INFO_PATH = VERIFY_DIR / "qwen7b" / "gpu_a100" / "server_info.json"
SERVER_LOG_PATH = VERIFY_DIR / "qwen7b" / "gpu_a100" / "server.log"
ENGINE_METRICS_PATH = RUN_DIR / "metrics" / "vllm_metrics.csv"
OUTPUT_DIR = VERIFY_DIR / "figures"

PAPER_COLOR = "#8c510a"
LOCAL_COLOR = "#4c78a8"
WAITING_COLOR = "#9aa3ad"
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
    normalized_latencies = []
    e2e_latencies = []
    for output_len, ttft, itls, error in zip(
        bench["output_lens"], bench["ttfts"], bench["itls"], bench["errors"]
    ):
        if error or output_len <= 0:
            continue
        e2e_latency = float(ttft) + sum(float(value) for value in itls)
        e2e_latencies.append(e2e_latency)
        normalized_latencies.append(e2e_latency / int(output_len))
    if not normalized_latencies:
        raise ValueError("benchmark contains no successful per-request latency samples")
    return {
        "successful_requests": len(normalized_latencies),
        "mean_e2e_latency_s": sum(e2e_latencies) / len(e2e_latencies),
        "mean_normalized_latency_s_per_token": sum(normalized_latencies) / len(normalized_latencies),
        "p50_normalized_latency_s_per_token": percentile(normalized_latencies, 0.50),
        "p99_normalized_latency_s_per_token": percentile(normalized_latencies, 0.99),
        "mean_output_length_tokens": sum(bench["output_lens"]) / len(bench["output_lens"]),
    }


def load_engine_timeline() -> tuple[list[float], list[float], list[float]]:
    elapsed = []
    running = []
    waiting = []
    with ENGINE_METRICS_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("elapsed_s"):
                continue
            elapsed.append(float(row["elapsed_s"]))
            running.append(float(row.get("num_requests_running") or 0))
            waiting.append(float(row.get("num_requests_waiting") or 0))
    if not elapsed:
        raise ValueError("engine metric timeline is empty")
    return elapsed, running, waiting


def peak_gpu_kv_usage_pct(server_log: str) -> float:
    values = [
        float(match.group(1))
        for match in re.finditer(r"GPU KV cache usage: ([0-9.]+)%", server_log)
    ]
    if not values:
        raise ValueError("V1 GPU KV usage was not found in the server log")
    return max(values)


def validate_run(bench: dict, result: dict, manifest: dict, server_info: dict, server_log: str) -> None:
    server_command = server_info["server_command"]
    forbidden_flags = {
        "--kv-offloading-size",
        "--kv-offloading-backend",
        "--swap-space",
        "--preemption-mode",
    }
    present = sorted(flag for flag in forbidden_flags if flag in server_command)
    checks = {
        "run status": result.get("status") == "passed",
        "200 successful requests": bench.get("completed") == 200 and bench.get("failed") == 0,
        "ShareGPT Poisson rate": bench.get("request_rate") == 2.0 and bench.get("burstiness") == 1.0,
        "zero waiting": result.get("max_requests_waiting") == 0,
        "zero preemptions": result.get("preemptions_during_run") == 0,
        "V1 engine": manifest.get("gpu_engine") == "v1",
        "90% GPU memory setting": manifest.get("gpu_memory_utilization") == "0.90",
        "CPU offload disabled": manifest.get("gpu_kv_offloading_size_gb") is None and not present,
        "no offload connector log": "OffloadingConnector" not in server_log and "KV Transfer metrics" not in server_log,
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


def write_comparison_csv(path: Path, reference: dict, local: dict, bench: dict) -> None:
    fieldnames = [
        "source",
        "model",
        "vllm_version",
        "gpu",
        "dataset",
        "request_rate_req_s",
        "normalized_latency_s_per_token",
        "statistic",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in reference["points"]:
            writer.writerow(
                {
                    "source": "PagedAttention Figure 12(a)",
                    "model": reference["configuration"]["model"],
                    "vllm_version": "SOSP 2023 implementation",
                    "gpu": reference["configuration"]["gpu"],
                    "dataset": "ShareGPT",
                    "request_rate_req_s": point["request_rate_req_s"],
                    "normalized_latency_s_per_token": point["normalized_latency_s_per_token"],
                    "statistic": "paper curve",
                }
            )
        writer.writerow(
            {
                "source": "local validation",
                "model": "Qwen2.5-Coder-7B-Instruct",
                "vllm_version": "0.25.1",
                "gpu": "1 x NVIDIA A100 40GB",
                "dataset": "ShareGPT",
                "request_rate_req_s": bench["request_rate"],
                "normalized_latency_s_per_token": local["mean_normalized_latency_s_per_token"],
                "statistic": "local mean",
            }
        )


def write_report(path: Path, reference: dict, local: dict, bench: dict, result: dict, peak_gpu_kv: float) -> None:
    paper_at_two = next(
        point["normalized_latency_s_per_token"]
        for point in reference["points"]
        if point["request_rate_req_s"] == 2.0
    )
    local_value = local["mean_normalized_latency_s_per_token"]
    report = f"""# PagedAttention method validation

Date: 2026-07-15

## Reference

{reference['citation']} ({reference['paper_url']}), {reference['figure']}.
The paper point uses OPT-13B on one A100 40GB with ShareGPT-derived lengths and
Poisson arrivals. Paper values were extracted from the original vector asset
with SHA256 `{reference['source']['asset_sha256']}`.

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
| Successful requests | {bench['completed']}/{bench['num_prompts']} |
| Failed requests | {bench['failed']} |
| Achieved request throughput | {bench['request_throughput']:.3f} req/s |
| Output throughput | {bench['output_throughput']:.2f} tok/s |
| Mean TTFT | {bench['mean_ttft_ms']:.2f} ms |
| Mean TPOT | {bench['mean_tpot_ms']:.2f} ms |
| Mean E2E latency | {local['mean_e2e_latency_s']:.3f} s |
| Mean normalized latency | {local_value:.5f} s/token |
| P99 normalized latency | {local['p99_normalized_latency_s_per_token']:.5f} s/token |
| Peak running requests | {result['max_requests_running']:.0f} |
| Peak waiting requests | {result['max_requests_waiting']:.0f} |
| Preemptions | {result['preemptions_during_run']:.0f} |
| Peak GPU KV usage (V1 log) | {peak_gpu_kv:.1f}% |
| CPU KV offload | disabled |

## Interpretation

At 2.0 requests/s, the paper's OPT-13B vLLM curve is {paper_at_two:.3f} s/token,
while this Qwen 7B vLLM 0.25.1 run is {local_value:.5f} s/token. These absolute
numbers are not close: the local run is {paper_at_two / local_value:.1f}x lower.
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
    elapsed, running, waiting = load_engine_timeline()
    peak_gpu_kv = peak_gpu_kv_usage_pct(server_log)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paper_rates = [point["request_rate_req_s"] for point in reference["points"]]
    paper_latencies = [point["normalized_latency_s_per_token"] for point in reference["points"]]
    local_mean = local["mean_normalized_latency_s_per_token"]
    local_p50 = local["p50_normalized_latency_s_per_token"]
    local_p99 = local["p99_normalized_latency_s_per_token"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=300)
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")

    latency_axis, queue_axis = axes
    latency_axis.plot(
        paper_rates,
        paper_latencies,
        color=PAPER_COLOR,
        marker="o",
        markersize=7,
        markeredgecolor="black",
        markeredgewidth=0.8,
        linewidth=2.4,
        label="Paper: OPT-13B vLLM",
        zorder=3,
    )
    latency_axis.errorbar(
        [bench["request_rate"]],
        [local_mean],
        yerr=[[local_mean - local_p50], [local_p99 - local_mean]],
        color=LOCAL_COLOR,
        marker="*",
        markersize=16,
        markeredgecolor="black",
        markeredgewidth=0.9,
        elinewidth=2.0,
        capsize=5,
        linewidth=0,
        label="Our run: Qwen 7B (P50-P99)",
        zorder=5,
    )
    latency_axis.set_yscale("log")
    latency_axis.set_xlim(0.95, 2.08)
    latency_axis.set_ylim(0.01, 1.7)
    latency_axis.set_xticks([1.0, 1.25, 1.5, 1.75, 2.0])
    latency_axis.set_xticklabels(["1.0", "1.25", "1.5", "1.75", "2.0"])
    latency_axis.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0])
    latency_axis.set_yticklabels(["0.01", "0.03", "0.10", "0.30", "1.00"])
    latency_axis.set_xlabel("Request rate (req/s)", fontsize=17, fontweight="bold")
    latency_axis.set_ylabel("Normalized latency (s/token, log)", fontsize=17, fontweight="bold")
    latency_axis.set_title("(a) Paper metric comparison", fontsize=18, fontweight="bold")
    latency_axis.annotate(
        f"Our mean: {local_mean:.4f}",
        xy=(2.0, local_mean),
        xytext=(1.55, 0.018),
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.2},
        fontsize=12,
        fontweight="bold",
    )
    style_axis(latency_axis)

    queue_axis.fill_between(
        elapsed,
        running,
        step="post",
        color=LOCAL_FILL,
        edgecolor=LOCAL_COLOR,
        hatch="//",
        linewidth=0.8,
        alpha=0.75,
        zorder=2,
    )
    queue_axis.step(
        elapsed,
        running,
        where="post",
        color=LOCAL_COLOR,
        linewidth=2.2,
        label="Running requests",
        zorder=3,
    )
    queue_axis.step(
        elapsed,
        waiting,
        where="post",
        color=WAITING_COLOR,
        linestyle="--",
        linewidth=2.4,
        label="Waiting requests",
        zorder=4,
    )
    queue_axis.set_xlim(0, max(elapsed))
    queue_axis.set_ylim(-0.3, max(running) + 2.0)
    queue_axis.set_xlabel("Validation time (s)", fontsize=17, fontweight="bold")
    queue_axis.set_ylabel("Requests", fontsize=17, fontweight="bold")
    queue_axis.set_title("(b) Queue stability at 2 req/s", fontsize=18, fontweight="bold")
    queue_axis.text(
        0.98,
        0.96,
        f"200/200 completed\nWaiting: 0\nPreemptions: 0\nPeak GPU KV: {peak_gpu_kv:.1f}%\nCPU offload: disabled",
        transform=queue_axis.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    style_axis(queue_axis)

    handles, labels = [], []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=12,
        handlelength=2.0,
        columnspacing=1.4,
    )
    fig.suptitle(
        "PagedAttention protocol validation on A100 40GB",
        fontsize=20,
        fontweight="bold",
        y=1.10,
    )
    fig.text(
        0.5,
        -0.01,
        "Paper curve: SOSP 2023 Figure 12(a). Different model/version: behavioral validation, not exact numerical reproduction.",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))

    png_path = OUTPUT_DIR / "pagedattention_method_validation.png"
    pdf_path = OUTPUT_DIR / "pagedattention_method_validation.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    comparison_path = VERIFY_DIR / "validation_comparison.csv"
    summary_path = VERIFY_DIR / "validation_summary.json"
    report_path = VERIFY_DIR / "validation_report.md"
    write_comparison_csv(comparison_path, reference, local, bench)
    paper_at_two = next(
        point["normalized_latency_s_per_token"]
        for point in reference["points"]
        if point["request_rate_req_s"] == 2.0
    )
    summary = {
        "reference": {
            "citation": reference["citation"],
            "paper_url": reference["paper_url"],
            "figure": reference["figure"],
            "asset_sha256": reference["source"]["asset_sha256"],
            "normalized_latency_at_2rps_s_per_token": paper_at_two,
        },
        "local": {
            **local,
            "configured_request_rate_req_s": bench["request_rate"],
            "achieved_request_rate_req_s": bench["request_throughput"],
            "output_throughput_tokens_s": bench["output_throughput"],
            "mean_ttft_ms": bench["mean_ttft_ms"],
            "mean_tpot_ms": bench["mean_tpot_ms"],
            "peak_running_requests": result["max_requests_running"],
            "peak_waiting_requests": result["max_requests_waiting"],
            "preemptions": result["preemptions_during_run"],
            "peak_gpu_kv_usage_pct": peak_gpu_kv,
            "cpu_kv_offload_enabled": False,
        },
        "comparison": {
            "paper_to_local_latency_ratio_at_2rps": paper_at_two / local_mean,
            "interpretation": "behavioral method validation, not exact numerical reproduction",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_report(report_path, reference, local, bench, result, peak_gpu_kv)

    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"CSV: {comparison_path}")
    print(f"JSON: {summary_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())