#!/usr/bin/env python3
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
SMALL = ROOT / "real_runs" / "small"
OUT = SMALL / "generated_figures"
WORKLOAD_ORDER = ["chatbot", "code_generation", "long_conversation"]
CONFIG_ORDER = ["cpu_64", "gpu_1", "gpu_2"]


def load_json(path):
    return json.loads(path.read_text())


def as_num(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def read_csv(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            converted = {}
            for key, value in row.items():
                number = as_num(value)
                converted[key] = number if number is not None else value
            rows.append(converted)
    return rows


def timestamp_elapsed(rows):
    if not rows:
        return []
    base = datetime.fromisoformat(rows[0]["timestamp_utc"])
    elapsed = []
    for row in rows:
        try:
            elapsed.append((datetime.fromisoformat(row["timestamp_utc"]) - base).total_seconds())
        except (KeyError, TypeError, ValueError):
            elapsed.append(len(elapsed))
    return elapsed


def parse_ts(row):
    try:
        return datetime.fromisoformat(row["timestamp_utc"])
    except (KeyError, TypeError, ValueError):
        return None


def elapsed_from_base(rows, base_ts):
    elapsed = []
    for row in rows:
        ts = parse_ts(row)
        elapsed.append((ts - base_ts).total_seconds() if ts else len(elapsed))
    return elapsed


def gpu_activity_phases(cpu_rows, gpu_rows, cfg):
    timestamps = [parse_ts(row) for row in cpu_rows + gpu_rows]
    timestamps = [ts for ts in timestamps if ts is not None]
    if not timestamps:
        return None, [], "No timestamped telemetry rows were available."
    base_ts = min(timestamps)
    end_s = max((ts - base_ts).total_seconds() for ts in timestamps)

    active_times = []
    for row in gpu_rows:
        sm = row.get("sm_pct") or 0.0
        rx = row.get("rxpci_mbps") or 0.0
        tx = row.get("txpci_mbps") or 0.0
        ts = parse_ts(row)
        if ts and (sm > 0 or rx > 0 or tx > 0):
            active_times.append((ts - base_ts).total_seconds())

    if not active_times:
        phases = [
            (0.0, end_s, "#e6e6e6", "no active GPU samples"),
        ]
        note = "No nonzero GPU SM or PCIe samples were observed in this run."
        return base_ts, phases, note

    first_active = max(0.0, min(active_times))
    last_active = min(end_s, max(active_times) + 1.0)
    phases = []
    if first_active > 0:
        phases.append((0.0, first_active, "#9ecae1", "setup / ready check"))
    phases.append((first_active, max(first_active, last_active), "#fdae6b", "requests active"))
    if end_s > last_active:
        phases.append((last_active, end_s, "#bdbdbd", "cooldown / save"))

    transfer_note = "CPU<->GPU PCIe" if cfg == "gpu_1" else "aggregate PCIe incl. CPU<->GPU and NCCL/GPU<->GPU"
    note = (
        "Shaded phases: setup/ready check = client startup, tokenizer/random prompt generation, endpoint check; "
        "GPU memory is allocated but kernels are mostly idle. "
        f"requests active = initial single-prompt test plus main benchmark requests, with prefill/decode kernels and {transfer_note}. "
        "cooldown/save = benchmark result writing and monitor shutdown."
    )
    return base_ts, phases, note


def cpu_activity_phases(cpu_rows):
    if not cpu_rows:
        return []
    end_s = max((row.get("elapsed_s") or 0.0 for row in cpu_rows), default=0.0)
    cpu_values = [row.get("server_cpu_percent") or 0.0 for row in cpu_rows]
    max_cpu = max(cpu_values, default=0.0)
    threshold = max(50.0, max_cpu * 0.05)
    active_times = [
        row.get("elapsed_s") or 0.0
        for row, value in zip(cpu_rows, cpu_values)
        if value >= threshold
    ]
    if not active_times:
        return [(0.0, end_s, "#e6e6e6", "no active CPU samples")]

    first_active = max(0.0, min(active_times))
    last_active = min(end_s, max(active_times) + 1.0)
    phases = []
    if first_active > 0:
        phases.append((0.0, first_active, "#9ecae1", "setup / ready check"))
    phases.append((first_active, max(first_active, last_active), "#fdae6b", "requests active"))
    if end_s > last_active:
        phases.append((last_active, end_s, "#bdbdbd", "cooldown / save"))
    return phases


def apply_phase_shading(axes, phases):
    color_by_label = {
        "setup / ready check": "#9ecae1",
        "requests active": "#fdae6b",
        "cooldown / save": "#bdbdbd",
        "no active GPU samples": "#e6e6e6",
        "no active CPU samples": "#e6e6e6",
    }
    for ax in axes:
        for start, end, color, label in phases:
            if end > start:
                ax.axvspan(start, end, color=color_by_label.get(label, color), alpha=0.22, zorder=0)

def add_phase_legend(fig, phases, show_ranges=False):
    color_by_label = {
        "setup / ready check": "#9ecae1",
        "requests active": "#fdae6b",
        "cooldown / save": "#bdbdbd",
        "no active GPU samples": "#e6e6e6",
        "no active CPU samples": "#e6e6e6",
    }
    entries = []
    for start, end, color, label in phases:
        if end <= start:
            continue
        display = f"{label} ({start:.1f}-{end:.1f}s)" if show_ranges else label
        if display not in [entry[0] for entry in entries]:
            entries.append((display, color_by_label.get(label, color)))
    handles = [
        Patch(facecolor=color, edgecolor="none", alpha=0.5, label=label)
        for label, color in entries
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncols=len(handles),
            fontsize=8,
            frameon=True,
            title="phase shading",
            title_fontsize=8,
        )


def bench_payload(run_dir):
    path = run_dir / "results" / "bench_serve.json"
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except json.JSONDecodeError:
        return {}


def result_entries():
    entries = []
    for cfg in CONFIG_ORDER:
        for workload in WORKLOAD_ORDER:
            path = SMALL / cfg / workload / "result.json"
            if path.exists():
                result = load_json(path)
                entries.append((cfg, workload, path.parent, result, bench_payload(path.parent)))
    return entries


def get_bench_value(result, bench, *keys):
    for key in keys:
        value = bench.get(key)
        if value is not None:
            return value
    for key in keys:
        value = result.get(key)
        if value is not None:
            return value
    return None


def finish_axes(fig, axes, title, output_path, xlabel="seconds from benchmark monitor start", top=0.97):
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(xlabel, labelpad=14)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, top])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def cumulative_amount_gib(rows, rate_key):
    if not rows:
        return [], []
    x = timestamp_elapsed(rows)
    totals = [0.0]
    for i in range(1, len(rows)):
        dt = max(0.0, x[i] - x[i - 1])
        rate_mb_s = rows[i - 1].get(rate_key) or 0.0
        totals.append(totals[-1] + rate_mb_s * dt / 1024.0)
    return x, totals


def plot_cpu_run(cfg, workload, run_dir, result, bench):
    rows = read_csv(run_dir / "metrics" / "cpu_metrics.csv")
    if not rows:
        return None
    x = [row.get("elapsed_s") or 0.0 for row in rows]
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    phases = cpu_activity_phases(rows)
    apply_phase_shading(axes, phases)
    add_phase_legend(fig, phases, show_ranges=True)

    axes[0].plot(x, [row.get("server_cpu_percent") or 0.0 for row in rows], label="server process CPU %")
    axes[0].plot(x, [row.get("system_cpu_total_percent") or 0.0 for row in rows], label="system CPU %")
    axes[0].set_ylabel("CPU %")
    axes[0].legend(loc="upper right", ncols=2, fontsize=8)

    axes[1].plot(x, [(row.get("server_rss_bytes") or 0.0) / (1024**3) for row in rows], label="server RSS")
    axes[1].plot(x, [(row.get("system_mem_used_bytes") or 0.0) / (1024**3) for row in rows], label="system used")
    axes[1].set_ylabel("Memory GiB")
    axes[1].legend(loc="upper right", ncols=2, fontsize=8)

    axes[2].plot(x, [row.get("cpu_temp_max_c") or 0.0 for row in rows], label="max CPU temp C")
    axes[2].plot(x, [row.get("powercap_package_power_w") or 0.0 for row in rows], label="package power W")
    axes[2].set_ylabel("Temp C / W")
    axes[2].legend(loc="upper right", ncols=2, fontsize=8)

    axes[3].plot(x, [(row.get("system_mem_available_bytes") or 0.0) / (1024**3) for row in rows], label="system available")
    axes[3].set_ylabel("Avail GiB")
    axes[3].legend(loc="upper right", fontsize=8)

    req_s = get_bench_value(result, bench, "request_throughput", "requests_per_second") or 0.0
    tok_s = get_bench_value(result, bench, "total_token_throughput", "tokens_per_second") or 0.0
    out_s = get_bench_value(result, bench, "output_throughput", "output_tokens_per_second") or 0.0
    axes[4].bar(["req/s", "tok/s", "out tok/s"], [req_s, tok_s, out_s], color=["#4c78a8", "#f58518", "#54a24b"])
    axes[4].set_ylabel("Benchmark")
    axes[4].tick_params(axis="x", pad=10)
    for label in axes[4].get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    output = OUT / cfg / f"small_{cfg}_{workload}_resource_timeline.png"
    finish_axes(
        fig,
        axes,
        f"small {cfg} {workload}: CPU resource timeline",
        output,
        xlabel="benchmark metric",
        top=0.88,
    )
    return output


def plot_gpu_run(cfg, workload, run_dir, result, bench):
    cpu_rows = read_csv(run_dir / "metrics" / "cpu_metrics.csv")
    gpu_rows = read_csv(run_dir / "metrics" / "gpu_dmon.csv")
    if not gpu_rows:
        return None
    base_ts, phases, _phase_note = gpu_activity_phases(cpu_rows, gpu_rows, cfg)

    by_gpu = defaultdict(list)
    for row in gpu_rows:
        gpu = int(row.get("gpu") or 0)
        by_gpu[gpu].append(row)

    fig, axes = plt.subplots(6, 1, figsize=(15, 16), sharex=True)
    colors = {0: "#4c78a8", 1: "#f58518", 2: "#54a24b", 3: "#b279a2"}
    apply_phase_shading(axes, phases)
    add_phase_legend(fig, phases, show_ranges=True)

    for gpu, rows in sorted(by_gpu.items()):
        x = elapsed_from_base(rows, base_ts) if base_ts else timestamp_elapsed(rows)
        color = colors.get(gpu, None)
        axes[0].plot(x, [(row.get("fb_mb") or 0.0) / 1024.0 for row in rows], label=f"GPU{gpu} FB", color=color)
        axes[1].plot(x, [row.get("sm_pct") or 0.0 for row in rows], label=f"GPU{gpu} SM", color=color)
        axes[1].plot(x, [row.get("mem_pct") or 0.0 for row in rows], linestyle="--", label=f"GPU{gpu} mem util", color=color)
        axes[2].plot(x, [row.get("rxpci_mbps") or 0.0 for row in rows], label=f"GPU{gpu} RX", color=color)
        axes[2].plot(x, [row.get("txpci_mbps") or 0.0 for row in rows], linestyle="--", label=f"GPU{gpu} TX", color=color)
        rx_x, rx_total = cumulative_amount_gib(rows, "rxpci_mbps")
        tx_x, tx_total = cumulative_amount_gib(rows, "txpci_mbps")
        if base_ts:
            rx_x = x
            tx_x = x
        axes[3].plot(rx_x, rx_total, label=f"GPU{gpu} cumulative RX", color=color)
        axes[3].plot(tx_x, tx_total, linestyle="--", label=f"GPU{gpu} cumulative TX", color=color)
        axes[4].plot(x, [row.get("pwr_w") or 0.0 for row in rows], label=f"GPU{gpu} power W", color=color)
        axes[4].plot(x, [row.get("gtemp_c") or 0.0 for row in rows], linestyle="--", label=f"GPU{gpu} temp C", color=color)

    cpu_x = elapsed_from_base(cpu_rows, base_ts) if base_ts else [row.get("elapsed_s") or 0.0 for row in cpu_rows]
    axes[5].plot(cpu_x, [row.get("server_cpu_percent") or 0.0 for row in cpu_rows], label="server CPU %", color="#4c78a8")
    axes[5].plot(cpu_x, [(row.get("server_rss_bytes") or 0.0) / (1024**3) for row in cpu_rows], label="server RSS GiB", color="#f58518")

    axes[0].set_ylabel("GPU mem GiB")
    axes[1].set_ylabel("Util %")
    if cfg == "gpu_1":
        axes[2].set_ylabel("CPU<->GPU PCIe MB/s")
        axes[3].set_ylabel("CPU<->GPU PCIe GiB")
        transfer_note = "PCIe RX/TX is CPU<->GPU for the active single-GPU run"
    else:
        axes[2].set_ylabel("Aggregate PCIe MB/s")
        axes[3].set_ylabel("Aggregate PCIe GiB")
        transfer_note = "PCIe RX/TX is aggregate CPU<->GPU plus GPU<->GPU/NCCL traffic; dmon does not decompose it"
    axes[4].set_ylabel("Power W / Temp C")
    axes[5].set_ylabel("Host")

    for ax in axes:
        ax.legend(loc="upper right", ncols=2, fontsize=8)
        ax.grid(True, alpha=0.25)

    req_s = get_bench_value(result, bench, "request_throughput", "requests_per_second")
    tok_s = get_bench_value(result, bench, "total_token_throughput", "tokens_per_second")
    out_s = get_bench_value(result, bench, "output_throughput", "output_tokens_per_second")
    title = f"small {cfg} {workload}: memory, utilization, power, and transfer"
    axes[-1].set_xlabel("seconds from monitor start")
    output = OUT / cfg / f"small_{cfg}_{workload}_transfer_memory_timeline.png"
    finish_axes(fig, axes, title, output, top=0.88)
    return output


def plot_overview(entries):
    labels = []
    req = []
    tok = []
    out = []
    ttft = []
    max_mem = []
    colors = []
    config_colors = {"cpu_64": "#4c78a8", "gpu_1": "#f58518", "gpu_2": "#54a24b"}
    for cfg, workload, _run_dir, result, bench in entries:
        labels.append(f"{cfg}\n{workload.replace('_', ' ')}")
        req.append(get_bench_value(result, bench, "request_throughput", "requests_per_second") or 0.0)
        tok.append(get_bench_value(result, bench, "total_token_throughput", "tokens_per_second") or 0.0)
        out.append(get_bench_value(result, bench, "output_throughput", "output_tokens_per_second") or 0.0)
        ttft.append(get_bench_value(result, bench, "mean_ttft_ms") or 0.0)
        if cfg == "cpu_64":
            max_mem.append(result.get("max_server_rss_gib") or 0.0)
        else:
            max_mem.append(result.get("max_gpu_fb_gib") or 0.0)
        colors.append(config_colors[cfg])

    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    x = list(range(len(labels)))
    axes[0].bar(x, req, color=colors)
    axes[0].set_ylabel("Request/s")
    axes[1].bar(x, tok, color=colors)
    axes[1].set_ylabel("Total tok/s")
    axes[2].bar(x, out, color=colors)
    axes[2].set_ylabel("Output tok/s")
    axes[3].bar(x, max_mem, color=colors)
    axes[3].set_ylabel("Max memory GiB")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=45, ha="right")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("small model overview: throughput and peak memory", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    output = OUT / "overview" / "small_performance_memory_overview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_runtime_comparison(entries):
    durations = {(cfg, workload): result.get("duration_s") or 0.0 for cfg, workload, _run_dir, result, _bench in entries}
    fig, ax = plt.subplots(figsize=(12, 7))
    x = list(range(len(WORKLOAD_ORDER)))
    width = 0.24
    offsets = {"cpu_64": -width, "gpu_1": 0.0, "gpu_2": width}
    colors = {"cpu_64": "#4c78a8", "gpu_1": "#f58518", "gpu_2": "#54a24b"}

    for cfg in CONFIG_ORDER:
        ys = [durations.get((cfg, workload), 0.0) for workload in WORKLOAD_ORDER]
        positions = [item + offsets[cfg] for item in x]
        bars = ax.bar(positions, ys, width=width, label=cfg, color=colors[cfg])
        for bar, value in zip(bars, ys):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(value, 0.1) * 1.05,
                f"{value:.1f}s",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_yscale("log")
    ax.set_ylabel("Runtime seconds, log scale")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", " ") for name in WORKLOAD_ORDER])
    ax.set_title("small model runtime comparison by workload")
    ax.legend(title="config")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = OUT / "overview" / "small_runtime_comparison_by_workload.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def write_summary(entries, generated):
    summary_path = OUT / "small_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        fieldnames = [
            "config",
            "workload",
            "status",
            "duration_s",
            "request_throughput",
            "total_token_throughput",
            "output_throughput",
            "mean_ttft_ms",
            "mean_tpot_ms",
            "max_server_rss_gib",
            "max_gpu_fb_gib",
            "max_gpu_sm_pct",
            "max_gpu_power_w",
            "max_gpu_temp_c",
            "max_pcie_rx_mbps",
            "max_pcie_tx_mbps",
            "figure",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cfg, workload, _run_dir, result, bench in entries:
            writer.writerow(
                {
                    "config": cfg,
                    "workload": workload,
                    "status": result.get("status"),
                    "duration_s": result.get("duration_s"),
                    "request_throughput": get_bench_value(result, bench, "request_throughput", "requests_per_second"),
                    "total_token_throughput": get_bench_value(result, bench, "total_token_throughput", "tokens_per_second"),
                    "output_throughput": get_bench_value(result, bench, "output_throughput", "output_tokens_per_second"),
                    "mean_ttft_ms": get_bench_value(result, bench, "mean_ttft_ms"),
                    "mean_tpot_ms": get_bench_value(result, bench, "mean_tpot_ms"),
                    "max_server_rss_gib": result.get("max_server_rss_gib"),
                    "max_gpu_fb_gib": result.get("max_gpu_fb_gib"),
                    "max_gpu_sm_pct": result.get("max_gpu_sm_pct"),
                    "max_gpu_power_w": result.get("max_gpu_power_w"),
                    "max_gpu_temp_c": result.get("max_gpu_temp_c"),
                    "max_pcie_rx_mbps": result.get("max_pcie_rx_mbps"),
                    "max_pcie_tx_mbps": result.get("max_pcie_tx_mbps"),
                    "figure": generated.get((cfg, workload), ""),
                }
            )

    index_path = OUT / "README.md"
    lines = [
        "# Small Model Figures",
        "",
        "Generated from the completed small-model run under `real_runs/small`.",
        "",
        "For `gpu_1`, PCIe RX/TX is interpreted as CPU<->GPU transfer. For `gpu_2`, `nvidia-smi dmon` reports aggregate PCIe RX/TX per GPU, so the plot shows CPU<->GPU plus GPU<->GPU/NCCL traffic together; it cannot split those paths after the fact.",
        "",
        "CPU and GPU timeline figures include shaded phases with time ranges. CPU active ranges are inferred from server CPU activity; GPU active ranges are inferred from first and last nonzero GPU SM or PCIe samples.",
        "",
        f"- Summary CSV: `{summary_path.relative_to(SMALL)}`",
        "- Overview: `overview/small_performance_memory_overview.png`",
        "- Runtime comparison: `overview/small_runtime_comparison_by_workload.png`",
        "",
        "## Per-Run Figures",
        "",
    ]
    for cfg in CONFIG_ORDER:
        lines.append(f"### {cfg}")
        for workload in WORKLOAD_ORDER:
            path = generated.get((cfg, workload))
            if path:
                lines.append(f"- {workload}: `{path.relative_to(OUT)}`")
        lines.append("")
    index_path.write_text("\n".join(lines) + "\n")
    return summary_path, index_path


def main():
    entries = result_entries()
    generated = {}
    for cfg, workload, run_dir, result, bench in entries:
        if cfg == "cpu_64":
            output = plot_cpu_run(cfg, workload, run_dir, result, bench)
        else:
            output = plot_gpu_run(cfg, workload, run_dir, result, bench)
        if output:
            generated[(cfg, workload)] = output
    overview = plot_overview(entries)
    runtime = plot_runtime_comparison(entries)
    summary_path, index_path = write_summary(entries, generated)
    print(f"generated {len(generated)} per-run figures")
    print(f"overview: {overview}")
    print(f"runtime: {runtime}")
    print(f"summary: {summary_path}")
    print(f"index: {index_path}")


if __name__ == "__main__":
    main()
