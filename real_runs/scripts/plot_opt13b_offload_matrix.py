#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
MATRIX_DIR = ROOT / "real_runs" / "a100_opt13b_offload_matrix"
TRACE_MANIFEST_PATH = (
    ROOT / "real_runs" / "datasets" / "opt13b_offload" / "trace_manifest.json"
)
OUTPUT_DIR = MATRIX_DIR / "figures"
GIB = float(1 << 30)
CPU_OFFLOAD_GIB = 40.0

CPU_COLOR = "#8c510a"
GPU_COLOR = "#4c78a8"
KV5_COLOR = "#8c510a"
KV10_COLOR = "#4c78a8"


@dataclass(frozen=True)
class CellSpec:
    budget_key: str
    budget_gib: int
    utilization: float
    application: str
    application_label: str
    workload: str
    trace_key: str

    @property
    def base_dir(self) -> Path:
        return (
            MATRIX_DIR
            / self.budget_key
            / self.application
            / "opt13b"
            / "gpu_a100"
            / self.workload
        )

    @property
    def cell_label(self) -> str:
        return f"+{self.budget_gib} GiB {self.application_label}"

    @property
    def budget_label(self) -> str:
        return f"+{self.budget_gib} GiB ({100 * self.utilization:.3f}%)"


CELLS = [
    CellSpec("kv5", 5, 0.75125, "chat", "Chat", "offload_chat_30gib", "chat"),
    CellSpec("kv5", 5, 0.75125, "code", "Code", "offload_code_30gib", "code"),
    CellSpec("kv10", 10, 0.87625, "chat", "Chat", "offload_chat_30gib", "chat"),
    CellSpec("kv10", 10, 0.87625, "code", "Code", "offload_code_30gib", "code"),
]


def add_unified_style() -> None:
    figures_root = Path("/mnt/near-mem/rec/figures")
    if figures_root.exists() and str(figures_root) not in sys.path:
        sys.path.insert(0, str(figures_root))
    try:
        from unified_style import apply_unified_style

        apply_unified_style(plt, size=16)
    except Exception:
        plt.rcParams.update(
            {
                "font.size": 16,
                "font.weight": "bold",
                "axes.labelweight": "bold",
                "axes.titleweight": "bold",
            }
        )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value in (None, ""):
                    row[key] = None
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def numeric_series(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    points = [
        (float(row["elapsed_s"]), float(row[key]))
        for row in rows
        if isinstance(row.get("elapsed_s"), (int, float))
        and isinstance(row.get(key), (int, float))
    ]
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    points.sort()
    return (
        np.asarray([point[0] for point in points], dtype=float),
        np.asarray([point[1] for point in points], dtype=float),
    )


def step_counter_series(
    rows: list[dict[str, Any]], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elapsed = []
    cumulative = []
    rates = []
    previous_elapsed = 0.0
    previous_value = 0.0
    current_value = 0.0
    for row in sorted(rows, key=lambda item: float(item.get("elapsed_s") or 0)):
        current_elapsed = float(row.get("elapsed_s") or 0)
        value = row.get(key)
        if isinstance(value, (int, float)):
            current_value = float(value)
        delta_time = current_elapsed - previous_elapsed
        delta_value = max(0.0, current_value - previous_value)
        elapsed.append(current_elapsed)
        cumulative.append(current_value)
        rates.append(delta_value / delta_time / GIB if delta_time > 0 else 0.0)
        previous_elapsed = current_elapsed
        previous_value = current_value
    return (
        np.asarray(elapsed, dtype=float),
        np.asarray(cumulative, dtype=float),
        np.asarray(rates, dtype=float),
    )


def counter_activity(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    elapsed, cumulative, rates = step_counter_series(rows, key)
    active_times = elapsed[rates > 0]
    bursts = 0
    previous = None
    for current in active_times:
        if previous is None or current - previous > 1.6:
            bursts += 1
        previous = current
    total_bytes = float(cumulative.max()) if cumulative.size else 0.0
    return {
        "bytes": total_bytes,
        "gib": total_bytes / GIB,
        "active_samples": int(active_times.size),
        "bursts": bursts,
        "first_active_s": float(active_times[0]) if active_times.size else None,
        "last_active_s": float(active_times[-1]) if active_times.size else None,
    }


def parse_server_memory(server_log: str) -> dict[str, Any]:
    patterns = {
        "model_weight_gib": r"Model loading took ([0-9.]+) GiB",
        "gpu_kv_pool_gib": r"Available KV cache memory: ([0-9.]+) GiB",
        "gpu_kv_pool_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
    }
    values: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, server_log)
        if not match:
            raise ValueError(f"server log missing {key}")
        raw = match.group(1).replace(",", "")
        values[key] = int(raw) if key.endswith("tokens") else float(raw)
    return values


def load_cell(spec: CellSpec, trace_manifest: dict[str, Any]) -> dict[str, Any]:
    base = spec.base_dir
    result = read_json(base / "result.json")
    bench = read_json(base / "results" / "bench_serve.json")
    cpu_rows = read_csv(base / "metrics" / "cpu_metrics.csv")
    gpu_rows = read_csv(base / "metrics" / "gpu_dmon.csv")
    engine_rows = read_csv(base / "metrics" / "vllm_metrics.csv")
    server_log_path = base.parent / "server.log"
    server_info_path = base.parent / "server_info.json"
    server_log = server_log_path.read_text(errors="replace")
    server_info = read_json(server_info_path)
    run_manifest = read_json(MATRIX_DIR / spec.budget_key / spec.application / "run_manifest.json")
    trace = trace_manifest["traces"][spec.trace_key]
    memory = parse_server_memory(server_log)

    transfer_config = None
    command = server_info["server_command"]
    if "--kv-transfer-config" in command:
        transfer_config = json.loads(command[command.index("--kv-transfer-config") + 1])
    checks = {
        "result passed": result.get("status") == "passed",
        "22 successful requests": bench.get("completed") == 22 and bench.get("failed") == 0,
        "expected model": result.get("model") == "facebook/opt-13b",
        "expected output tokens": bench.get("total_output_tokens") == 33792,
        "expected trace": result.get("dataset_path") == trace["path"],
        "GPU utilization": run_manifest.get("gpu_memory_utilization")
        == f"{spec.utilization:.5f}",
        "full decode offload": run_manifest.get("gpu_kv_offload_prompt_only") is False,
        "connector command": transfer_config is not None
        and transfer_config.get("kv_connector") == "OffloadingConnector",
        "connector direction": transfer_config is not None
        and transfer_config.get("kv_role") == "kv_both",
        "connector decode blocks": transfer_config is not None
        and transfer_config.get("kv_connector_extra_config", {}).get(
            "offload_prompt_only"
        )
        is False,
        "40GiB CPU buffer": transfer_config is not None
        and transfer_config.get("kv_connector_extra_config", {}).get(
            "cpu_bytes_to_use"
        )
        == 40 * (1 << 30),
        "GPU to CPU stores": result.get("kv_offload_gpu_to_cpu_gib", 0) > 0,
        "CPU to GPU reloads": result.get("kv_offload_cpu_to_gpu_gib", 0) > 0,
        "load operations": result.get("kv_offload_load_operations", 0) > 0,
        "hardware PCIe TX": result.get("pcie_tx_gib", 0) > 0,
        "hardware PCIe RX": result.get("pcie_rx_gib", 0) > 0,
        "CPU timeline": bool(cpu_rows),
        "GPU timeline": bool(gpu_rows),
        "engine timeline": bool(engine_rows),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError(f"{spec.cell_label} failed gates: {', '.join(failures)}")

    load_activity = counter_activity(engine_rows, "kv_offload_load_bytes_total")
    store_activity = counter_activity(engine_rows, "kv_offload_store_bytes_total")
    duration = float(result["duration_s"])
    return {
        "spec": spec,
        "base": base,
        "result": result,
        "bench": bench,
        "cpu_rows": cpu_rows,
        "gpu_rows": gpu_rows,
        "engine_rows": engine_rows,
        "server_log": server_log,
        "server_info": server_info,
        "run_manifest": run_manifest,
        "trace": trace,
        "memory": memory,
        "load_activity": load_activity,
        "store_activity": store_activity,
        "load_operations_per_s": float(result["kv_offload_load_operations"])
        / duration,
        "seconds_per_load_burst": duration / load_activity["bursts"],
    }


def interpolate(
    rows: list[dict[str, Any]],
    key: str,
    grid: np.ndarray,
    left_value: float | None = None,
) -> np.ndarray:
    x, y = numeric_series(rows, key)
    if not x.size:
        return np.zeros_like(grid)
    return np.interp(
        grid,
        x,
        y,
        left=y[0] if left_value is None else left_value,
        right=y[-1],
    )


def forward_fill_counter(
    rows: list[dict[str, Any]], key: str, grid: np.ndarray
) -> np.ndarray:
    x, cumulative, _rates = step_counter_series(rows, key)
    if not x.size:
        return np.zeros_like(grid)
    indices = np.searchsorted(x, grid, side="right") - 1
    values = np.zeros_like(grid)
    valid = indices >= 0
    values[valid] = cumulative[indices[valid]]
    return values


def resample_cell(cell: dict[str, Any]) -> list[dict[str, Any]]:
    result = cell["result"]
    max_elapsed = max(
        max((float(row.get("elapsed_s") or 0) for row in cell[key]), default=0)
        for key in ("cpu_rows", "gpu_rows", "engine_rows")
    )
    grid = np.arange(0.0, math.ceil(max_elapsed) + 1.0, 1.0)
    gpu_fb = interpolate(cell["gpu_rows"], "fb_mb", grid) / 1024
    server_rss = interpolate(cell["cpu_rows"], "server_rss_bytes", grid) / GIB
    system_memory = interpolate(cell["cpu_rows"], "system_mem_used_bytes", grid) / GIB
    gpu_kv_fraction = interpolate(
        cell["engine_rows"], "gpu_kv_cache_usage_fraction", grid
    )
    cpu_kv_fraction = interpolate(
        cell["engine_rows"],
        "cpu_kv_cache_usage_fraction",
        grid,
        left_value=0.0,
    )
    pcie_tx = interpolate(cell["gpu_rows"], "txpci_mbps", grid) / 1024
    pcie_rx = interpolate(cell["gpu_rows"], "rxpci_mbps", grid) / 1024
    sm = interpolate(cell["gpu_rows"], "sm_pct", grid)
    store_counter = forward_fill_counter(
        cell["engine_rows"], "kv_offload_store_bytes_total", grid
    )
    load_counter = forward_fill_counter(
        cell["engine_rows"], "kv_offload_load_bytes_total", grid
    )
    store_rate = np.diff(store_counter, prepend=0) / GIB
    load_rate = np.diff(load_counter, prepend=0) / GIB
    rows = []
    for index, elapsed in enumerate(grid):
        rows.append(
            {
                "budget_gib": cell["spec"].budget_gib,
                "gpu_memory_utilization": cell["spec"].utilization,
                "application": cell["spec"].application,
                "elapsed_s": elapsed,
                "gpu_fb_gib": gpu_fb[index],
                "server_rss_gib": server_rss[index],
                "system_memory_used_gib": system_memory[index],
                "gpu_kv_usage_gib": gpu_kv_fraction[index]
                * cell["memory"]["gpu_kv_pool_gib"],
                "cpu_kv_usage_gib": cpu_kv_fraction[index] * CPU_OFFLOAD_GIB,
                "pcie_gpu_to_cpu_gib_s": pcie_tx[index],
                "pcie_cpu_to_gpu_gib_s": pcie_rx[index],
                "connector_gpu_to_cpu_gib_s": store_rate[index],
                "connector_cpu_to_gpu_gib_s": load_rate[index],
                "connector_gpu_to_cpu_cumulative_gib": store_counter[index] / GIB,
                "connector_cpu_to_gpu_cumulative_gib": load_counter[index] / GIB,
                "gpu_sm_pct": sm[index],
                "output_throughput_tokens_s": result["output_tokens_per_second"],
            }
        )
    return rows


def style_axis(axis, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, linestyle="--", alpha=0.30, linewidth=0.9, zorder=0)
    axis.tick_params(axis="both", width=1.8, length=6, direction="out", labelsize=11)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.9)
    axis.spines["bottom"].set_linewidth(1.9)


def plot_timelines(cells: list[dict[str, Any]]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(4, 3, figsize=(16.5, 15.0), dpi=300)
    fig.patch.set_facecolor("white")
    for row_index, cell in enumerate(cells):
        spec: CellSpec = cell["spec"]
        memory_axis, occupancy_axis, traffic_axis = axes[row_index]
        for axis in axes[row_index]:
            axis.set_facecolor("white")

        engine_x, gpu_kv_fraction = numeric_series(
            cell["engine_rows"], "gpu_kv_cache_usage_fraction"
        )
        cpu_engine_x, cpu_kv_fraction = numeric_series(
            cell["engine_rows"], "cpu_kv_cache_usage_fraction"
        )
        if not cpu_kv_fraction.size:
            cpu_engine_x = engine_x
            cpu_kv_fraction = np.zeros_like(engine_x)
        gpu_kv_gib = gpu_kv_fraction * cell["memory"]["gpu_kv_pool_gib"]
        cpu_kv_gib = cpu_kv_fraction * CPU_OFFLOAD_GIB

        memory_axis.plot(
            cpu_engine_x,
            cpu_kv_gib,
            color=CPU_COLOR,
            linewidth=2.0,
            label="CPU",
        )
        memory_axis.plot(
            engine_x,
            gpu_kv_gib,
            color=GPU_COLOR,
            linewidth=2.0,
            label="GPU",
        )
        peak_kv_gib = max(
            float(cpu_kv_gib.max()) if cpu_kv_gib.size else 0.0,
            float(gpu_kv_gib.max()) if gpu_kv_gib.size else 0.0,
        )
        memory_axis.set_ylabel("KV cache memory (GiB)", fontsize=12, fontweight="bold")
        memory_axis.set_ylim(0, max(6.0, peak_kv_gib * 1.15))
        style_axis(memory_axis)

        occupancy_axis.plot(
            engine_x,
            100 * gpu_kv_fraction,
            color=GPU_COLOR,
            linewidth=2.0,
            label="GPU",
        )
        occupancy_axis.plot(
            cpu_engine_x,
            100 * cpu_kv_fraction,
            color=CPU_COLOR,
            linewidth=2.0,
            label="CPU",
        )
        occupancy_axis.set_ylabel("KV cache usage (%)", fontsize=12, fontweight="bold")
        occupancy_axis.set_ylim(0, 105)
        style_axis(occupancy_axis)

        tx_x, tx = numeric_series(cell["gpu_rows"], "txpci_mbps")
        rx_x, rx = numeric_series(cell["gpu_rows"], "rxpci_mbps")
        store_x, _store_cumulative, store_rate = step_counter_series(
            cell["engine_rows"], "kv_offload_store_bytes_total"
        )
        load_x, _load_cumulative, load_rate = step_counter_series(
            cell["engine_rows"], "kv_offload_load_bytes_total"
        )
        traffic_axis.plot(
            tx_x,
            tx / 1024,
            color=CPU_COLOR,
            linewidth=1.4,
            alpha=0.35,
            label="CPU",
        )
        traffic_axis.plot(
            rx_x,
            rx / 1024,
            color=GPU_COLOR,
            linewidth=1.4,
            alpha=0.35,
            label="GPU",
        )
        traffic_axis.step(
            store_x,
            store_rate,
            where="post",
            color=CPU_COLOR,
            linewidth=2.0,
            label="CPU",
        )
        traffic_axis.step(
            load_x,
            load_rate,
            where="post",
            color=GPU_COLOR,
            linewidth=2.0,
            label="GPU",
        )
        traffic_axis.set_ylabel("KV cache transfer (GiB/s)", fontsize=12, fontweight="bold")
        style_axis(traffic_axis)

        max_elapsed = max(
            max((float(row.get("elapsed_s") or 0) for row in cell[key]), default=0)
            for key in ("cpu_rows", "gpu_rows", "engine_rows")
        )
        for axis in axes[row_index]:
            axis.set_xlim(0, max_elapsed)
            if row_index == len(cells) - 1:
                axis.set_xlabel("Benchmark time (s)", fontsize=12, fontweight="bold")
        memory_axis.text(
            0.02,
            0.94,
            f"{spec.application_label} | {spec.budget_label}\nActual GPU KV pool: {cell['memory']['gpu_kv_pool_gib']:.2f} GiB",
            transform=memory_axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        traffic_axis.text(
            0.98,
            0.94,
            f"Stores {cell['result']['kv_offload_gpu_to_cpu_gib']:.1f} GiB\nLoads {cell['result']['kv_offload_cpu_to_gpu_gib']:.1f} GiB / {cell['result']['kv_offload_load_operations']:.0f} ops",
            transform=traffic_axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

    axes[0, 0].set_title("(a) Memory usage for host & GPU", fontsize=15, fontweight="bold")
    axes[0, 1].set_title("(b) Live KV occupancy", fontsize=15, fontweight="bold")
    axes[0, 2].set_title("(c) KV cache transfer", fontsize=15, fontweight="bold")
    handles = [
        Patch(facecolor=CPU_COLOR, edgecolor="black", label="CPU"),
        Patch(facecolor=GPU_COLOR, edgecolor="black", label="GPU"),
    ]
    fig.legend(
        handles=handles,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        fontsize=11,
        columnspacing=2.0,
    )
    fig.text(
        0.5,
        0.005,
        "Transfer panel: CPU color = GPU→CPU; GPU color = CPU→GPU. Faint lines are aggregate PCIe; solid steps are KV-specific.",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.965), h_pad=1.2, w_pad=1.0)
    png_path = OUTPUT_DIR / "opt13b_offload_timelines.png"
    pdf_path = OUTPUT_DIR / "opt13b_offload_timelines.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_summary(cells: list[dict[str, Any]]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(16.3, 5.3), dpi=300)
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")

    throughput_axis, transfer_axis, revisit_axis = axes
    applications = ["Chat", "Code"]
    centers = np.arange(2, dtype=float)
    width = 0.32
    budget_groups = [(5, KV5_COLOR, None), (10, KV10_COLOR, "xx")]
    for offset_index, (budget, color, hatch) in enumerate(budget_groups):
        values = [
            next(
                cell["result"]["output_tokens_per_second"]
                for cell in cells
                if cell["spec"].budget_gib == budget
                and cell["spec"].application_label == application
            )
            for application in applications
        ]
        positions = centers + (offset_index - 0.5) * width
        bars = throughput_axis.bar(
            positions,
            values,
            width=width,
            color=color,
            edgecolor="black",
            hatch=hatch,
            linewidth=0.9,
            zorder=3,
            label=f"+{budget} GiB",
        )
        for bar, value in zip(bars, values):
            throughput_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 8,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
    throughput_axis.set_xticks(centers)
    throughput_axis.set_xticklabels(applications, fontsize=13, fontweight="bold")
    throughput_axis.set_ylim(0, 440)
    throughput_axis.set_ylabel("Output throughput (tokens/s)", fontsize=14, fontweight="bold")
    throughput_axis.set_title("(a) GPU serving performance", fontsize=16, fontweight="bold")
    throughput_axis.legend(frameon=False, fontsize=11, loc="upper left")
    style_axis(throughput_axis)

    cell_centers = np.arange(len(cells), dtype=float)
    store_values = [cell["result"]["kv_offload_gpu_to_cpu_gib"] for cell in cells]
    load_values = [cell["result"]["kv_offload_cpu_to_gpu_gib"] for cell in cells]
    transfer_axis.bar(
        cell_centers - width / 2,
        store_values,
        width=width,
        color=CPU_COLOR,
        edgecolor="black",
        linewidth=0.9,
        label="GPU→CPU store",
        zorder=3,
    )
    transfer_axis.bar(
        cell_centers + width / 2,
        load_values,
        width=width,
        color=GPU_COLOR,
        edgecolor="black",
        hatch="xx",
        linewidth=0.9,
        label="CPU→GPU load",
        zorder=3,
    )
    transfer_axis.set_xticks(cell_centers)
    transfer_axis.set_xticklabels(
        [f"+{cell['spec'].budget_gib}\n{cell['spec'].application_label}" for cell in cells],
        fontsize=10,
        fontweight="bold",
    )
    transfer_axis.set_ylabel("KV-specific transfer (GiB)", fontsize=14, fontweight="bold")
    transfer_axis.set_title("(b) Directional KV traffic", fontsize=16, fontweight="bold")
    transfer_axis.legend(frameon=False, fontsize=10, loc="upper left")
    style_axis(transfer_axis)

    load_operations = [cell["result"]["kv_offload_load_operations"] for cell in cells]
    bars = revisit_axis.bar(
        cell_centers,
        load_operations,
        width=0.58,
        color=GPU_COLOR,
        edgecolor="black",
        hatch="//",
        linewidth=0.9,
        zorder=3,
    )
    for bar, cell in zip(bars, cells):
        operations = cell["result"]["kv_offload_load_operations"]
        revisit_axis.text(
            bar.get_x() + bar.get_width() / 2,
            operations + 2,
            f"{operations:.0f} ops\n{cell['load_activity']['bursts']} bursts",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    revisit_axis.set_xticks(cell_centers)
    revisit_axis.set_xticklabels(
        [f"+{cell['spec'].budget_gib}\n{cell['spec'].application_label}" for cell in cells],
        fontsize=10,
        fontweight="bold",
    )
    revisit_axis.set_ylim(0, max(load_operations) * 1.30)
    revisit_axis.set_ylabel("Connector CPU→GPU load operations", fontsize=14, fontweight="bold")
    revisit_axis.set_title("(c) Offloaded-KV revisits", fontsize=16, fontweight="bold")
    style_axis(revisit_axis)

    fig.suptitle(
        "OPT-13B offload matrix: 29.994 GiB logical KV per application",
        fontsize=19,
        fontweight="bold",
        y=1.06,
    )
    fig.text(
        0.5,
        -0.01,
        "+5 GiB = 75.125% GPU limit; +10 GiB = 87.625%; +20 GiB omitted because it requires 112.625% of A100 40GB.",
        ha="center",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98), w_pad=1.3)
    png_path = OUTPUT_DIR / "opt13b_offload_summary.png"
    pdf_path = OUTPUT_DIR / "opt13b_offload_summary.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def write_outputs(cells: list[dict[str, Any]], trace_manifest: dict[str, Any]) -> None:
    summary_rows = []
    timeline_rows = []
    for cell in cells:
        spec: CellSpec = cell["spec"]
        result = cell["result"]
        summary_rows.append(
            {
                "budget_label": spec.budget_label,
                "budget_gib": spec.budget_gib,
                "gpu_memory_utilization": spec.utilization,
                "application": spec.application,
                "logical_kv_gib": cell["trace"]["logical_kv_gib"],
                "actual_gpu_kv_pool_gib": cell["memory"]["gpu_kv_pool_gib"],
                "successful_requests": cell["bench"]["completed"],
                "duration_s": result["duration_s"],
                "output_throughput_tokens_s": result["output_tokens_per_second"],
                "mean_ttft_ms": result["mean_ttft_ms"],
                "mean_tpot_ms": result["mean_tpot_ms"],
                "preemptions": result["preemptions_during_run"],
                "peak_waiting_requests": result["max_requests_waiting"],
                "peak_gpu_kv_usage_pct": result["max_gpu_kv_cache_usage_pct"],
                "peak_cpu_kv_usage_pct": result["max_cpu_kv_cache_usage_pct"],
                "kv_gpu_to_cpu_gib": result["kv_offload_gpu_to_cpu_gib"],
                "kv_cpu_to_gpu_gib": result["kv_offload_cpu_to_gpu_gib"],
                "kv_store_operations": result["kv_offload_store_operations"],
                "kv_load_operations": result["kv_offload_load_operations"],
                "kv_load_operations_per_s": cell["load_operations_per_s"],
                "observed_load_bursts_1s": cell["load_activity"]["bursts"],
                "seconds_per_observed_load_burst": cell["seconds_per_load_burst"],
                "pcie_gpu_tx_gib": result["pcie_tx_gib"],
                "pcie_gpu_rx_gib": result["pcie_rx_gib"],
                "peak_pcie_bidirectional_gib_s": result["max_pcie_rx_tx_gib_s"],
                "peak_server_rss_gib": result["max_server_rss_gib"],
                "peak_gpu_fb_gib": result["max_gpu_fb_gib"],
                "average_gpu_sm_pct": result["avg_gpu_sm_pct"],
                "gpu_energy_j": result["gpu_energy_j"],
            }
        )
        timeline_rows.extend(resample_cell(cell))

    summary_path = MATRIX_DIR / "matrix_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    timeline_path = MATRIX_DIR / "combined_timeline_1s.csv"
    with timeline_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timeline_rows[0]))
        writer.writeheader()
        writer.writerows(timeline_rows)

    chat_kv5 = next(cell for cell in cells if cell["spec"].budget_gib == 5 and cell["spec"].application == "chat")
    chat_kv10 = next(cell for cell in cells if cell["spec"].budget_gib == 10 and cell["spec"].application == "chat")
    code_kv5 = next(cell for cell in cells if cell["spec"].budget_gib == 5 and cell["spec"].application == "code")
    code_kv10 = next(cell for cell in cells if cell["spec"].budget_gib == 10 and cell["spec"].application == "code")
    machine_summary = {
        "model": trace_manifest["model"],
        "model_revision": trace_manifest["model_revision"],
        "precision": trace_manifest["precision"],
        "kv_bytes_per_token": trace_manifest["kv_bytes_per_token"],
        "logical_kv_gib_per_application": trace_manifest["traces"]["chat"]["logical_kv_gib"],
        "budget_derivation": trace_manifest["gpu_budgets"],
        "omitted_budget": trace_manifest["omitted_budget"],
        "direction_semantics": {
            "nvidia_pcie_tx": "GPU to host/CPU, aggregate PCIe, 20ms window sampled once per second",
            "nvidia_pcie_rx": "host/CPU to GPU, aggregate PCIe, 20ms window sampled once per second",
            "connector_store": "GPU to CPU KV-specific cumulative bytes/operations",
            "connector_load": "CPU to GPU KV-specific cumulative bytes/operations",
        },
        "cells": summary_rows,
        "comparisons": {
            "chat_throughput_gain_kv10_vs_kv5_pct": 100
            * (
                chat_kv10["result"]["output_tokens_per_second"]
                / chat_kv5["result"]["output_tokens_per_second"]
                - 1
            ),
            "code_throughput_gain_kv10_vs_kv5_pct": 100
            * (
                code_kv10["result"]["output_tokens_per_second"]
                / code_kv5["result"]["output_tokens_per_second"]
                - 1
            ),
            "chat_load_operation_reduction_kv10_vs_kv5_pct": 100
            * (
                1
                - chat_kv10["result"]["kv_offload_load_operations"]
                / chat_kv5["result"]["kv_offload_load_operations"]
            ),
            "code_load_operation_reduction_kv10_vs_kv5_pct": 100
            * (
                1
                - code_kv10["result"]["kv_offload_load_operations"]
                / code_kv5["result"]["kv_offload_load_operations"]
            ),
        },
        "all_gates_passed": True,
    }
    (MATRIX_DIR / "matrix_summary.json").write_text(
        json.dumps(machine_summary, indent=2) + "\n"
    )

    report_lines = [
        "# OPT-13B directional KV-offload matrix",
        "",
        "Date: 2026-07-15",
        "",
        "## Design",
        "",
        "- Model: `facebook/opt-13b` at revision `e515202d...`, FP16, one A100 40GB.",
        "- Applications: ShareGPT chat and InstructCoder code editing.",
        "- Each application: 22 prompts x (251 input + 1,536 output tokens), exactly 29.994 GiB logical KV.",
        "- Native vLLM 0.25.1 `OffloadingConnector` with `offload_prompt_only=false` and a 40 GiB CPU buffer.",
        "- Each cell starts and stops a fresh server; no CPU/GPU cache state crosses cells.",
        "",
        "## GPU budget calculation",
        "",
        "Measured OPT-13B model memory was 23.94 GiB and measured non-KV runtime overhead was 1.11 GiB, for 25.05 GiB before the GPU KV pool.",
        "",
        "| Requested GPU KV headroom | Executor target | A100 percentage | Feasible |",
        "| ---: | ---: | ---: | :---: |",
        "| +5 GiB | 30.05 GiB | 75.125% | Yes |",
        "| +10 GiB | 35.05 GiB | 87.625% | Yes |",
        "| +20 GiB | 45.05 GiB | 112.625% | No, omitted |",
        "",
        "## Results",
        "",
        "| Budget | App | Output tok/s | Mean TPOT | GPU→CPU KV | CPU→GPU KV | Load ops | 1s load bursts | Preemptions |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in cells:
        spec = cell["spec"]
        result = cell["result"]
        report_lines.append(
            f"| +{spec.budget_gib} GiB ({100 * spec.utilization:.3f}%) | {spec.application_label} | "
            f"{result['output_tokens_per_second']:.2f} | {result['mean_tpot_ms']:.2f} ms | "
            f"{result['kv_offload_gpu_to_cpu_gib']:.2f} GiB | {result['kv_offload_cpu_to_gpu_gib']:.2f} GiB | "
            f"{result['kv_offload_load_operations']:.0f} | {cell['load_activity']['bursts']} | "
            f"{result['preemptions_during_run']:.0f} |"
        )
    report_lines.extend(
        [
            "",
            "## Direction and revisits",
            "",
            "NVIDIA defines PCIe TX/RX from the GPU's perspective: TX is GPU-to-host and RX is host-to-GPU. In the connector, store is GPU-to-CPU and load is CPU-to-GPU. Connector counters are KV-specific and establish attribution; dmon is aggregate PCIe traffic and is used for the time shape.",
            "",
            "CPU-to-GPU reloads occurred in every cell, so offloaded KV is revisited. Connector-reported load operations ranged from 35 to 71 per cell. At one-second resolution, these grouped into 8 to 22 observed bursts. The exact per-cell operation rate and seconds per burst are in `matrix_summary.csv`.",
            "",
            f"Increasing the GPU KV budget from +5 to +10 GiB raised chat throughput by {machine_summary['comparisons']['chat_throughput_gain_kv10_vs_kv5_pct']:.1f}% and code throughput by {machine_summary['comparisons']['code_throughput_gain_kv10_vs_kv5_pct']:.1f}%. It reduced connector load-operation count by {machine_summary['comparisons']['chat_load_operation_reduction_kv10_vs_kv5_pct']:.1f}% for chat and {machine_summary['comparisons']['code_load_operation_reduction_kv10_vs_kv5_pct']:.1f}% for code.",
            "",
            "## Memory interpretation",
            "",
            "The timeline figure excludes model weights and runtime allocations. Panel (a) shows only live KV bytes: CPU occupancy multiplied by the 40 GiB offload buffer and GPU occupancy multiplied by the actual GPU KV pool reported by vLLM. Panel (b) shows the same occupancy as percentages. Raw physical CPU RSS and GPU FB allocation remain available in each cell's metric CSV and in `combined_timeline_1s.csv`.",
            "",
            "## Measurement limits",
            "",
            "- `nvidia-smi dmon` PCIe values are GPU-centric MB/s over the previous 20 ms, sampled once per second. Short bursts can be aliased or missed.",
            "- Hardware PCIe includes model/runtime transfers as well as KV traffic; it is not used alone to attribute bytes to KV offload.",
            "- Connector load/store counters are authoritative for KV direction and totals, but operation count is connector-reported transfer operations, not individual KV blocks.",
            "- Output throughput in tokens/s is the primary GPU serving-performance unit, consistent with vLLM/Punica-style systems papers. TTFT and TPOT are retained in the summary data.",
            "",
            "## Artifacts",
            "",
            "- Timeline figure: `figures/opt13b_offload_timelines.png` and `.pdf`",
            "- Performance/traffic summary: `figures/opt13b_offload_summary.png` and `.pdf`",
            "- Cell summary: `matrix_summary.csv` and `matrix_summary.json`",
            "- Resampled timeline: `combined_timeline_1s.csv`",
            "- Original one-second streams remain under each cell's `metrics/` directory.",
        ]
    )
    (MATRIX_DIR / "final_report.md").write_text("\n".join(report_lines) + "\n")


def main() -> int:
    add_unified_style()
    trace_manifest = read_json(TRACE_MANIFEST_PATH)
    cells = [load_cell(spec, trace_manifest) for spec in CELLS]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timeline_png, timeline_pdf = plot_timelines(cells)
    summary_png, summary_pdf = plot_summary(cells)
    write_outputs(cells, trace_manifest)
    print(f"Timeline PNG: {timeline_png}")
    print(f"Timeline PDF: {timeline_pdf}")
    print(f"Summary PNG: {summary_png}")
    print(f"Summary PDF: {summary_pdf}")
    print(f"CSV: {MATRIX_DIR / 'matrix_summary.csv'}")
    print(f"Timeline CSV: {MATRIX_DIR / 'combined_timeline_1s.csv'}")
    print(f"JSON: {MATRIX_DIR / 'matrix_summary.json'}")
    print(f"Report: {MATRIX_DIR / 'final_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())