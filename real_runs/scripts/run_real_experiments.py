#!/usr/bin/env python3
import csv
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import psutil


ROOT = Path(__file__).resolve().parents[2]


def path_env(name, default):
    value = os.environ.get(name)
    path = Path(value).expanduser() if value else Path(default).expanduser()
    return path if path.is_absolute() else ROOT / path


def json_config(file_env, raw_env, default):
    path = os.environ.get(file_env)
    raw = os.environ.get(raw_env)
    if path and raw:
        raise SystemExit(f"set only one of {file_env} or {raw_env}")
    if path:
        config_path = Path(path).expanduser()
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        return json.loads(config_path.read_text())
    if raw:
        return json.loads(raw)
    return default


REAL = path_env("REAL_RUN_DIR", ROOT / "real_runs")
HF_HOME = path_env("REAL_RUN_HF_HOME", os.environ.get("HF_HOME", ROOT / ".hf_cache" / "huggingface"))
CPU_VLLM = path_env("REAL_RUN_CPU_VLLM", ROOT / ".venv-vllm-cpu-0102" / "bin" / "vllm")
GPU_VLLM = path_env("REAL_RUN_GPU_VLLM", ROOT / ".venv-vllm-gpu-0102" / "bin" / "vllm")
SERVER_HOST = os.environ.get("REAL_RUN_HOST", "127.0.0.1")
CONFIG_DIR = path_env("REAL_RUN_CONFIG_DIR", ROOT / "real_runs" / "config")


def read_config(name):
    path = CONFIG_DIR / name
    if not path.exists():
        raise SystemExit(f"missing config file: {path}")
    return json.loads(path.read_text())


DEFAULT_MODELS = read_config("models.json")
DEFAULT_WORKLOADS = read_config("workloads.json")
DEFAULT_CPU_CORE_COUNTS = read_config("cpu_core_counts.json")
DEFAULT_GPU_MODES = read_config("gpu_modes.json")

MAX_MODEL_LEN = int(os.environ.get("REAL_RUN_MAX_MODEL_LEN", "8192"))
SERVER_READY_TIMEOUT_S = int(os.environ.get("REAL_RUN_SERVER_READY_TIMEOUT_S", "7200"))
DEFAULT_BENCH_TIMEOUT_S = int(os.environ.get("REAL_RUN_BENCH_TIMEOUT_S", "300"))
SMALL_BENCH_TIMEOUT_S = int(os.environ.get("REAL_RUN_SMALL_BENCH_TIMEOUT_S", "300"))
COOLDOWN_S = int(os.environ.get("REAL_RUN_COOLDOWN_S", "15"))
CPU_SERVER_PORT = int(os.environ.get("REAL_RUN_CPU_SERVER_PORT", "18200"))
GPU_MEMORY_UTILIZATION = os.environ.get("REAL_RUN_GPU_MEMORY_UTILIZATION", "0.90")
GPU_ENGINE = os.environ.get("REAL_RUN_GPU_ENGINE", "v0")
GPU_SWAP_SPACE_GB = os.environ.get("REAL_RUN_GPU_SWAP_SPACE_GB", "64")
GPU_PREEMPTION_MODE = os.environ.get("REAL_RUN_GPU_PREEMPTION_MODE", "swap")
GPU_KV_OFFLOADING_SIZE_GB = os.environ.get("REAL_RUN_GPU_KV_OFFLOADING_SIZE_GB")
GPU_KV_OFFLOADING_BACKEND = os.environ.get("REAL_RUN_GPU_KV_OFFLOADING_BACKEND", "native")
GPU_KV_OFFLOAD_PROMPT_ONLY_RAW = os.environ.get("REAL_RUN_GPU_KV_OFFLOAD_PROMPT_ONLY", "true").lower()
GPU_USE_FLASHINFER_SAMPLER = os.environ.get("REAL_RUN_GPU_USE_FLASHINFER_SAMPLER", "0")
CPU_KVCACHE_GB = os.environ.get("REAL_RUN_CPU_KVCACHE_GB", "32")
CLIENT_CPUSET = os.environ.get("REAL_RUN_CLIENT_CPUSET", str(max((os.cpu_count() or 1) - 1, 0)))

if GPU_ENGINE not in {"v0", "v1"}:
    raise SystemExit("REAL_RUN_GPU_ENGINE must be v0 or v1")
if GPU_KV_OFFLOAD_PROMPT_ONLY_RAW not in {"true", "false"}:
    raise SystemExit("REAL_RUN_GPU_KV_OFFLOAD_PROMPT_ONLY must be true or false")
GPU_KV_OFFLOAD_PROMPT_ONLY = GPU_KV_OFFLOAD_PROMPT_ONLY_RAW == "true"


def csv_env(name):
    return [value.strip() for value in os.environ.get(name, "").split(",") if value.strip()]


def select_items_by_env(items, env_name, keys):
    values = set(csv_env(env_name))
    if not values:
        return items
    selected = [item for item in items if any(str(item.get(key)) in values for key in keys)]
    if not selected:
        raise SystemExit(f"{env_name}={sorted(values)} did not match any configured item")
    return selected


def int_list_env(name, default):
    values = csv_env(name)
    if not values:
        return default
    parsed = [int(value) for value in values]
    if any(value <= 0 for value in parsed):
        raise SystemExit(f"{name} must contain positive integers")
    return parsed


def load_models(default):
    payload = json_config("REAL_RUN_MODELS_FILE", "REAL_RUN_MODELS_JSON", default)
    if not isinstance(payload, list):
        raise SystemExit("REAL_RUN_MODELS must be a list")
    models = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("size") or not item.get("model"):
            raise SystemExit("each model must contain size and model")
        models.append({"size": str(item["size"]), "model": str(item["model"])})
    return models


def load_gpu_modes(default):
    payload = json_config("REAL_RUN_GPU_MODES_FILE", "REAL_RUN_GPU_MODES_JSON", default)
    if not isinstance(payload, list):
        raise SystemExit("REAL_RUN_GPU_MODES must be a list")
    modes = []
    for item in payload:
        if not isinstance(item, dict):
            raise SystemExit("each GPU mode must be an object")
        for key in ("name", "visible_devices", "tensor_parallel_size", "port"):
            if key not in item:
                raise SystemExit(f"GPU mode missing {key}")
        modes.append(
            {
                "name": str(item["name"]),
                "visible_devices": str(item["visible_devices"]),
                "tensor_parallel_size": int(item["tensor_parallel_size"]),
                "port": int(item["port"]),
            }
        )
    return modes


def load_workloads(default):
    payload = json_config("REAL_RUN_WORKLOADS_FILE", "REAL_RUN_WORKLOADS_JSON", default)

    workloads = {}
    for name, workload in payload.items():
        if not isinstance(workload, dict):
            raise SystemExit(f"workload {name} must be an object")
        item = dict(workload)
        for key in ("num_prompts", "max_num_seqs"):
            if key not in item:
                raise SystemExit(f"workload {name} missing {key}")
            item[key] = int(item[key])
            if item[key] <= 0:
                raise SystemExit(f"workload {name} {key} must be positive")
        item["dataset_name"] = str(item.get("dataset_name", "random"))
        if item["dataset_name"] not in {"random", "sharegpt", "hf", "spec_bench", "custom"}:
            raise SystemExit(f"workload {name} has unsupported dataset_name={item['dataset_name']}")
        for key in ("input_len", "output_len", "max_concurrency"):
            if item.get(key) is not None:
                item[key] = int(item[key])
                if item[key] <= 0:
                    raise SystemExit(f"workload {name} {key} must be positive")
        if item.get("temperature") is not None:
            item["temperature"] = float(item["temperature"])
            if item["temperature"] < 0:
                raise SystemExit(f"workload {name} temperature must be non-negative")
        if item["dataset_name"] == "random" and (item.get("input_len") is None or item.get("output_len") is None):
            raise SystemExit(f"random workload {name} requires input_len and output_len")
        if item.get("input_len") and item.get("output_len") and item["input_len"] + item["output_len"] > MAX_MODEL_LEN:
            raise SystemExit(
                f"workload {name} input_len+output_len={item['input_len'] + item['output_len']} exceeds MAX_MODEL_LEN={MAX_MODEL_LEN}"
            )
        if item["max_num_seqs"] > item["num_prompts"]:
            raise SystemExit(f"workload {name} max_num_seqs cannot exceed num_prompts")
        if item.get("max_concurrency") and item["max_concurrency"] > item["num_prompts"]:
            raise SystemExit(f"workload {name} max_concurrency cannot exceed num_prompts")
        if item["dataset_name"] in {"sharegpt", "spec_bench", "custom"} and not item.get("dataset_path"):
            raise SystemExit(f"workload {name} requires dataset_path")
        item["ignore_eos"] = bool(item.get("ignore_eos", False))
        item["no_stream"] = bool(item.get("no_stream", False))
        item["save_detailed"] = bool(item.get("save_detailed", True))
        item["skip_chat_template"] = bool(item.get("skip_chat_template", False))
        item["disable_shuffle"] = bool(item.get("disable_shuffle", False))
        item["no_oversample"] = bool(item.get("no_oversample", False))
        workloads[name] = item

    selected = set(csv_env("REAL_RUN_WORKLOADS"))
    if selected:
        workloads = {name: workload for name, workload in workloads.items() if name in selected}
        if not workloads:
            raise SystemExit(f"REAL_RUN_WORKLOADS={sorted(selected)} did not match any configured workload")
    return workloads


MODELS = select_items_by_env(load_models(DEFAULT_MODELS), "REAL_RUN_MODEL_SIZES", ("size", "model"))
GPU_MODES = select_items_by_env(load_gpu_modes(DEFAULT_GPU_MODES), "REAL_RUN_GPU_MODES", ("name", "visible_devices"))
CPU_CORE_COUNTS = int_list_env("REAL_RUN_CPU_CORE_COUNTS", DEFAULT_CPU_CORE_COUNTS)
WORKLOADS = load_workloads(DEFAULT_WORKLOADS)
RUN_TARGETS = set(csv_env("REAL_RUN_TARGETS"))


def target_enabled(target_name):
    if not RUN_TARGETS:
        return True
    if target_name == "cpu":
        return "cpu" in RUN_TARGETS
    return target_name in RUN_TARGETS or "gpu" in RUN_TARGETS


def cpu_server_cpuset():
    return os.environ.get("REAL_RUN_CPU_SERVER_CPUSET", f"0-{max(CPU_CORE_COUNTS) - 1}")


def cpu_server_threads():
    return os.environ.get("REAL_RUN_CPU_SERVER_OMP_THREADS", str(max(CPU_CORE_COUNTS)))


def server_max_num_seqs():
    return max((workload["max_num_seqs"] for workload in WORKLOADS.values()), default=4)


DMON_FIELDS = [
    "gpu",
    "pwr_w",
    "gtemp_c",
    "mtemp_c",
    "sm_pct",
    "mem_pct",
    "enc_pct",
    "dec_pct",
    "jpg_pct",
    "ofa_pct",
    "mclk_mhz",
    "pclk_mhz",
    "fb_mb",
    "bar1_mb",
    "ccpm_mb",
    "rxpci_mbps",
    "txpci_mbps",
]

ENGINE_METRICS = {
    "vllm:num_requests_running": "num_requests_running",
    "vllm:num_requests_waiting": "num_requests_waiting",
    "vllm:num_requests_swapped": "num_requests_swapped",
    "vllm:gpu_cache_usage_perc": "gpu_kv_cache_usage_fraction",
    "vllm:kv_cache_usage_perc": "gpu_kv_cache_usage_fraction",
    "vllm:cpu_cache_usage_perc": "cpu_kv_cache_usage_fraction",
    "vllm:kv_offload_cpu_cache_usage_perc": "cpu_kv_cache_usage_fraction",
    "vllm:kv_offload_load_bytes": "kv_offload_load_bytes_total",
    "vllm:kv_offload_load_bytes_total": "kv_offload_load_bytes_total",
    "vllm:kv_offload_store_bytes": "kv_offload_store_bytes_total",
    "vllm:kv_offload_store_bytes_total": "kv_offload_store_bytes_total",
    "vllm:kv_offload_load_time": "kv_offload_load_time_seconds_total",
    "vllm:kv_offload_load_time_total": "kv_offload_load_time_seconds_total",
    "vllm:kv_offload_store_time": "kv_offload_store_time_seconds_total",
    "vllm:kv_offload_store_time_total": "kv_offload_store_time_seconds_total",
    "vllm:kv_offload_load_size_count": "kv_offload_load_operations_total",
    "vllm:kv_offload_store_size_count": "kv_offload_store_operations_total",
    "vllm:kv_offload_load_size_sum": "kv_offload_load_size_bytes_total",
    "vllm:kv_offload_store_size_sum": "kv_offload_store_size_bytes_total",
    "vllm:kv_offload_stores_skipped": "kv_offload_stores_skipped_total",
    "vllm:kv_offload_stores_skipped_total": "kv_offload_stores_skipped_total",
    "vllm:num_preemptions_total": "num_preemptions_total",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
    "vllm:request_success_total": "request_success_total",
    "vllm:time_to_first_token_seconds_sum": "time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count": "time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum": "inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count": "inter_token_latency_seconds_count",
    "vllm:e2e_request_latency_seconds_sum": "e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count": "e2e_request_latency_seconds_count",
    "vllm:request_queue_time_seconds_sum": "request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_count": "request_queue_time_seconds_count",
}

ENGINE_LOG_PATTERN = re.compile(
    r"Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation>[0-9.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Swapped: (?P<swapped>\d+) reqs, "
    r"Pending: (?P<pending>\d+) reqs, GPU KV cache usage: (?P<gpu_kv>[0-9.]+)%, "
    r"CPU KV cache usage: (?P<cpu_kv>[0-9.]+)%\."
)
ENGINE_LOG_PATTERN_V1 = re.compile(
    r"Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation>[0-9.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<gpu_kv>[0-9.]+)%"
)
PREEMPTION_LOG_PATTERN = re.compile(
    r"preempted by (?P<mode>\w+) mode.*total_cumulative_preemption_cnt=(?P<count>\d+)"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def capture_command(cmd):
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return {"command": cmd, "returncode": completed.returncode, "output": completed.stdout.strip()}
    except OSError as exc:
        return {"command": cmd, "returncode": None, "output": "", "error": repr(exc)}


def system_snapshot():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage(ROOT)
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    vllm_python = GPU_VLLM.parent / "python"
    software = capture_command(
        [
            str(vllm_python),
            "-c",
            (
                "import json, torch, transformers, vllm; "
                "print(json.dumps({'vllm': vllm.__version__, 'torch': torch.__version__, "
                "'torch_cuda': torch.version.cuda, 'transformers': transformers.__version__, "
                "'cuda_available': torch.cuda.is_available()}))"
            ),
        ]
    )
    gpu_query = capture_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,compute_cap,memory.total,power.limit,clocks.max.sm,clocks.max.memory,pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader,nounits",
        ]
    )
    energy_paths = sorted(str(path) for path in Path("/sys/class/powercap").glob("**/energy_uj"))
    return {
        "timestamp_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu_model": cpu_model,
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "memory_total_bytes": memory.total,
        "memory_available_bytes": memory.available,
        "swap_total_bytes": swap.total,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "cpu_energy_available": bool(energy_paths),
        "cpu_energy_paths": energy_paths,
        "cpu_energy_note": (
            "Measured from Linux powercap energy_uj counters."
            if energy_paths
            else "Unavailable: this VM exposes no Linux powercap/RAPL energy counter."
        ),
        "gpu_energy_method": "trapezoidal integral of 1-second nvidia-smi board-power samples",
        "gpu_query": gpu_query,
        "gpu_topology": capture_command(["nvidia-smi", "topo", "-m"]),
        "numa_topology": capture_command(["numactl", "--hardware"]),
        "pci_device": capture_command(["lspci", "-vv", "-s", "06:00.0"]),
        "lscpu": capture_command(["lscpu", "--json"]),
        "software": software,
        "vllm_source_revision": capture_command(["git", "-C", str(ROOT / "vllm_source_0102"), "rev-parse", "HEAD"]),
    }


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def parse_num(value):
    if value in {"", "-"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_prometheus_metrics(text):
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric_name = parts[0].split("{", 1)[0]
        output_name = ENGINE_METRICS.get(metric_name)
        if output_name is None:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        values[output_name] = values.get(output_name, 0.0) + value
    return values


def integrate_samples(rows, value_key):
    total = 0.0
    previous = None
    for row in rows:
        elapsed_s = row.get("elapsed_s")
        value = row.get(value_key)
        if elapsed_s is None or not isinstance(value, (int, float)):
            continue
        if previous is not None:
            previous_elapsed, previous_value = previous
            total += 0.5 * (previous_value + value) * max(0.0, elapsed_s - previous_elapsed)
        previous = (elapsed_s, value)
    return total


def cumulative_delta(rows, value_key):
    values = [row.get(value_key) for row in rows]
    values = [value for value in values if isinstance(value, (int, float))]
    if not values:
        return 0.0
    first_sample = rows[0].get(value_key) if rows else None
    baseline = first_sample if isinstance(first_sample, (int, float)) else 0.0
    return max(0.0, max(values) - baseline)


def parse_engine_log(text):
    rows = []
    preemption_counts = []
    preemption_modes = set()
    for line in text.splitlines():
        match = ENGINE_LOG_PATTERN.search(line)
        if match:
            rows.append(
                {
                    "prompt_throughput_tps": float(match.group("prompt")),
                    "generation_throughput_tps": float(match.group("generation")),
                    "num_requests_running": int(match.group("running")),
                    "num_requests_swapped": int(match.group("swapped")),
                    "num_requests_pending": int(match.group("pending")),
                    "gpu_kv_cache_usage_pct": float(match.group("gpu_kv")),
                    "cpu_kv_cache_usage_pct": float(match.group("cpu_kv")),
                }
            )
        else:
            match = ENGINE_LOG_PATTERN_V1.search(line)
            if match:
                rows.append(
                    {
                        "prompt_throughput_tps": float(match.group("prompt")),
                        "generation_throughput_tps": float(match.group("generation")),
                        "num_requests_running": int(match.group("running")),
                        "num_requests_waiting": int(match.group("waiting")),
                        "num_requests_swapped": 0,
                        "num_requests_pending": int(match.group("waiting")),
                        "gpu_kv_cache_usage_pct": float(match.group("gpu_kv")),
                    }
                )
        match = PREEMPTION_LOG_PATTERN.search(line)
        if match:
            preemption_modes.add(match.group("mode"))
            preemption_counts.append(int(match.group("count")))
    return {
        "rows": rows,
        "preemption_modes": sorted(preemption_modes),
        "max_cumulative_preemption_count": max(preemption_counts, default=0),
        "preemption_warning_count": len(preemption_counts),
    }


def process_tree(pid):
    try:
        parent = psutil.Process(pid)
        return [parent] + parent.children(recursive=True)
    except psutil.Error:
        return []


def set_tree_affinity(pid, cpus):
    for proc in process_tree(pid):
        try:
            proc.cpu_affinity(cpus)
        except (psutil.Error, AttributeError):
            continue


def terminate_tree(proc, timeout_s=20):
    procs = process_tree(proc.pid)
    for item in reversed(procs):
        try:
            item.send_signal(signal.SIGTERM)
        except psutil.Error:
            pass
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        for item in reversed(process_tree(proc.pid)):
            try:
                item.kill()
            except psutil.Error:
                pass
        try:
            proc.kill()
        except OSError:
            pass


def terminate_process_group(proc, timeout_s=10):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass


def collect_temperatures():
    readings = []
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name = read_text(hwmon / "name")
        for temp_file in sorted(hwmon.glob("temp*_input")):
            value = read_text(temp_file)
            if value is None:
                continue
            try:
                temp_c = int(value) / 1000.0
            except ValueError:
                continue
            label = read_text(temp_file.with_name(temp_file.name.replace("_input", "_label")))
            readings.append({"name": name, "label": label, "temp_c": temp_c})
    return readings


def powercap_energy_uj():
    total = 0
    found = False
    for path in Path("/sys/class/powercap").glob("*/energy_uj"):
        value = read_text(path)
        if value is None:
            continue
        try:
            total += int(value)
            found = True
        except ValueError:
            pass
    return total if found else None


class CpuMonitor:
    def __init__(self, run_id, server_pid, target, sample_s=1.0):
        self.run_id = run_id
        self.server_pid = server_pid
        self.target = target
        self.sample_s = sample_s
        self.rows = []
        self._stop = threading.Event()
        self._thread = None
        self._last_elapsed = None
        self._last_cpu_time = None
        self._last_energy = None
        self._last_energy_elapsed = None

    def start(self):
        psutil.cpu_percent(interval=None, percpu=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        started = time.time()
        while not self._stop.is_set():
            self.rows.append(self._sample(time.time() - started))
            time.sleep(self.sample_s)

    def _sample(self, elapsed_s):
        rss = 0
        vms = 0
        cpu_time = 0.0
        read_bytes = 0
        write_bytes = 0
        proc_count = 0
        for proc in process_tree(self.server_pid):
            try:
                with proc.oneshot():
                    mem = proc.memory_info()
                    rss += mem.rss
                    vms += mem.vms
                    times = proc.cpu_times()
                    cpu_time += times.user + times.system
                    proc_count += 1
                    try:
                        io = proc.io_counters()
                        read_bytes += io.read_bytes
                        write_bytes += io.write_bytes
                    except (psutil.Error, AttributeError):
                        pass
            except psutil.Error:
                continue
        proc_cpu_pct = 0.0
        if self._last_elapsed is not None and elapsed_s > self._last_elapsed:
            proc_cpu_pct = 100.0 * (cpu_time - self._last_cpu_time) / (elapsed_s - self._last_elapsed)
            proc_cpu_pct = max(0.0, proc_cpu_pct)
        self._last_elapsed = elapsed_s
        self._last_cpu_time = cpu_time

        energy = powercap_energy_uj()
        package_power_w = None
        if energy is not None and self._last_energy is not None and elapsed_s > self._last_energy_elapsed:
            package_power_w = ((energy - self._last_energy) / 1_000_000.0) / (elapsed_s - self._last_energy_elapsed)
        self._last_energy = energy
        self._last_energy_elapsed = elapsed_s

        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()
        cpu_stats = psutil.cpu_stats()
        cpu_freq = psutil.cpu_freq()
        temps = [row["temp_c"] for row in collect_temperatures()]
        return {
            "timestamp_utc": utc_now(),
            "run_id": self.run_id,
            "target": self.target,
            "elapsed_s": elapsed_s,
            "server_process_count": proc_count,
            "server_rss_bytes": rss,
            "server_vms_bytes": vms,
            "server_cpu_time_s": cpu_time,
            "server_cpu_percent": proc_cpu_pct,
            "server_read_bytes": read_bytes,
            "server_write_bytes": write_bytes,
            "system_cpu_total_percent": sum(per_cpu) / len(per_cpu) if per_cpu else None,
            "system_cpu_max_core_percent": max(per_cpu) if per_cpu else None,
            "system_cpu_freq_mhz": cpu_freq.current if cpu_freq else None,
            "system_load_1m": os.getloadavg()[0],
            "system_context_switches": cpu_stats.ctx_switches,
            "system_mem_used_bytes": mem.used,
            "system_mem_available_bytes": mem.available,
            "system_swap_used_bytes": swap.used,
            "system_disk_read_bytes": disk.read_bytes if disk else None,
            "system_disk_write_bytes": disk.write_bytes if disk else None,
            "system_net_recv_bytes": net.bytes_recv if net else None,
            "system_net_sent_bytes": net.bytes_sent if net else None,
            "cpu_temp_max_c": max(temps) if temps else None,
            "powercap_energy_uj": energy,
            "powercap_package_power_w": package_power_w,
        }


class DmonMonitor:
    def __init__(self, run_id):
        self.run_id = run_id
        self.rows = []
        self.raw_lines = []
        self.proc = None
        self.thread = None
        self.started = None

    def start(self):
        self.started = time.time()
        self.proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "pucmt", "-d", "1"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=5)

    def _reader(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            self.raw_lines.append(line)
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != len(DMON_FIELDS):
                continue
            row = {
                "timestamp_utc": utc_now(),
                "run_id": self.run_id,
                "elapsed_s": time.time() - self.started if self.started else None,
            }
            row.update({name: parse_num(value) for name, value in zip(DMON_FIELDS, parts)})
            self.rows.append(row)


class EngineMonitor:
    def __init__(self, run_id, port, sample_s=1.0):
        self.run_id = run_id
        self.port = port
        self.sample_s = sample_s
        self.rows = []
        self.errors = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        started = time.time()
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(
                    f"http://{SERVER_HOST}:{self.port}/metrics", timeout=5
                ) as response:
                    metrics = parse_prometheus_metrics(response.read().decode("utf-8", errors="replace"))
                row = {
                    "timestamp_utc": utc_now(),
                    "run_id": self.run_id,
                    "elapsed_s": time.time() - started,
                }
                row.update(metrics)
                self.rows.append(row)
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                self.errors.append(f"{utc_now()} {exc!r}")
            self._stop.wait(self.sample_s)


def server_ready(port, served_name):
    url = f"http://{SERVER_HOST}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return resp.status == 200 and served_name in text
    except (urllib.error.URLError, TimeoutError, socket.timeout):
        return False


def wait_for_server(proc, port, served_name, timeout_s, log_path):
    started = time.time()
    while timeout_s <= 0 or time.time() - started < timeout_s:
        if proc.poll() is not None:
            return False, f"server_exited_returncode_{proc.returncode}"
        if server_ready(port, served_name):
            return True, "ready"
        time.sleep(5)
    return False, f"server_ready_timeout_after_{timeout_s}s"


def base_env():
    env = os.environ.copy()
    env["VLLM_NO_USAGE_STATS"] = "1"
    env["HF_HOME"] = str(HF_HOME)
    env["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    return env


def start_server(model, target_name, port, model_dir, gpu_mode=None):
    served_name = f"{model['size']}-{target_name}"
    log_path = model_dir / target_name / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = base_env()
    if target_name == "cpu":
        cpu_range = cpu_server_cpuset()
        env.update(
            {
                "VLLM_TARGET_DEVICE": "cpu",
                "VLLM_CPU_KVCACHE_SPACE": CPU_KVCACHE_GB,
                "VLLM_CPU_OMP_THREADS_BIND": cpu_range,
                "OMP_NUM_THREADS": cpu_server_threads(),
            }
        )
        cmd = [
            "taskset",
            "-c",
            cpu_range,
            str(CPU_VLLM),
            "serve",
            model["model"],
            "--host",
            SERVER_HOST,
            "--port",
            str(port),
            "--served-model-name",
            served_name,
            "--dtype",
            "float32",
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--max-num-seqs",
            str(server_max_num_seqs()),
            "--disable-log-requests",
        ]
    else:
        env["CUDA_VISIBLE_DEVICES"] = gpu_mode["visible_devices"]
        if GPU_ENGINE == "v0":
            env["VLLM_USE_V1"] = "0"
        else:
            env.pop("VLLM_USE_V1", None)
            env["VLLM_USE_FLASHINFER_SAMPLER"] = GPU_USE_FLASHINFER_SAMPLER
        cmd = [
            str(GPU_VLLM),
            "serve",
            model["model"],
            "--host",
            SERVER_HOST,
            "--port",
            str(port),
            "--served-model-name",
            served_name,
            "--dtype",
            "float16",
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--max-num-seqs",
            str(server_max_num_seqs()),
            "--gpu-memory-utilization",
            GPU_MEMORY_UTILIZATION,
            "--tensor-parallel-size",
            str(gpu_mode["tensor_parallel_size"]),
            "--disable-custom-all-reduce",
        ]
        if GPU_ENGINE == "v0":
            cmd.extend(
                [
                    "--disable-log-requests",
                    "--swap-space",
                    GPU_SWAP_SPACE_GB,
                    "--preemption-mode",
                    GPU_PREEMPTION_MODE,
                ]
            )
        elif GPU_KV_OFFLOADING_SIZE_GB:
            if GPU_KV_OFFLOADING_BACKEND == "native" and not GPU_KV_OFFLOAD_PROMPT_ONLY:
                cpu_bytes_to_use = int(float(GPU_KV_OFFLOADING_SIZE_GB) * (1 << 30))
                transfer_config = {
                    "kv_connector": "OffloadingConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "cpu_bytes_to_use": cpu_bytes_to_use,
                        "offload_prompt_only": False,
                    },
                }
                cmd.extend(
                    [
                        "--kv-transfer-config",
                        json.dumps(transfer_config, separators=(",", ":")),
                    ]
                )
            else:
                cmd.extend(
                    [
                        "--kv-offloading-size",
                        GPU_KV_OFFLOADING_SIZE_GB,
                        "--kv-offloading-backend",
                        GPU_KV_OFFLOADING_BACKEND,
                    ]
                )
    with log_path.open("w") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    return proc, served_name, log_path, cmd


def bench_command(vllm_bin, port, served_name, model_id, workload, run_dir):
    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(vllm_bin),
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--host",
        SERVER_HOST,
        "--port",
        str(port),
        "--model",
        model_id,
        "--served-model-name",
        served_name,
        "--tokenizer",
        model_id,
        "--dataset-name",
        workload["dataset_name"],
        "--num-prompts",
        str(workload["num_prompts"]),
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        "bench_serve.json",
        "--metadata",
        f"max_model_len={MAX_MODEL_LEN}",
    ]
    dataset_name = workload["dataset_name"]
    if dataset_name == "random":
        cmd.extend(
            [
                "--random-input-len",
                str(workload["input_len"]),
                "--random-output-len",
                str(workload["output_len"]),
                "--random-range-ratio",
                str(workload.get("random_range_ratio", 0)),
            ]
        )
    elif dataset_name == "sharegpt":
        cmd.extend(["--dataset-path", str(workload["dataset_path"])])
        if workload.get("output_len") is not None:
            cmd.extend(["--sharegpt-output-len", str(workload["output_len"])])
    elif dataset_name == "hf":
        cmd.extend(["--dataset-path", str(workload["dataset_path"])])
        for key, flag in (
            ("hf_name", "--hf-name"),
            ("hf_subset", "--hf-subset"),
            ("hf_split", "--hf-split"),
        ):
            if workload.get(key) is not None:
                cmd.extend([flag, str(workload[key])])
        if workload.get("output_len") is not None:
            cmd.extend(["--hf-output-len", str(workload["output_len"])])
    elif dataset_name == "spec_bench":
        cmd.extend(["--dataset-path", str(workload["dataset_path"])])
        if workload.get("output_len") is not None:
            cmd.extend(["--spec-bench-output-len", str(workload["output_len"])])
        if workload.get("spec_bench_category") is not None:
            cmd.extend(["--spec-bench-category", str(workload["spec_bench_category"])])
    elif dataset_name == "custom":
        cmd.extend(["--dataset-path", str(workload["dataset_path"])])
        if workload.get("output_len") is not None:
            cmd.extend(["--custom-output-len", str(workload["output_len"])])
    if workload.get("max_concurrency") is not None:
        cmd.extend(["--max-concurrency", str(workload["max_concurrency"])])
    if workload.get("request_rate") is not None:
        cmd.extend(["--request-rate", str(workload["request_rate"])])
    if workload.get("burstiness") is not None:
        cmd.extend(["--burstiness", str(workload["burstiness"])])
    if workload.get("temperature") is not None:
        cmd.extend(["--temperature", str(workload["temperature"])])
    if workload["skip_chat_template"]:
        cmd.append("--skip-chat-template")
    if workload["disable_shuffle"]:
        cmd.append("--disable-shuffle")
    if workload["no_oversample"]:
        cmd.append("--no-oversample")
    if workload["ignore_eos"]:
        cmd.append("--ignore-eos")
    if workload["no_stream"]:
        cmd.append("--no-stream")
    if workload["save_detailed"]:
        cmd.append("--save-detailed")
    return cmd


def benchmark_timeout_s(model_size):
    if model_size == "small":
        value = SMALL_BENCH_TIMEOUT_S
    else:
        value = DEFAULT_BENCH_TIMEOUT_S
    return None if value <= 0 else value


def run_benchmark_client(cmd, run_dir, env, target, timeout_s):
    log_path = run_dir / "logs" / "bench_client.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    client_cmd = cmd
    if target == "cpu":
        client_cmd = ["taskset", "-c", CLIENT_CPUSET] + cmd
    elif target.startswith("gpu"):
        client_cmd = ["taskset", "-c", CLIENT_CPUSET] + cmd
    with log_path.open("w") as f:
        started = time.time()
        proc = subprocess.Popen(
            client_cmd,
            cwd=ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            if timeout_s is None:
                returncode = proc.wait()
            else:
                returncode = proc.wait(timeout=timeout_s)
            return returncode, False, time.time() - started, log_path, client_cmd
        except subprocess.TimeoutExpired:
            terminate_process_group(proc)
            f.write(f"\nBENCHMARK_TIMEOUT_EXCEEDED timeout_s={timeout_s}\n")
            return None, True, time.time() - started, log_path, client_cmd


def load_bench_json(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if isinstance(payload, list) and payload:
        return payload[-1]
    return payload


def run_dir_for(model_size, target, workload, cpu_cores=None):
    if target == "cpu":
        return REAL / model_size / f"cpu_{cpu_cores}" / workload
    return REAL / model_size / target / workload


def load_existing_result(run_dir):
    path = run_dir / "result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def is_completed_result(result):
    if not result:
        return False
    if result.get("status") == "passed":
        return True
    if result.get("status") == "timeout":
        health = result.get("timeout_health_check") or {}
        return not health.get("timeout_looks_like_unhealthy_hang", False)
    return False


def planned_cells(model, target_name):
    cells = []
    if target_name == "cpu":
        for cores in CPU_CORE_COUNTS:
            for workload_name, workload in WORKLOADS.items():
                cells.append((workload_name, workload, run_dir_for(model["size"], "cpu", workload_name, cores), cores))
    else:
        for workload_name, workload in WORKLOADS.items():
            cells.append((workload_name, workload, run_dir_for(model["size"], target_name, workload_name), None))
    return cells


def metric_summary(cpu_rows, gpu_rows, engine_rows, engine_log_stats=None):
    engine_log_stats = engine_log_stats or {"rows": [], "preemption_modes": [], "max_cumulative_preemption_count": 0, "preemption_warning_count": 0}
    engine_log_rows = engine_log_stats["rows"]
    gpu_energy_j = integrate_samples(gpu_rows, "pwr_w")
    cpu_package_energy_j = integrate_samples(cpu_rows, "powercap_package_power_w")
    pcie_rx_mb = integrate_samples(gpu_rows, "rxpci_mbps")
    pcie_tx_mb = integrate_samples(gpu_rows, "txpci_mbps")
    kv_load_bytes = cumulative_delta(engine_rows, "kv_offload_load_bytes_total")
    kv_store_bytes = cumulative_delta(engine_rows, "kv_offload_store_bytes_total")
    preemption_values = [row.get("num_preemptions_total") for row in engine_rows]
    preemption_values = [value for value in preemption_values if isinstance(value, (int, float))]
    out = {
        "cpu_metric_rows": len(cpu_rows),
        "gpu_metric_rows": len(gpu_rows),
        "engine_metric_rows": len(engine_rows),
        "max_server_rss_gib": max((row["server_rss_bytes"] for row in cpu_rows), default=0) / (1024**3),
        "max_server_cpu_percent": max((row["server_cpu_percent"] for row in cpu_rows), default=0),
        "max_system_cpu_percent": max((row["system_cpu_total_percent"] or 0 for row in cpu_rows), default=0),
        "max_cpu_temp_c": max((row["cpu_temp_max_c"] or 0 for row in cpu_rows), default=0),
        "max_powercap_package_power_w": max((row["powercap_package_power_w"] or 0 for row in cpu_rows), default=0),
        "cpu_energy_available": any(row.get("powercap_energy_uj") is not None for row in cpu_rows),
        "cpu_package_energy_j": cpu_package_energy_j if cpu_package_energy_j > 0 else None,
        "max_gpu_fb_gib": max((row.get("fb_mb") or 0 for row in gpu_rows), default=0) / 1024,
        "max_gpu_sm_pct": max((row.get("sm_pct") or 0 for row in gpu_rows), default=0),
        "avg_gpu_sm_pct": sum((row.get("sm_pct") or 0 for row in gpu_rows)) / len(gpu_rows) if gpu_rows else 0,
        "max_gpu_mem_pct": max((row.get("mem_pct") or 0 for row in gpu_rows), default=0),
        "avg_gpu_mem_pct": sum((row.get("mem_pct") or 0 for row in gpu_rows)) / len(gpu_rows) if gpu_rows else 0,
        "max_gpu_power_w": max((row.get("pwr_w") or 0 for row in gpu_rows), default=0),
        "avg_gpu_power_w": sum((row.get("pwr_w") or 0 for row in gpu_rows)) / len(gpu_rows) if gpu_rows else 0,
        "gpu_energy_j": gpu_energy_j,
        "gpu_energy_method": "trapezoidal_integral_of_1s_nvidia_smi_board_power",
        "max_gpu_temp_c": max((row.get("gtemp_c") or 0 for row in gpu_rows), default=0),
        "max_pcie_rx_mbps": max((row.get("rxpci_mbps") or 0 for row in gpu_rows), default=0),
        "max_pcie_tx_mbps": max((row.get("txpci_mbps") or 0 for row in gpu_rows), default=0),
        "max_pcie_rx_tx_gib_s": max(((row.get("rxpci_mbps") or 0) + (row.get("txpci_mbps") or 0) for row in gpu_rows), default=0) / 1024,
        "pcie_rx_gib": pcie_rx_mb / 1024,
        "pcie_tx_gib": pcie_tx_mb / 1024,
        "kv_offload_cpu_to_gpu_gib": kv_load_bytes / (1024**3),
        "kv_offload_gpu_to_cpu_gib": kv_store_bytes / (1024**3),
        "kv_offload_cpu_to_gpu_time_s": cumulative_delta(
            engine_rows, "kv_offload_load_time_seconds_total"
        ),
        "kv_offload_gpu_to_cpu_time_s": cumulative_delta(
            engine_rows, "kv_offload_store_time_seconds_total"
        ),
        "kv_offload_load_operations": cumulative_delta(
            engine_rows, "kv_offload_load_operations_total"
        ),
        "kv_offload_store_operations": cumulative_delta(
            engine_rows, "kv_offload_store_operations_total"
        ),
        "kv_offload_stores_skipped": cumulative_delta(
            engine_rows, "kv_offload_stores_skipped_total"
        ),
        "max_gpu_kv_cache_usage_pct": max(
            100 * max((row.get("gpu_kv_cache_usage_fraction") or 0 for row in engine_rows), default=0),
            max((row["gpu_kv_cache_usage_pct"] for row in engine_log_rows), default=0),
        ),
        "max_cpu_kv_cache_usage_pct": max(
            100 * max((row.get("cpu_kv_cache_usage_fraction") or 0 for row in engine_rows), default=0),
            max((row.get("cpu_kv_cache_usage_pct", 0) for row in engine_log_rows), default=0),
        ),
        "max_requests_running": max(
            max((row.get("num_requests_running") or 0 for row in engine_rows), default=0),
            max((row["num_requests_running"] for row in engine_log_rows), default=0),
        ),
        "max_requests_waiting": max(
            max((row.get("num_requests_waiting") or 0 for row in engine_rows), default=0),
            max((row.get("num_requests_waiting", 0) for row in engine_log_rows), default=0),
        ),
        "max_requests_swapped": max(
            max((row.get("num_requests_swapped") or 0 for row in engine_rows), default=0),
            max((row["num_requests_swapped"] for row in engine_log_rows), default=0),
        ),
        "max_requests_pending": max((row["num_requests_pending"] for row in engine_log_rows), default=0),
        "preemptions_during_run": max(
            (max(preemption_values) - min(preemption_values)) if preemption_values else 0,
            engine_log_stats["preemption_warning_count"],
        ),
        "preemption_modes": engine_log_stats["preemption_modes"],
        "max_cumulative_preemption_count": engine_log_stats["max_cumulative_preemption_count"],
        "engine_log_metric_rows": len(engine_log_rows),
    }
    return out


def plot_run(run_dir, run_id, target, cpu_rows, gpu_rows):
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if target == "cpu":
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
        x = [row["elapsed_s"] for row in cpu_rows]
        axes[0].plot(x, [row["server_cpu_percent"] for row in cpu_rows], label="server process CPU %")
        axes[0].plot(x, [row["system_cpu_total_percent"] for row in cpu_rows], label="system CPU %", alpha=0.75)
        axes[0].set_ylabel("CPU %")
        axes[0].legend()
        axes[1].plot(x, [row["server_rss_bytes"] / (1024**3) for row in cpu_rows], label="server RSS GiB")
        axes[1].set_ylabel("RSS GiB")
        axes[1].legend()
        axes[2].plot(x, [row["system_mem_used_bytes"] / (1024**3) for row in cpu_rows], label="system used memory GiB")
        axes[2].set_ylabel("Memory GiB")
        axes[2].legend()
        axes[3].plot(x, [row["cpu_temp_max_c"] or 0 for row in cpu_rows], label="max CPU temp C")
        axes[3].plot(x, [row["powercap_package_power_w"] or 0 for row in cpu_rows], label="powercap package W")
        axes[3].set_ylabel("Temp C / W")
        axes[3].legend()
    else:
        fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
        x_cpu = [row["elapsed_s"] for row in cpu_rows]
        axes[0].plot(x_cpu, [row["server_cpu_percent"] for row in cpu_rows], label="server CPU %")
        axes[0].set_ylabel("CPU %")
        axes[0].legend()
        axes[1].plot(x_cpu, [row["server_rss_bytes"] / (1024**3) for row in cpu_rows], label="server RSS GiB")
        axes[1].set_ylabel("Host RSS GiB")
        axes[1].legend()

        by_gpu = defaultdict(list)
        if gpu_rows:
            base_ts = datetime.fromisoformat(gpu_rows[0]["timestamp_utc"])
            for row in gpu_rows:
                row = dict(row)
                row["elapsed_s"] = (datetime.fromisoformat(row["timestamp_utc"]) - base_ts).total_seconds()
                by_gpu[int(row["gpu"])].append(row)
        for gpu, rows in sorted(by_gpu.items()):
            x = [row["elapsed_s"] for row in rows]
            axes[2].plot(x, [(row.get("fb_mb") or 0) / 1024 for row in rows], label=f"GPU{gpu} FB GiB")
            axes[3].plot(x, [row.get("sm_pct") or 0 for row in rows], label=f"GPU{gpu} SM %")
            axes[3].plot(x, [row.get("mem_pct") or 0 for row in rows], linestyle="--", label=f"GPU{gpu} mem util %")
            rx = [(row.get("rxpci_mbps") or 0) / 1024 for row in rows]
            tx = [(row.get("txpci_mbps") or 0) / 1024 for row in rows]
            axes[4].plot(x, rx, label=f"GPU{gpu} PCIe RX GiB/s")
            axes[4].plot(x, tx, linestyle="--", label=f"GPU{gpu} PCIe TX GiB/s")
        axes[2].set_ylabel("GPU memory GiB")
        axes[3].set_ylabel("GPU util %")
        axes[4].set_ylabel("PCIe GiB/s")
        for ax in axes[2:]:
            ax.legend(ncols=2, fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("seconds from benchmark monitor start")
    fig.suptitle(run_id)
    fig.tight_layout()
    fig_path = fig_dir / f"{run_id}_timeline.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def validate_metrics(target, cpu_rows, gpu_rows, engine_rows):
    issues = []
    if not cpu_rows:
        issues.append("cpu_metrics_empty")
    if target == "cpu":
        if not any(row["server_rss_bytes"] > 0 for row in cpu_rows):
            issues.append("cpu_rss_all_zero")
        if not any(row["server_cpu_percent"] > 0 for row in cpu_rows[1:]):
            issues.append("cpu_percent_all_zero")
    else:
        if not gpu_rows:
            issues.append("gpu_metrics_empty")
        if not any((row.get("fb_mb") or 0) > 0 for row in gpu_rows):
            issues.append("gpu_fb_all_zero")
        if not any((row.get("pwr_w") or 0) > 0 for row in gpu_rows):
            issues.append("gpu_power_all_zero")
        if not engine_rows:
            issues.append("engine_metrics_empty")
    return issues


def timeout_health_note(target, server_proc, cpu_rows, gpu_rows):
    server_alive = server_proc.poll() is None
    cpu_nonempty = bool(cpu_rows)
    cpu_active = any((row.get("server_cpu_percent") or 0) > 0 for row in cpu_rows[1:])
    gpu_active = any((row.get("sm_pct") or 0) > 0 for row in gpu_rows) if gpu_rows else False
    gpu_mem = any((row.get("fb_mb") or 0) > 0 for row in gpu_rows) if gpu_rows else False
    if target == "cpu":
        healthy = server_alive and cpu_nonempty and cpu_active
    else:
        healthy = server_alive and cpu_nonempty and (gpu_active or gpu_mem)
    return {
        "server_alive_after_timeout": server_alive,
        "monitor_data_nonempty": cpu_nonempty and (target == "cpu" or bool(gpu_rows)),
        "cpu_activity_observed": cpu_active,
        "gpu_activity_observed": gpu_active,
        "gpu_memory_observed": gpu_mem,
        "timeout_looks_like_unhealthy_hang": not healthy,
    }


def run_one_cell(model, target, workload_name, workload, server_proc, served_name, port, run_dir, cpu_cores=None):
    run_id = f"{model['size']}_{target}_{workload_name}" if target != "cpu" else f"{model['size']}_cpu{cpu_cores}_{workload_name}"
    log(f"run start {run_id}")
    for sub in ("logs", "metrics", "figures", "results"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    if target == "cpu" and cpu_cores:
        cpus = list(range(cpu_cores))
        set_tree_affinity(server_proc.pid, cpus)
        time.sleep(2)

    server_log_path = run_dir.parent / "server.log"
    server_log_offset = server_log_path.stat().st_size if server_log_path.exists() else 0

    cpu_monitor = CpuMonitor(run_id, server_proc.pid, target)
    gpu_monitor = DmonMonitor(run_id) if target.startswith("gpu") else None
    engine_monitor = EngineMonitor(run_id, port) if target.startswith("gpu") else None
    cpu_monitor.start()
    if gpu_monitor:
        gpu_monitor.start()
    if engine_monitor:
        engine_monitor.start()
    if gpu_monitor:
        time.sleep(1)

    env = base_env()
    env["VLLM_NO_USAGE_STATS"] = "1"
    if target == "cpu":
        env["VLLM_TARGET_DEVICE"] = "cpu"
        vllm_bin = CPU_VLLM
    else:
        if GPU_ENGINE == "v0":
            env["VLLM_USE_V1"] = "0"
        else:
            env.pop("VLLM_USE_V1", None)
            env["VLLM_USE_FLASHINFER_SAMPLER"] = GPU_USE_FLASHINFER_SAMPLER
        vllm_bin = GPU_VLLM
    cmd = bench_command(vllm_bin, port, served_name, model["model"], workload, run_dir)
    timeout_s = benchmark_timeout_s(model["size"])
    returncode, timed_out, duration_s, client_log, client_cmd = run_benchmark_client(
        cmd, run_dir, env, target, timeout_s
    )

    if gpu_monitor:
        time.sleep(1)
        gpu_monitor.stop()
    if engine_monitor:
        engine_monitor.stop()
    cpu_monitor.stop()

    cpu_metrics_path = run_dir / "metrics" / "cpu_metrics.csv"
    gpu_metrics_path = run_dir / "metrics" / "gpu_dmon.csv"
    engine_metrics_path = run_dir / "metrics" / "vllm_metrics.csv"
    write_csv(cpu_metrics_path, cpu_monitor.rows)
    gpu_rows = gpu_monitor.rows if gpu_monitor else []
    engine_rows = engine_monitor.rows if engine_monitor else []
    if gpu_monitor:
        write_csv(gpu_metrics_path, gpu_rows)
        (run_dir / "logs" / "gpu_dmon_raw.log").write_text("\n".join(gpu_monitor.raw_lines) + "\n")
    if engine_monitor:
        write_csv(engine_metrics_path, engine_rows)
        (run_dir / "logs" / "vllm_metrics_errors.log").write_text("\n".join(engine_monitor.errors) + "\n")

    engine_log_text = ""
    if server_log_path.exists():
        with server_log_path.open(errors="replace") as server_stream:
            server_stream.seek(server_log_offset)
            engine_log_text = server_stream.read()
    engine_log_path = run_dir / "logs" / "vllm_engine.log"
    engine_log_path.write_text(engine_log_text)
    engine_log_stats = parse_engine_log(engine_log_text)

    bench_json = run_dir / "results" / "bench_serve.json"
    bench = load_bench_json(bench_json)
    status = "passed" if returncode == 0 and bench else "failed"
    if timed_out:
        status = "timeout"
    validation_issues = validate_metrics(target, cpu_monitor.rows, gpu_rows, engine_rows)
    timeout_note = timeout_health_note(target, server_proc, cpu_monitor.rows, gpu_rows) if timed_out else None
    summary = metric_summary(cpu_monitor.rows, gpu_rows, engine_rows, engine_log_stats)
    fig_path = plot_run(run_dir, run_id, target, cpu_monitor.rows, gpu_rows)
    result = {
        "timestamp_utc": utc_now(),
        "run_id": run_id,
        "model_size": model["size"],
        "model": model["model"],
        "target": target,
        "cpu_cores": cpu_cores,
        "workload": workload_name,
        "dataset_name": workload["dataset_name"],
        "dataset_path": workload.get("dataset_path"),
        "input_len": workload.get("input_len"),
        "output_len": workload.get("output_len"),
        "num_prompts": workload["num_prompts"],
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": workload["max_num_seqs"],
        "max_concurrency": workload.get("max_concurrency"),
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "benchmark_timeout_s": timeout_s,
        "timeout_policy_applied": timeout_s is not None,
        "stop_reason": (
            f"benchmark_timeout_exceeded_{timeout_s}s"
            if timed_out
            else ("completed_or_failed_without_timeout" if timeout_s is None else "completed_or_failed_before_timeout")
        ),
        "timeout_health_check": timeout_note,
        "duration_s": duration_s,
        "validation_issues": validation_issues,
        "bench_json": str(bench_json),
        "client_log": str(client_log),
        "cpu_metrics": str(cpu_metrics_path),
        "gpu_metrics": str(gpu_metrics_path) if gpu_monitor else None,
        "engine_metrics": str(engine_metrics_path) if engine_monitor else None,
        "engine_log": str(engine_log_path),
        "figure": str(fig_path),
        "client_command": client_cmd,
        "requests_per_second": bench.get("request_throughput") if bench else None,
        "tokens_per_second": bench.get("total_token_throughput") if bench else None,
        "output_tokens_per_second": bench.get("output_throughput") if bench else None,
        "mean_ttft_ms": bench.get("mean_ttft_ms") if bench else None,
        "mean_tpot_ms": bench.get("mean_tpot_ms") if bench else None,
    }
    result.update(summary)
    write_json(run_dir / "result.json", result)
    append_jsonl(REAL / "all_results.jsonl", result)
    log(f"run done {run_id} status={status} duration_s={duration_s:.1f} issues={validation_issues}")
    return result


def cooldown(target):
    log(f"cooldown {target} {COOLDOWN_S}s")
    time.sleep(COOLDOWN_S)


def run_backend(model, target_name, model_dir, gpu_mode=None):
    cells = planned_cells(model, target_name)
    completed = []
    pending = []
    for workload_name, workload, run_dir, cores in cells:
        existing = load_existing_result(run_dir)
        if is_completed_result(existing):
            completed.append(existing)
        else:
            pending.append((workload_name, workload, run_dir, cores))

    if not pending:
        log(f"skip server model={model['size']} target={target_name}; all cells already completed")
        return completed

    port = CPU_SERVER_PORT if target_name == "cpu" else gpu_mode["port"]
    log(f"starting server model={model['size']} target={target_name} port={port}")
    server_proc, served_name, server_log, server_cmd = start_server(model, target_name, port, model_dir, gpu_mode)
    server_info = {
        "timestamp_utc": utc_now(),
        "model_size": model["size"],
        "model": model["model"],
        "target": target_name,
        "port": port,
        "served_name": served_name,
        "server_pid": server_proc.pid,
        "server_log": str(server_log),
        "server_command": server_cmd,
    }
    write_json(model_dir / target_name / "server_info.json", server_info)
    ready, reason = wait_for_server(server_proc, port, served_name, SERVER_READY_TIMEOUT_S, server_log)
    server_info["ready"] = ready
    server_info["ready_reason"] = reason
    write_json(model_dir / target_name / "server_info.json", server_info)
    results = []
    results.extend(completed)
    if not ready:
        log(f"server failed model={model['size']} target={target_name} reason={reason}")
        for workload_name, workload in WORKLOADS.items():
            if target_name == "cpu":
                for cores in CPU_CORE_COUNTS:
                    run_dir = run_dir_for(model["size"], target_name, workload_name, cores)
                    result = {
                        "timestamp_utc": utc_now(),
                        "run_id": f"{model['size']}_cpu{cores}_{workload_name}",
                        "model_size": model["size"],
                        "model": model["model"],
                        "target": "cpu",
                        "cpu_cores": cores,
                        "workload": workload_name,
                        "status": "server_failed",
                        "server_ready_reason": reason,
                        "server_log": str(server_log),
                    }
                    write_json(run_dir / "result.json", result)
                    append_jsonl(REAL / "all_results.jsonl", result)
                    results.append(result)
            else:
                run_dir = run_dir_for(model["size"], target_name, workload_name)
                result = {
                    "timestamp_utc": utc_now(),
                    "run_id": f"{model['size']}_{target_name}_{workload_name}",
                    "model_size": model["size"],
                    "model": model["model"],
                    "target": target_name,
                    "workload": workload_name,
                    "status": "server_failed",
                    "server_ready_reason": reason,
                    "server_log": str(server_log),
                }
                write_json(run_dir / "result.json", result)
                append_jsonl(REAL / "all_results.jsonl", result)
                results.append(result)
        terminate_tree(server_proc)
        cooldown(target_name)
        return results

    log(f"server ready model={model['size']} target={target_name}")
    try:
        if target_name == "cpu":
            for workload_name, workload, run_dir, cores in pending:
                results.append(run_one_cell(model, "cpu", workload_name, workload, server_proc, served_name, port, run_dir, cores))
                cooldown("cpu_loaded_server")
        else:
            for workload_name, workload, run_dir, _cores in pending:
                results.append(run_one_cell(model, target_name, workload_name, workload, server_proc, served_name, port, run_dir))
                cooldown(f"{target_name}_loaded_server")
    finally:
        log(f"stopping server model={model['size']} target={target_name}")
        terminate_tree(server_proc)
        cooldown(target_name)
    return results


def summarize_model(model_dir, results):
    write_csv(model_dir / "results.csv", results)
    write_json(
        model_dir / "summary.json",
        {
            "timestamp_utc": utc_now(),
            "runs": len(results),
            "passed": sum(1 for row in results if row.get("status") == "passed"),
            "failed": sum(1 for row in results if row.get("status") != "passed"),
            "results_csv": str(model_dir / "results.csv"),
        },
    )


def main():
    REAL.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    snapshot_path = REAL / "system_snapshot.json"
    write_json(snapshot_path, system_snapshot())
    nvidia_smi_q_path = REAL / "nvidia_smi_q.txt"
    nvidia_smi_q = capture_command(["nvidia-smi", "-q"])
    nvidia_smi_q_path.write_text(nvidia_smi_q.get("output", "") + "\n")
    manifest = {
        "timestamp_utc": utc_now(),
        "root": str(ROOT),
        "real_runs_dir": str(REAL),
        "hf_home": str(HF_HOME),
        "server_host": SERVER_HOST,
        "cpu_vllm": str(CPU_VLLM),
        "gpu_vllm": str(GPU_VLLM),
        "client_cpuset": CLIENT_CPUSET,
        "cpu_server_cpuset": cpu_server_cpuset(),
        "cpu_server_omp_threads": cpu_server_threads(),
        "cpu_server_port": CPU_SERVER_PORT,
        "models": MODELS,
        "workloads": WORKLOADS,
        "cpu_core_counts": CPU_CORE_COUNTS,
        "gpu_modes": GPU_MODES,
        "run_targets": sorted(RUN_TARGETS) if RUN_TARGETS else ["cpu", "gpu"],
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "gpu_engine": GPU_ENGINE,
        "gpu_swap_space_gb": GPU_SWAP_SPACE_GB,
        "gpu_preemption_mode": GPU_PREEMPTION_MODE,
        "gpu_kv_offloading_size_gb": GPU_KV_OFFLOADING_SIZE_GB,
        "gpu_kv_offloading_backend": GPU_KV_OFFLOADING_BACKEND,
        "gpu_kv_offload_prompt_only": GPU_KV_OFFLOAD_PROMPT_ONLY,
        "gpu_use_flashinfer_sampler": GPU_USE_FLASHINFER_SAMPLER,
        "max_model_len": MAX_MODEL_LEN,
        "system_snapshot": str(snapshot_path),
        "nvidia_smi_q": str(nvidia_smi_q_path),
        "cpu_note": "CPU model is loaded once per model on the max configured CPU core count, then server affinity is applied before each CPU-core run.",
        "gpu_note": "GPU topology is fixed at vLLM server launch, so each model is loaded once for 1GPU and once for 2GPU.",
        "transfer_note": "GPU PCIe RX/TX counters are aggregate; for 2GPU they include CPU<->GPU plus tensor-parallel GPU<->GPU traffic over PCIe/SYS.",
    }
    write_json(REAL / "run_manifest.json", manifest)
    log("real experiment driver started")
    log(f"HF_HOME={HF_HOME}")
    all_results = []
    for model in MODELS:
        model_dir = REAL / model["size"]
        model_dir.mkdir(parents=True, exist_ok=True)
        log(f"model start {model['size']} {model['model']}")
        model_results = []
        if target_enabled("cpu"):
            model_results.extend(run_backend(model, "cpu", model_dir))
        for gpu_mode in GPU_MODES:
            if target_enabled(gpu_mode["name"]):
                model_results.extend(run_backend(model, gpu_mode["name"], model_dir, gpu_mode))
        summarize_model(model_dir, model_results)
        all_results.extend(model_results)
        log(f"model done {model['size']}")
    write_csv(REAL / "all_results.csv", all_results)
    write_json(
        REAL / "summary.json",
        {
            "timestamp_utc": utc_now(),
            "runs": len(all_results),
            "passed": sum(1 for row in all_results if row.get("status") == "passed"),
            "failed": sum(1 for row in all_results if row.get("status") != "passed"),
        },
    )
    log("real experiment driver finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"fatal error: {exc!r}")
        raise
