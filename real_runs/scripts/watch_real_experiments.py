#!/usr/bin/env python3
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
REAL = Path(os.environ.get("REAL_RUN_DIR", ROOT / "real_runs")).expanduser()
if not REAL.is_absolute():
    REAL = ROOT / REAL
PID_FILE = REAL / "driver.pid"
DRIVER_LOG = REAL / "driver.log"
STATUS_FILE = REAL / "watchdog_status.json"
WATCHDOG_LOG = REAL / "watchdog.log"
POLL_S = int(os.environ.get("REAL_RUN_WATCHDOG_POLL_S", "60"))
EXPECTED_RUNS = int(os.environ.get("REAL_RUN_EXPECTED_RUNS", "27"))


def csv_env(name):
    return {value.strip() for value in os.environ.get(name, "").split(",") if value.strip()}


MODEL_DIRS = csv_env("REAL_RUN_MODEL_SIZES") or {"small", "medium", "large"}
TARGETS = csv_env("REAL_RUN_TARGETS")


def target_selected(target):
    if not TARGETS:
        return True
    if target in TARGETS:
        return True
    return target and target.startswith("gpu") and "gpu" in TARGETS


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_log(message):
    with WATCHDOG_LOG.open("a") as f:
        f.write(f"[{utc_now()}] {message}\n")


def write_status(payload):
    payload = {"timestamp_utc": utc_now(), **payload}
    STATUS_FILE.write_text(json.dumps(payload, indent=2) + "\n")


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_driver_pid():
    try:
        return int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def process_alive(pid):
    if pid is None:
        return False
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def stop_tree(pid):
    if pid is None:
        return
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    procs = parent.children(recursive=True) + [parent]
    for proc in procs:
        try:
            proc.send_signal(signal.SIGTERM)
        except psutil.Error:
            pass
    gone, alive = psutil.wait_procs(procs, timeout=20)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass


def stop_orphan_servers():
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
        except psutil.Error:
            continue
        if "vllm" in cmd and (" serve " in f" {cmd} " or " bench " in f" {cmd} " or "nvidia-smi dmon" in cmd):
            try:
                proc.send_signal(signal.SIGTERM)
            except psutil.Error:
                pass


def summarize_results():
    results = []
    issues = []
    for path in sorted(REAL.glob("*/*/*/result.json")):
        if path.relative_to(REAL).parts[0] not in MODEL_DIRS:
            continue
        result = read_json(path)
        if not result:
            issues.append({"path": str(path), "issue": "unreadable_result_json"})
            continue
        if not target_selected(result.get("target")):
            continue
        status = result.get("status")
        validation_issues = result.get("validation_issues") or []
        timeout_health = result.get("timeout_health_check") or {}
        unhealthy_timeout = status == "timeout" and timeout_health.get("timeout_looks_like_unhealthy_hang", False)
        bad_status = status not in {"passed", "timeout"}
        bad_validation = bool(validation_issues)
        if bad_status or unhealthy_timeout or bad_validation:
            issues.append(
                {
                    "path": str(path),
                    "run_id": result.get("run_id"),
                    "status": status,
                    "validation_issues": validation_issues,
                    "timeout_health_check": timeout_health,
                }
            )
        results.append(result)
    counts = {}
    for result in results:
        counts[result.get("status", "unknown")] = counts.get(result.get("status", "unknown"), 0) + 1
    return {
        "results_seen": len(results),
        "expected_runs": EXPECTED_RUNS,
        "counts": counts,
        "issues": issues,
    }


def driver_log_tail():
    try:
        lines = DRIVER_LOG.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return lines[-20:]


def main():
    append_log(f"watchdog started poll_s={POLL_S}")
    while True:
        pid = read_driver_pid()
        alive = process_alive(pid)
        summary = summarize_results()
        tail = driver_log_tail()
        fatal_in_log = any("fatal error:" in line.lower() for line in tail)
        status = {
            "state": "running" if alive else "driver_not_running",
            "driver_pid": pid,
            "driver_alive": alive,
            "summary": summary,
            "driver_log_tail": tail,
        }
        write_status(status)

        if summary["issues"]:
            append_log(f"halting driver due to result issue: {summary['issues'][0]}")
            stop_tree(pid)
            write_status({**status, "state": "halted_on_result_issue"})
            return

        if fatal_in_log:
            append_log("halting driver due to fatal error in driver log")
            stop_tree(pid)
            write_status({**status, "state": "halted_on_driver_fatal"})
            return

        if not alive:
            final_summary = read_json(REAL / "summary.json")
            if final_summary and final_summary.get("runs") == EXPECTED_RUNS:
                append_log("driver completed full run")
                write_status({**status, "state": "completed", "final_summary": final_summary})
                return
            append_log("driver is not alive before full completion; stopping orphan servers")
            stop_orphan_servers()
            write_status({**status, "state": "halted_driver_not_alive"})
            return

        append_log(
            f"heartbeat pid={pid} results={summary['results_seen']}/{EXPECTED_RUNS} counts={summary['counts']}"
        )
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
