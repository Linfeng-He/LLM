#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_OUTPUT_TOKENS = 32
DEFAULT_GPU_MEMORY_UTILIZATION = "0.40"
DEFAULT_BLOCK_SIZE = 16
DEFAULT_QUESTIONS_FILE = ROOT / "real_runs" / "config" / "kv_reuse_long_conversation_questions.txt"


GPU_MODES = {
    1: {"visible_devices": "0", "port_base": 18401},
    2: {"visible_devices": "0,1", "port_base": 18402},
    4: {"visible_devices": "0,1,2,3", "port_base": 18404},
}


TARGET_PROMPT_TOKENS = [
    600,
    1200,
    2000,
    3000,
    4500,
    6500,
    8500,
    10500,
    12500,
    14500,
    16500,
    18500,
    20500,
    22500,
    24500,
    26000,
    27200,
    28200,
    29000,
    29600,
    30200,
    30700,
    31100,
    31400,
    31700,
    31900,
    32000,
    32100,
]


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"[{utc_now()}] {message}", flush=True)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_questions_file(path):
    text = path.read_text()
    questions = []
    current = []
    seen_turn = False

    for line in text.splitlines():
        if line.startswith("### TURN "):
            seen_turn = True
            if current:
                question = "\n".join(current).strip()
                if question:
                    questions.append(question)
                current = []
            continue
        if seen_turn:
            current.append(line)

    if current:
        question = "\n".join(current).strip()
        if question:
            questions.append(question)

    if not questions:
        raise SystemExit(f"no questions found in {path}")
    return questions


def write_questions_file(path, turns):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Canonical user questions for the KV prefix-cache long-conversation benchmark.",
        "#",
        "# Format: each question starts after a '### TURN N' marker and continues",
        "# until the next marker. Keep this file stable when comparing TP/cache modes.",
        "# The benchmark pairs these questions with deterministic assistant stub",
        "# messages, so generated model answers do not change future prompt prefixes.",
        "",
    ]
    for turn in turns:
        user_text = turn["messages"][-1]["content"].strip()
        lines.append(f"### TURN {turn['turn_id']}")
        lines.append(user_text)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def parse_csv_ints(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_words(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def chat_token_ids(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )


def long_common_prefix(a, b):
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return idx
    return limit


def paragraph(label, ordinal):
    return (
        f"\n[{label} section {ordinal:03d}]\n"
        "We are studying one continuous vLLM inference conversation on V100 GPUs. "
        "The topic stays fixed: Qwen inference, CUDA memory pressure, tensor parallel "
        "execution, prompt prefill, automatic prefix caching, KV block hashing, cache "
        "eviction, and how to interpret cached-token measurements. "
        "Use this deterministic evidence list as part of the user query: "
        "GPU memory budget, model weight size, prompt token count, cached token count, "
        "block size, prefix-hit coverage, end-to-end latency, and server prefix-hit logs. "
        "The benchmark needs exact token-prefix reuse, so every previous turn must stay "
        "byte-for-byte stable while the new user query appends fresh analysis requests. "
    )


def fixed_assistant(turn_id):
    return (
        f"Turn {turn_id} checkpoint: keep the prior evidence unchanged, separate "
        "logical KV reuse from latency, and report cached tokens against both total "
        "prompt tokens and expected reusable prefix tokens."
    )


def grow_user_message(tokenizer, messages, target_prompt_tokens, turn_id):
    label = f"turn-{turn_id:02d}"
    user_text = (
        f"Turn {turn_id}: continue the same CUDA/vLLM KV-cache investigation. "
        "Give a concrete analysis request for this stage and preserve the exact "
        "older transcript so prefix caching can reuse it."
    )

    ordinal = 1
    while True:
        candidate_messages = messages + [{"role": "user", "content": user_text}]
        count = len(chat_token_ids(tokenizer, candidate_messages))
        if count >= target_prompt_tokens:
            break
        user_text += paragraph(label, ordinal)
        ordinal += 1

    # Trim the final synthetic section at token granularity so the schedule stays
    # close to the intended prompt length without exceeding the model cap.
    plain_ids = tokenizer.encode(user_text, add_special_tokens=False)
    low, high = 1, len(plain_ids)
    best_text = user_text
    best_count = count
    while low <= high:
        mid = (low + high) // 2
        candidate = tokenizer.decode(plain_ids[:mid])
        candidate_messages = messages + [{"role": "user", "content": candidate}]
        candidate_count = len(chat_token_ids(tokenizer, candidate_messages))
        if candidate_count <= target_prompt_tokens:
            best_text = candidate
            best_count = candidate_count
            low = mid + 1
        else:
            high = mid - 1

    if best_count < target_prompt_tokens:
        # If decode/token boundaries made the binary-search result undershoot, use
        # the untrimmed text as long as it remains valid for the model.
        candidate_messages = messages + [{"role": "user", "content": user_text}]
        candidate_count = len(chat_token_ids(tokenizer, candidate_messages))
        if candidate_count <= target_prompt_tokens + 32:
            best_text = user_text

    return best_text


def build_conversation(
    tokenizer,
    max_model_len,
    output_tokens,
    block_size,
    question_texts=None,
):
    system = (
        "You are a precise benchmark assistant. The transcript is a deterministic "
        "long conversation about vLLM KV-cache reuse on V100 GPUs. Keep all older "
        "turns stable; every new turn appends more evidence about the same topic."
    )
    messages = [{"role": "system", "content": system}]
    turns = []
    previous_ids = None

    if question_texts is None:
        turn_inputs = [(target, None) for target in TARGET_PROMPT_TOKENS]
    else:
        turn_inputs = [(None, question) for question in question_texts]

    for turn_id, (target, question_text) in enumerate(turn_inputs, start=1):
        user_text = (
            question_text
            if question_text is not None
            else grow_user_message(tokenizer, messages, target, turn_id)
        )
        request_messages = messages + [{"role": "user", "content": user_text}]
        prompt_ids = chat_token_ids(tokenizer, request_messages)
        prompt_tokens = len(prompt_ids)
        if prompt_tokens + output_tokens > max_model_len:
            raise SystemExit(
                f"turn {turn_id} prompt_tokens+output_tokens="
                f"{prompt_tokens + output_tokens} exceeds {max_model_len}"
            )

        reusable_prefix_tokens = (
            long_common_prefix(previous_ids, prompt_ids) if previous_ids is not None else 0
        )
        expected_cacheable_tokens = (reusable_prefix_tokens // block_size) * block_size
        turns.append(
            {
                "turn_id": turn_id,
                "messages": request_messages,
                "prompt_tokens_local": prompt_tokens,
                "target_prompt_tokens": target or prompt_tokens,
                "reusable_prefix_tokens": reusable_prefix_tokens,
                "expected_cacheable_tokens": expected_cacheable_tokens,
                "expected_cacheable_blocks": expected_cacheable_tokens // block_size,
                "current_user_tokens": token_count(tokenizer, user_text),
            }
        )
        previous_ids = prompt_ids
        messages = request_messages + [
            {"role": "assistant", "content": fixed_assistant(turn_id)}
        ]

    return turns


def request_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode()
    return json.loads(body) if body else {}


def request_text(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def wait_ready(host, port, timeout_s):
    deadline = time.time() + timeout_s
    url = f"http://{host}:{port}/v1/models"
    last_error = None
    while time.time() < deadline:
        try:
            return request_json(url, timeout=5)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"server did not become ready: {last_error}")


def terminate_process(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    for _ in range(30):
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def nvidia_smi_snapshot():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return [{"error": str(exc)}]
    rows = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "utilization_gpu_pct": int(parts[4]),
                }
            )
    return rows


def start_server(args, run_dir, tp_size, cache_mode, served_name, port):
    gpu_mode = GPU_MODES[tp_size]
    env = os.environ.copy()
    hf_home = Path(args.hf_home).expanduser()
    env.update(
        {
            "VLLM_USE_V1": "0",
            "VLLM_NO_USAGE_STATS": "1",
            "CUDA_VISIBLE_DEVICES": gpu_mode["visible_devices"],
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
        }
    )
    cmd = [
        args.vllm_bin,
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(port),
        "--served-model-name",
        served_name,
        "--dtype",
        "float16",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(tp_size),
        "--disable-custom-all-reduce",
        "--disable-log-requests",
        "--enable-prompt-tokens-details",
    ]
    if cache_mode == "prefix":
        cmd.append("--enable-prefix-caching")

    log_path = run_dir / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc, handle, cmd


def chat_completion(host, port, served_name, messages, output_tokens, timeout_s):
    payload = {
        "model": served_name,
        "messages": messages,
        "max_tokens": output_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
    }
    start = time.time()
    data = request_json(
        f"http://{host}:{port}/v1/chat/completions",
        payload=payload,
        timeout=timeout_s,
    )
    elapsed_s = time.time() - start
    return data, elapsed_s


def extract_cached_tokens(usage):
    details = usage.get("prompt_tokens_details") if usage else None
    if not details:
        return 0
    return int(details.get("cached_tokens") or 0)


def run_requests(args, run_dir, tp_size, cache_mode, served_name, port, turns):
    rows = []
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_jsonl = metrics_dir / "server_metrics.jsonl"

    for turn in turns:
        turn_id = turn["turn_id"]
        log(
            f"request start tp={tp_size} mode={cache_mode} turn={turn_id} "
            f"prompt_tokens={turn['prompt_tokens_local']}"
        )
        started_at = utc_now()
        try:
            data, elapsed_s = chat_completion(
                args.host,
                port,
                served_name,
                turn["messages"],
                args.output_tokens,
                args.request_timeout_s,
            )
            error = ""
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            data = {"error": body[:4000]}
            elapsed_s = None
            error = f"http_{exc.code}"
        except Exception as exc:  # noqa: BLE001
            data = {"error": str(exc)}
            elapsed_s = None
            error = type(exc).__name__

        usage = data.get("usage", {})
        cached_tokens = extract_cached_tokens(usage)
        prompt_tokens_api = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or 0)
        expected_cacheable = turn["expected_cacheable_tokens"]
        finish_reason = ""
        if data.get("choices"):
            finish_reason = data["choices"][0].get("finish_reason") or ""

        row = {
            "started_at": started_at,
            "tp_size": tp_size,
            "cache_mode": cache_mode,
            "turn_id": turn_id,
            "target_prompt_tokens": turn["target_prompt_tokens"],
            "prompt_tokens_local": turn["prompt_tokens_local"],
            "prompt_tokens_api": prompt_tokens_api,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "cached_blocks": cached_tokens // args.block_size,
            "reusable_prefix_tokens": turn["reusable_prefix_tokens"],
            "expected_cacheable_tokens": expected_cacheable,
            "expected_cacheable_blocks": turn["expected_cacheable_blocks"],
            "current_user_tokens": turn["current_user_tokens"],
            "reuse_event": int(cached_tokens > 0),
            "reuse_rate_prompt": cached_tokens / prompt_tokens_api
            if prompt_tokens_api
            else 0.0,
            "reuse_rate_expected": cached_tokens / expected_cacheable
            if expected_cacheable
            else 0.0,
            "e2e_latency_ms": round(elapsed_s * 1000, 3)
            if elapsed_s is not None
            else "",
            "tokens_per_second_e2e": total_tokens / elapsed_s
            if elapsed_s and total_tokens
            else "",
            "finish_reason": finish_reason,
            "error": error,
        }
        rows.append(row)

        with (run_dir / "responses.jsonl").open("a") as handle:
            handle.write(json.dumps({"turn": turn_id, "response": data}) + "\n")

        try:
            metrics_text = request_text(
                f"http://{args.host}:{port}/metrics",
                timeout=10,
            )
            with metrics_jsonl.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "turn_id": turn_id,
                            "captured_at": utc_now(),
                            "metrics_text": metrics_text,
                        }
                    )
                    + "\n"
                )
        except Exception as exc:  # noqa: BLE001
            with metrics_jsonl.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "turn_id": turn_id,
                            "captured_at": utc_now(),
                            "metrics_error": str(exc),
                        }
                    )
                    + "\n"
                )

        log(
            f"request done tp={tp_size} mode={cache_mode} turn={turn_id} "
            f"cached={cached_tokens} "
            f"latency_ms={row['e2e_latency_ms']} error={error or 'none'}"
        )
        if error:
            break

    write_csv(run_dir / "per_request.csv", rows)
    return rows


def summarize_run(tp_size, cache_mode, rows, started, ended):
    total_prompt = sum(int(row.get("prompt_tokens_api") or 0) for row in rows)
    total_cached = sum(int(row.get("cached_tokens") or 0) for row in rows)
    total_expected = sum(int(row.get("expected_cacheable_tokens") or 0) for row in rows)
    total_completion = sum(int(row.get("completion_tokens") or 0) for row in rows)
    latencies = [
        float(row["e2e_latency_ms"])
        for row in rows
        if row.get("e2e_latency_ms") not in {"", None}
    ]
    failed = sum(1 for row in rows if row.get("error"))
    return {
        "tp_size": tp_size,
        "cache_mode": cache_mode,
        "started_at": started,
        "ended_at": ended,
        "duration_s": round(
            datetime.fromisoformat(ended).timestamp()
            - datetime.fromisoformat(started).timestamp(),
            3,
        ),
        "requests": len(rows),
        "failed_requests": failed,
        "reuse_events": sum(int(row.get("reuse_event") or 0) for row in rows),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_cached_tokens": total_cached,
        "total_expected_cacheable_tokens": total_expected,
        "aggregate_reuse_rate_prompt": total_cached / total_prompt
        if total_prompt
        else 0.0,
        "aggregate_reuse_rate_expected": total_cached / total_expected
        if total_expected
        else 0.0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else "",
        "max_latency_ms": max(latencies) if latencies else "",
    }


def run_one(args, output_dir, tp_size, cache_mode, turns):
    port = GPU_MODES[tp_size]["port_base"] + (10 if cache_mode == "prefix" else 0)
    served_name = f"qwen1p5b-kv-tp{tp_size}-{cache_mode}"
    run_dir = output_dir / f"tp{tp_size}_{cache_mode}"
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        run_dir / "gpu_before.json",
        nvidia_smi_snapshot(),
    )
    proc, handle, cmd = start_server(args, run_dir, tp_size, cache_mode, served_name, port)
    write_json(
        run_dir / "server_command.json",
        {
            "cmd": cmd,
            "port": port,
            "served_name": served_name,
            "tp_size": tp_size,
            "cache_mode": cache_mode,
        },
    )

    started = utc_now()
    rows = []
    try:
        wait_ready(args.host, port, args.server_ready_timeout_s)
        rows = run_requests(args, run_dir, tp_size, cache_mode, served_name, port, turns)
    finally:
        terminate_process(proc)
        handle.close()
        write_json(run_dir / "gpu_after.json", nvidia_smi_snapshot())

    ended = utc_now()
    summary = summarize_run(tp_size, cache_mode, rows, started, ended)
    write_json(run_dir / "summary.json", summary)
    return rows, summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Replay a deterministic long conversation to measure vLLM KV prefix-cache reuse."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--vllm-bin",
        default=str(ROOT / ".venv-vllm-gpu-0102" / "bin" / "vllm"),
    )
    parser.add_argument(
        "--hf-home",
        default=str(ROOT / ".hf_cache" / "huggingface"),
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument(
        "--gpu-memory-utilization",
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--tp-sizes", default="1,2,4")
    parser.add_argument("--cache-modes", default="vanilla,prefix")
    parser.add_argument(
        "--questions-file",
        default=str(DEFAULT_QUESTIONS_FILE),
        help=(
            "Text file with canonical user questions. If it exists, the benchmark "
            "loads it; otherwise the deterministic generator creates it."
        ),
    )
    parser.add_argument(
        "--regenerate-questions-file",
        action="store_true",
        help="Ignore the existing questions file and rewrite it from the generator.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "real_runs" / f"kv_reuse_qwen1p5b_{utc_stamp()}"),
    )
    parser.add_argument("--server-ready-timeout-s", type=int, default=600)
    parser.add_argument("--request-timeout-s", type=int, default=1200)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate the deterministic traffic and summary files.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tp_sizes = parse_csv_ints(args.tp_sizes)
    cache_modes = parse_csv_words(args.cache_modes)
    for tp_size in tp_sizes:
        if tp_size not in GPU_MODES:
            raise SystemExit(f"unsupported tp size: {tp_size}")
    for cache_mode in cache_modes:
        if cache_mode not in {"vanilla", "prefix"}:
            raise SystemExit(f"unsupported cache mode: {cache_mode}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    questions_path = resolve_path(args.questions_file) if args.questions_file else None
    question_texts = None
    question_source = "generated"
    if questions_path and questions_path.exists() and not args.regenerate_questions_file:
        question_texts = load_questions_file(questions_path)
        question_source = "file"

    turns = build_conversation(
        tokenizer,
        args.max_model_len,
        args.output_tokens,
        args.block_size,
        question_texts=question_texts,
    )
    if questions_path and (
        question_texts is None or args.regenerate_questions_file
    ):
        write_questions_file(questions_path, turns)
        question_source = "generated_and_written"

    manifest = {
        "created_at": utc_now(),
        "model": args.model,
        "max_model_len": args.max_model_len,
        "output_tokens": args.output_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "block_size": args.block_size,
        "tp_sizes": tp_sizes,
        "cache_modes": cache_modes,
        "turn_count": len(turns),
        "questions_file": str(questions_path) if questions_path else "",
        "question_source": question_source,
        "target_note": "TP=1 workload is intentionally long enough to take about 80s or more.",
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(
        output_dir / "traffic" / "turns.json",
        [
            {
                key: value
                for key, value in turn.items()
                if key != "messages"
            }
            for turn in turns
        ],
    )
    write_csv(
        output_dir / "traffic" / "turns.csv",
        [
            {
                key: value
                for key, value in turn.items()
                if key != "messages"
            }
            for turn in turns
        ],
    )
    write_json(
        output_dir / "traffic" / "conversation_messages.json",
        [{"turn_id": turn["turn_id"], "messages": turn["messages"]} for turn in turns],
    )
    write_questions_file(output_dir / "traffic" / "questions.txt", turns)

    log(f"traffic ready at {output_dir}; turns={len(turns)}")
    log(
        "final prompt tokens="
        f"{turns[-1]['prompt_tokens_local']} output_tokens={args.output_tokens}"
    )
    if args.dry_run:
        return

    all_rows = []
    summaries = []
    for tp_size in tp_sizes:
        for cache_mode in cache_modes:
            log(f"run start tp={tp_size} mode={cache_mode}")
            rows, summary = run_one(args, output_dir, tp_size, cache_mode, turns)
            all_rows.extend(rows)
            summaries.append(summary)
            write_csv(output_dir / "summary_by_run.csv", summaries)
            write_csv(output_dir / "per_request_all.csv", all_rows)
            log(
                f"run done tp={tp_size} mode={cache_mode} "
                f"duration_s={summary['duration_s']} "
                f"reuse_rate_prompt={summary['aggregate_reuse_rate_prompt']:.4f} "
                f"reuse_rate_expected={summary['aggregate_reuse_rate_expected']:.4f}"
            )

    write_json(output_dir / "summary.json", {"runs": summaries})
    log(f"all done output_dir={output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(130)
