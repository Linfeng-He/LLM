#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "real_runs" / "datasets" / "opt13b_offload"
SHAREGPT_PATH = (
    ROOT
    / "real_runs"
    / "datasets"
    / "sharegpt"
    / "ShareGPT_V3_unfiltered_cleaned_split.json"
)
INSTRUCTCODER_PATH = ROOT / "real_runs" / "datasets" / "instructcoder" / "train.json"
MODEL_ID = "facebook/opt-13b"
MODEL_REVISION = "e515202d1e7750da62d245fbccb2723b9c1790f5"
REQUESTS = 22
TARGET_PROMPT_TOKENS = 251
MIN_PROMPT_TOKENS = 200
MAX_PROMPT_TOKENS = 300
REAL_OUTPUT_TOKENS = 1536
SMOKE_REQUESTS = 8
SMOKE_OUTPUT_TOKENS = 1024
MODEL_WEIGHT_GIB = 23.94
MEASURED_NON_KV_GIB = 25.05
GPU_TOTAL_GIB = 40.0
CPU_OFFLOAD_GIB = 40.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_tokens(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=True).input_ids)


def select_prompts(tokenizer, source_name: str, prompts: list[tuple[int, str]]) -> list[dict]:
    candidates = []
    for source_index, prompt in prompts:
        prompt = prompt.strip()
        if not prompt:
            continue
        token_count = prompt_tokens(tokenizer, prompt)
        if MIN_PROMPT_TOKENS <= token_count <= MAX_PROMPT_TOKENS:
            candidates.append(
                (
                    abs(token_count - TARGET_PROMPT_TOKENS),
                    source_index,
                    token_count,
                    prompt,
                )
            )
    candidates.sort(key=lambda row: (row[0], row[1]))
    if len(candidates) < REQUESTS:
        raise ValueError(f"{source_name} has only {len(candidates)} eligible prompts")
    return [
        {
            "prompt": prompt,
            "source_dataset": source_name,
            "source_index": source_index,
            "prompt_tokens": token_count,
        }
        for _distance, source_index, token_count, prompt in candidates[:REQUESTS]
    ]


def load_sharegpt_prompts() -> list[tuple[int, str]]:
    payload = json.loads(SHAREGPT_PATH.read_text())
    prompts = []
    for index, row in enumerate(payload):
        conversations = row.get("conversations") or []
        human = next(
            (
                turn.get("value", "")
                for turn in conversations
                if turn.get("from") in {"human", "user"}
            ),
            "",
        )
        prompts.append((index, human))
    return prompts


def load_instructcoder_prompts() -> list[tuple[int, str]]:
    payload = json.loads(INSTRUCTCODER_PATH.read_text())
    prompts = []
    for index, row in enumerate(payload):
        instruction = str(row.get("instruction", "")).strip()
        input_code = str(row.get("input", "")).strip()
        prompt = (
            f"Instruction:\n{instruction}\n\n"
            f"Input code:\n{input_code}\n\n"
            "Edited code:\n"
        )
        prompts.append((index, prompt))
    return prompts


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def trace_summary(path: Path, rows: list[dict], output_tokens: int, kv_bytes_per_token: int) -> dict:
    total_prompt_tokens = sum(row["prompt_tokens"] for row in rows)
    total_output_tokens = len(rows) * output_tokens
    total_tokens = total_prompt_tokens + total_output_tokens
    return {
        "path": str(path),
        "sha256": sha256(path),
        "requests": len(rows),
        "prompt_tokens": [row["prompt_tokens"] for row in rows],
        "total_prompt_tokens": total_prompt_tokens,
        "output_tokens_per_request": output_tokens,
        "total_output_tokens": total_output_tokens,
        "total_logical_kv_tokens": total_tokens,
        "logical_kv_bytes": total_tokens * kv_bytes_per_token,
        "logical_kv_gib": total_tokens * kv_bytes_per_token / (1 << 30),
    }


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    kv_bytes_per_token = (
        2
        * int(config.hidden_size)
        * int(config.num_hidden_layers)
        * 2
    )
    if kv_bytes_per_token != 819200:
        raise ValueError(f"unexpected OPT-13B KV bytes/token: {kv_bytes_per_token}")

    chat_rows = select_prompts(tokenizer, "ShareGPT", load_sharegpt_prompts())
    code_rows = select_prompts(
        tokenizer,
        "likaixin/InstructCoder train",
        load_instructcoder_prompts(),
    )
    chat_path = OUTPUT_DIR / "chat_22x1536.jsonl"
    code_path = OUTPUT_DIR / "code_22x1536.jsonl"
    smoke_path = OUTPUT_DIR / "smoke_chat_8x1024.jsonl"
    write_jsonl(chat_path, chat_rows)
    write_jsonl(code_path, code_rows)
    write_jsonl(smoke_path, chat_rows[:SMOKE_REQUESTS])

    budgets = []
    for kv_budget_gib in (5.0, 10.0, 20.0):
        executor_target_gib = MEASURED_NON_KV_GIB + kv_budget_gib
        budgets.append(
            {
                "label": f"model+{int(kv_budget_gib)}GiB_KV",
                "model_weight_gib": MODEL_WEIGHT_GIB,
                "measured_runtime_overhead_gib": MEASURED_NON_KV_GIB
                - MODEL_WEIGHT_GIB,
                "gpu_kv_budget_gib": kv_budget_gib,
                "executor_target_gib": executor_target_gib,
                "gpu_memory_utilization": executor_target_gib / GPU_TOTAL_GIB,
                "feasible": executor_target_gib <= GPU_TOTAL_GIB,
            }
        )

    manifest = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "precision": "float16",
        "max_model_len": 2048,
        "kv_bytes_per_token": kv_bytes_per_token,
        "cpu_offload_buffer_gib": CPU_OFFLOAD_GIB,
        "offload_prompt_only": False,
        "selection": {
            "requests": REQUESTS,
            "target_prompt_tokens": TARGET_PROMPT_TOKENS,
            "eligible_prompt_token_range": [MIN_PROMPT_TOKENS, MAX_PROMPT_TOKENS],
            "method": "smallest absolute distance to target, then source index",
        },
        "source_sha256": {
            "sharegpt": sha256(SHAREGPT_PATH),
            "instructcoder_train": sha256(INSTRUCTCODER_PATH),
        },
        "traces": {
            "chat": trace_summary(
                chat_path,
                chat_rows,
                REAL_OUTPUT_TOKENS,
                kv_bytes_per_token,
            ),
            "code": trace_summary(
                code_path,
                code_rows,
                REAL_OUTPUT_TOKENS,
                kv_bytes_per_token,
            ),
            "smoke": trace_summary(
                smoke_path,
                chat_rows[:SMOKE_REQUESTS],
                SMOKE_OUTPUT_TOKENS,
                kv_bytes_per_token,
            ),
        },
        "gpu_budgets": budgets,
        "planned_real_cells": 4,
        "omitted_budget": "model+20GiB_KV requires 112.625% of a 40GiB GPU",
    }
    manifest_path = OUTPUT_DIR / "trace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)
    for name, summary in manifest["traces"].items():
        print(name, f"{summary['logical_kv_gib']:.3f} GiB")
    for budget in budgets:
        print(
            budget["label"],
            f"{100 * budget['gpu_memory_utilization']:.3f}%",
            "feasible" if budget["feasible"] else "infeasible",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())