"""Benchmark script for Strands Decider (CPU/GPU standalone benchmark, NEVER run in pytest).

Evaluates warm inference latency (median and p95) across:
- State sizes: ~20, ~60, ~150 tokens
- Single question (turn_complete) vs Dual questions (turn_complete + mid_thought) in one call.
- Enforces strict RAM gate (>= 2 GB free) before model loading.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Read HF_HOME from environment or config, never hardcoding a drive letter
if "HF_HOME" not in os.environ:
    from backend.app.config import get_config
    _cfg_hf = get_config().hf_home
    if _cfg_hf:
        os.environ["HF_HOME"] = _cfg_hf

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if "HF_HUB_OFFLINE" not in os.environ:
    from backend.app.config import get_config
    if getattr(get_config(), "hf_offline", False):
        os.environ["HF_HUB_OFFLINE"] = "1"

if os.name == "nt" and Path("E:/tmp").exists():
    os.environ.setdefault("TEMP", "E:\\tmp")
    os.environ.setdefault("TMP", "E:\\tmp")

from backend.app.config import get_config
from backend.app.questions import MID_THOUGHT_QUESTION, TURN_COMPLETE_QUESTION


def check_ram_gate(required_gb: float = 2.0) -> None:
    available_bytes = psutil.virtual_memory().available
    available_gb = available_bytes / (1024**3)
    print(f"[bench_decider] Available system RAM: {available_gb:.2f} GB (Gate: {required_gb:.2f} GB)")
    if available_gb < required_gb:
        print(
            f"[bench_decider] ERROR: Available RAM ({available_gb:.2f} GB) is below required "
            f"{required_gb:.2f} GB gate. Halting benchmark."
        )
        sys.exit(1)


def build_state_text(target_words: int, speech: str = "Can you transfer this call over to", silence_ms: int = 700) -> str:
    prefix = ""
    if target_words > 20:
        fillers = (
            "Context: Customer calling clinical intake support regarding previous claims, "
            "prescription refills, and appointment booking. The dialogue has been ongoing for 3 turns. "
        )
        words = fillers.split()
        needed = max(0, target_words - len(speech.split()) - 6)
        prefix = " ".join(words[:needed]) + " "
    return f"{prefix}Speaker: \"{speech}\"\n[Pause silence: {silence_ms} ms]\nPace profile: average pause 650 ms."


def benchmark() -> None:
    check_ram_gate(2.0)

    cfg = get_config()
    import torch
    from strands_decider.infer import load_engine

    torch.set_num_threads(cfg.decider_torch_threads)
    print(f"[bench_decider] PyTorch threads set to: {torch.get_num_threads()}")
    def resolve_checkpoint(checkpoint_id: str) -> str:
        if os.path.isdir(checkpoint_id):
            return checkpoint_id
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(checkpoint_id, local_files_only=True)
        except Exception:
            return checkpoint_id

    ckpt_path = resolve_checkpoint(cfg.decider_model_id)
    print(f"[bench_decider] Loading engine from {ckpt_path} on {cfg.decider_device}...")

    t0_load = time.perf_counter()
    engine = load_engine(ckpt_path, device=cfg.decider_device)
    load_duration = time.perf_counter() - t0_load
    print(f"[bench_decider] Engine loaded in {load_duration:.2f} s")

    sizes = [20, 60, 150]
    repeats = 5  # 5 warm runs after warm-up

    q1_dict = {"turn": TURN_COMPLETE_QUESTION}
    q2_dict = {"turn": TURN_COMPLETE_QUESTION, "mid": MID_THOUGHT_QUESTION}

    results = []

    print("\n--- Running Inference Latency Benchmark ---")
    for sz in sizes:
        state = build_state_text(sz)

        # 1. Single Question Benchmark
        # Warm-up run
        _ = engine.ask(state, q1_dict)
        durations_1q = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = engine.ask(state, q1_dict)
            durations_1q.append((time.perf_counter() - t0) * 1000.0)

        med_1q = float(np.median(durations_1q))
        p95_1q = float(np.percentile(durations_1q, 95))

        # 2. Dual Question Benchmark (same call)
        # Warm-up run
        _ = engine.ask(state, q2_dict)
        durations_2q = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = engine.ask(state, q2_dict)
            durations_2q.append((time.perf_counter() - t0) * 1000.0)

        med_2q = float(np.median(durations_2q))
        p95_2q = float(np.percentile(durations_2q, 95))

        overhead_ms = med_2q - med_1q

        results.append({
            "size": sz,
            "med_1q": round(med_1q, 1),
            "p95_1q": round(p95_1q, 1),
            "med_2q": round(med_2q, 1),
            "p95_2q": round(p95_2q, 1),
            "overhead_ms": round(overhead_ms, 1),
        })
        print(f"Tokens ~{sz}: 1Q med={med_1q:.1f}ms | 2Q med={med_2q:.1f}ms (overhead: {overhead_ms:+.1f}ms)")

    device_str = cfg.decider_device
    device_detail = device_str.upper()
    if device_str.startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_detail = f"CUDA (GPU: {gpu_name})"
    elif device_str == "cpu":
        device_detail = "CPU"

    model_dtype = "unknown"
    if hasattr(engine, "model") and hasattr(engine.model, "dtype"):
        model_dtype = str(engine.model.dtype).replace("torch.", "")
    elif hasattr(engine, "dtype"):
        model_dtype = str(engine.dtype).replace("torch.", "")

    # Print Formatted Markdown Table
    print(f"\n### Benchmark Results Table (Measured on {device_detail}, dtype: {model_dtype})")
    print("| State Tokens | 1 Question Median (ms) | 1 Question P95 (ms) | 2 Questions Median (ms) | 2 Questions P95 (ms) | 2Q Overhead (ms) |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['size']} tokens | {r['med_1q']} | {r['p95_1q']} | {r['med_2q']} | {r['p95_2q']} | {r['overhead_ms']} |")


if __name__ == "__main__":
    benchmark()
