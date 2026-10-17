"""Precompute Decider decisions over dev and test split states into a resumable JSONL cache.

SYNTHETIC EVALUATION DECISION CACHE.
Keyed by sha256 hash of (model_id, question_id, state_text).
Enforces:
- Free RAM >= 2.0 GB before local model loading
- Runtime estimation guard: halts and reports if estimated runtime > 60 minutes
- Resumable: skips keys already cached
- Supports DECIDER_BACKEND=remote (for GPU execution on Colab/Kaggle)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import psutil

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Enforce cache directories before importing torch / strands_decider
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "E:\\hf-cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TEMP"] = os.environ.get("TEMP", "E:\\tmp")
os.environ["TMP"] = os.environ.get("TMP", "E:\\tmp")

from backend.app.config import get_config
from backend.app.decider_service import (
    DeciderClient,
    Decision,
    LocalDecider,
    MockDecider,
    RemoteDecider,
)
from backend.app.questions import TURN_COMPLETE_QUESTION
from backend.app.state_builder import StateBuilder

CACHE_PATH = REPO_ROOT / "backend" / "eval" / "data" / "decisions_cache.jsonl"
DEV_DATA_PATH = REPO_ROOT / "backend" / "eval" / "data" / "dev_turns.jsonl"
TEST_DATA_PATH = REPO_ROOT / "backend" / "eval" / "data" / "test_turns.jsonl"

MEASURED_CPU_MEDIAN_S = 3.19  # Live CPU median from NOTES.md
REMOTE_GPU_ESTIMATE_S = 0.05  # Estimated remote GPU median


def compute_cache_key(model_id: str, question_id: str, state_text: str) -> str:
    raw = f"{model_id}:{question_id}:{state_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache() -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cache[item["key"]] = item
    return cache


def collect_unique_states() -> list[dict[str, Any]]:
    """Extract every unique (state_text, question) needed across all checkpoints."""
    cfg = get_config()
    model_id = cfg.decider_model_id
    checkpoints = sorted(cfg.turn_manager.checkpoints_ms)

    turns: list[dict[str, Any]] = []
    for path in [DEV_DATA_PATH, TEST_DATA_PATH]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        turns.append(json.loads(line))

    unique_states: dict[str, dict[str, Any]] = {}
    sb = StateBuilder()

    for t in turns:
        sb.clear()
        sb.append_transcript(t["text"])
        silence_ms = t["silence_ms"]

        for cp in checkpoints:
            if silence_ms >= cp:
                state_text = sb.build_state_text(silence_ms=cp)
                key = compute_cache_key(model_id, "turn_complete", state_text)
                if key not in unique_states:
                    unique_states[key] = {
                        "key": key,
                        "model_id": model_id,
                        "question_id": "turn_complete",
                        "state_text": state_text,
                        "checkpoint_ms": cp,
                    }

    return list(unique_states.values())


def run_precompute(force_backend: str | None = None) -> None:
    cfg = get_config()
    backend = force_backend or cfg.decider_backend.lower()

    unique_items = collect_unique_states()
    cache = load_cache()

    uncached = [item for item in unique_items if item["key"] not in cache]
    n_uncached = len(uncached)

    unit_time_s = REMOTE_GPU_ESTIMATE_S if backend == "remote" else MEASURED_CPU_MEDIAN_S
    estimated_seconds = n_uncached * unit_time_s
    estimated_minutes = estimated_seconds / 60.0

    print("=" * 60)
    print("Precompute Decisions Cache Status")
    print(f"  - Target backend: {backend}")
    print(f"  - Total unique (state, question) pairs: {len(unique_items)}")
    print(f"  - Already cached: {len(unique_items) - n_uncached}")
    print(f"  - Uncached keys remaining: {n_uncached}")
    print(f"  - Estimated runtime: {estimated_minutes:.1f} minutes ({estimated_seconds:.0f} seconds)")
    print("=" * 60)

    if n_uncached == 0:
        print("[precompute] All required decision keys are already cached. Nothing to compute.")
        return

    # Runtime Guard: Halt if estimated runtime exceeds 60 minutes
    if backend != "remote" and estimated_seconds > 3600.0:
        print(
            f"\n[precompute] STOPPED: Estimated runtime ({estimated_minutes:.1f} minutes) exceeds "
            f"the 60-minute threshold.\n"
            f"Halting local run. You can run precompute on a Colab/Kaggle GPU using:\n"
            f"  DECIDER_BACKEND=remote DECIDER_URL=<gpu_endpoint> python scripts/precompute_decisions.py"
        )
        return

    # Memory Guard: Check free RAM >= 2.0 GB before initializing local engine
    if backend == "local":
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < 2.0:
            print(f"[precompute] ERROR: Available RAM ({available_gb:.2f} GB) is below 2.0 GB gate. Halting.")
            sys.exit(1)

    # Initialize decider client
    client: DeciderClient
    if backend == "mock":
        client = MockDecider(fake_latency_ms=0.0)
    elif backend == "remote":
        client = RemoteDecider(url=cfg.decider_url)
    elif backend == "local":
        client = LocalDecider(cfg)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[precompute] Starting evaluation for {n_uncached} keys using {backend} decider...")

    count = 0
    t_start = time.perf_counter()

    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        for item in uncached:
            count += 1
            decision: Decision = client.decide(item["state_text"], TURN_COMPLETE_QUESTION)

            entry = {
                "key": item["key"],
                "model_id": item["model_id"],
                "question_id": item["question_id"],
                "state_text": item["state_text"],
                "checkpoint_ms": item["checkpoint_ms"],
                "choice": decision.choice,
                "probabilities": decision.probabilities,
                "confidence": decision.confidence,
                "latency_ms": decision.latency_ms,
                "backend": decision.backend,
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()

            if count % 10 == 0 or count == n_uncached:
                elapsed = time.perf_counter() - t_start
                rate = count / elapsed if elapsed > 0 else 0
                print(f"  Processed {count}/{n_uncached} keys ({rate:.2f} keys/s)...")

    print(f"[precompute] Completed {count} evaluations. Cache written to {CACHE_PATH}.")


if __name__ == "__main__":
    force_b = sys.argv[1] if len(sys.argv) > 1 else None
    run_precompute(force_backend=force_b)
