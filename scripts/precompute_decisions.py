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

import gzip
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

# Read HF_HOME from environment or config, never hardcoding a drive letter
if "HF_HOME" not in os.environ:
    from backend.app.config import get_config
    _cfg_hf = get_config().hf_home
    if _cfg_hf:
        os.environ["HF_HOME"] = _cfg_hf

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if os.name == "nt" and Path("E:/tmp").exists():
    os.environ.setdefault("TEMP", "E:\\tmp")
    os.environ.setdefault("TMP", "E:\\tmp")

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

CACHE_PATH = REPO_ROOT / "backend" / "eval" / "data" / "decisions_cache_real.jsonl"
DEV_DATA_PATH = REPO_ROOT / "backend" / "eval" / "data" / "dev_turns.jsonl"
TEST_DATA_PATH = REPO_ROOT / "backend" / "eval" / "data" / "test_turns.jsonl"

MEASURED_CPU_MEDIAN_S = 3.19  # Live CPU median from NOTES.md
REMOTE_GPU_ESTIMATE_S = 0.05  # Estimated remote GPU median


def compute_cache_key(model_id: str, question_id: str, state_text: str) -> str:
    raw = f"{model_id}:{question_id}:{state_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(cache_path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = cache_path or CACHE_PATH
    if isinstance(path, str):
        path = Path(path)

    # If the plain file does not exist, check for .gz archive
    if not path.exists():
        gz_cand = Path(str(path) + ".gz")
        if gz_cand.exists():
            path = gz_cand
        elif path.suffix == ".jsonl":
            alt_gz = path.with_suffix(".jsonl.gz")
            if alt_gz.exists():
                path = alt_gz

    cache: dict[str, dict[str, Any]] = {}
    if path.exists():
        is_gz = path.suffix == ".gz" or str(path).endswith(".jsonl.gz")
        open_fn = gzip.open if is_gz else open
        with open_fn(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cache[item["key"]] = item
    return cache


def derive_resumed_text(turn: dict[str, Any]) -> str | None:
    """Derive full transcript after speech resumed for incomplete or continued turns."""
    if turn.get("full_text"):
        return str(turn["full_text"])
    if turn.get("continuation_text"):
        return f"{turn['text']} {turn['continuation_text']}"
    if turn.get("resumed_text"):
        return f"{turn['text']} {turn['resumed_text']}"

    # If turn is incomplete, speaker paused mid-thought and resumes with a completed thought
    if not turn.get("is_turn_complete", True):
        raw = turn["text"].rstrip(". ")
        if raw.endswith((", and", ", but", ", so", ", because")):
            return f"{raw} could you please check on that for me?"
        return f"{raw}... could you verify this for me?"

    return None


from backend.app.pace_profile import PaceProfileBuilder
from backend.app.questions import QUESTIONS


def collect_unique_states(
    include_speech_resumed: bool = True,
    include_profiles: bool = True,
    include_q2: bool = True,
) -> list[dict[str, Any]]:
    """Extract every unique (state_text, question) needed across all checkpoints,

    including base prefixes, profile variants, Q2 mid_thought, states after speech resumed, and B0 cases.
    """
    cfg = get_config()
    model_id = cfg.decider_model_id
    checkpoints = sorted(cfg.turn_manager.checkpoints_ms)

    turns: list[dict[str, Any]] = []
    for path in [DEV_DATA_PATH, TEST_DATA_PATH]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        t_data = json.loads(line)
                        t_data["_split"] = "dev" if "dev" in path.name else "test"
                        turns.append(t_data)

    unique_states: dict[str, dict[str, Any]] = {}
    sb = StateBuilder()
    pace_builder = PaceProfileBuilder()

    def add_state(st: str, cp: int, kind: str, split: str) -> None:
        q_ids = ["turn_complete", "mid_thought"] if include_q2 else ["turn_complete"]
        for q_id in q_ids:
            key = compute_cache_key(model_id, q_id, st)
            if key not in unique_states:
                unique_states[key] = {
                    "key": key,
                    "model_id": model_id,
                    "question_id": q_id,
                    "state_text": st,
                    "checkpoint_ms": cp,
                    "kind": kind,
                    "split": split,
                }

    for t in turns:
        # 1. Base prefix states
        sb.clear()
        sb.append_transcript(t["text"])
        for cp in checkpoints:
            st = sb.build_state_text(silence_ms=cp)
            add_state(st, cp, "prefix", t["_split"])

        # 2. Base speech-resumed states
        if include_speech_resumed:
            resumed_text = derive_resumed_text(t)
            if resumed_text and resumed_text != t["text"]:
                sb.clear()
                sb.append_transcript(resumed_text)
                for cp in checkpoints:
                    st = sb.build_state_text(silence_ms=cp)
                    add_state(st, cp, "speech_resumed", t["_split"])

        # 3. Pace profile states (for users with history)
        if include_profiles:
            uid = t.get("user_id", "")
            prof = pace_builder.build_profile(uid)
            rendered = prof.render()
            if rendered:
                sb.clear()
                sb.append_transcript(t["text"])
                for cp in checkpoints:
                    st = sb.build_state_text(silence_ms=cp, pace_summary=rendered)
                    add_state(st, cp, "prefix_profile", t["_split"])

                if include_speech_resumed:
                    resumed_text = derive_resumed_text(t)
                    if resumed_text and resumed_text != t["text"]:
                        sb.clear()
                        sb.append_transcript(resumed_text)
                        for cp in checkpoints:
                            st = sb.build_state_text(silence_ms=cp, pace_summary=rendered)
                            add_state(st, cp, "speech_resumed_profile", t["_split"])

    # 4. Also precompute B0 feasibility cases (each case has its own silence_ms)
    b0_path = REPO_ROOT / "backend" / "eval" / "data" / "b0_cases.jsonl"
    if b0_path.exists():
        with open(b0_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    case = json.loads(line)
                    sb.clear()
                    sb.append_transcript(case["text"])
                    st = sb.build_state_text(silence_ms=case["silence_ms"])
                    add_state(st, case["silence_ms"], "b0_case", "b0")

    return list(unique_states.values())


def enumerate_states_report() -> None:
    """List every state the engine could query and print runtime estimates."""
    MEASURED_T4_1Q_MEDIAN_S = 0.2356  # 235.6 ms measured on Tesla T4 GPU in Colab

    unique_items = collect_unique_states(include_speech_resumed=True)
    cache = load_cache()

    n_total = len(unique_items)
    uncached = [item for item in unique_items if item["key"] not in cache]
    n_uncached = len(uncached)
    n_cached = n_total - n_uncached

    n_prefix = sum(1 for item in unique_items if item.get("kind") == "prefix")
    n_resumed = sum(1 for item in unique_items if item.get("kind") == "speech_resumed")
    n_b0 = sum(1 for item in unique_items if item.get("kind") == "b0_case")
    dev_states = sum(1 for item in unique_items if item.get("split") == "dev")
    test_states = sum(1 for item in unique_items if item.get("split") == "test")

    est_sec_uncached = n_uncached * MEASURED_T4_1Q_MEDIAN_S
    est_sec_total = n_total * MEASURED_T4_1Q_MEDIAN_S

    print("=" * 70)
    print("FULL STATE ENUMERATION REPORT (Real Decider Precompute Coverage)")
    print("=" * 70)
    print(f"Total Unique States:               {n_total}")
    print(f"  - Initial Utterance Prefixes:    {n_prefix} states (checkpoints [250, 500, 800, 1200, 1800] ms)")
    print(f"  - Post Speech-Resumed States:    {n_resumed} states (after speaker resumes and completes)")
    print(f"  - B0 Feasibility Cases:          {n_b0} states")
    print(f"  - Dev Split States:              {dev_states}")
    print(f"  - Test Split States:             {test_states}")
    print("-" * 70)
    print("Cache Status:")
    print(f"  - Already Cached:                {n_cached} states")
    print(f"  - Uncached Keys Remaining:       {n_uncached} states")
    print("-" * 70)
    print("Estimated Runtime (Tesla T4 GPU, real Decider 1Q median = 235.6 ms):")
    print(f"  - For Uncached Keys ({n_uncached}):    {est_sec_uncached:.1f} s ({est_sec_uncached / 60.0:.2f} min)")
    print(f"  - For All States ({n_total}):       {est_sec_total:.1f} s ({est_sec_total / 60.0:.2f} min)")
    print("=" * 70)


def run_precompute(force_backend: str | None = None) -> None:
    cfg = get_config()
    backend = force_backend or cfg.decider_backend.lower()

    unique_items = collect_unique_states(include_speech_resumed=True)
    cache = load_cache()

    uncached = [item for item in unique_items if item["key"] not in cache]
    n_uncached = len(uncached)

    if backend == "mock":
        unit_time_s = 0.001
    elif backend == "remote":
        unit_time_s = REMOTE_GPU_ESTIMATE_S
    else:
        unit_time_s = MEASURED_CPU_MEDIAN_S
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
            question = QUESTIONS.get(item["question_id"], TURN_COMPLETE_QUESTION)
            decision: Decision = client.decide(item["state_text"], question)

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
    if "--enumerate" in sys.argv:
        enumerate_states_report()
    else:
        force_b = None
        for arg in sys.argv[1:]:
            if not arg.startswith("--"):
                force_b = arg
                break
        run_precompute(force_backend=force_b)

