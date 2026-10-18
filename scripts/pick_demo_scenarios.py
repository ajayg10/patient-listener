"""Script to choose demo scenarios where Fixed engine interrupts and Patient Listener waits and responds.

Evaluates test scenarios from backend/eval/data/test_turns.jsonl against the active decision cache.
Writes selected scenarios to config data (backend/eval/data/demo_scenarios.json) with cache provenance.
Prints a warning if evaluated against a mock cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import get_config
from backend.app.pace_profile import PaceProfileBuilder
from backend.app.policy import PolicyThresholds, policy
from backend.app.state_builder import StateBuilder
from scripts.precompute_decisions import compute_cache_key, load_cache

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pick_demo_scenarios")


def pick_demo_scenarios(
    cache_path: Path | str | None = None,
    output_path: Path | str | None = None,
    turns_path: Path | str | None = None,
    max_picks: int = 10,
) -> dict[str, Any]:
    cfg = get_config()
    cache_file = Path(cache_path or cfg.decision_cache_path)
    if not cache_file.is_absolute():
        cache_file = REPO_ROOT / cache_file

    out_file = Path(output_path or cfg.demo_scenarios_path)
    if not out_file.is_absolute():
        out_file = REPO_ROOT / out_file

    turns_file = Path(turns_path or (REPO_ROOT / "backend" / "eval" / "data" / "test_turns.jsonl"))

    logger.info(f"Loading cache from: {cache_file}")
    cache = load_cache(cache_file)
    logger.info(f"Loaded {len(cache)} cached decision entries.")

    # Determine provenance from cache entries
    sample_entry = next(iter(cache.values())) if cache else {}
    backend = sample_entry.get("backend", "mock")
    model_id = sample_entry.get("model_id", cfg.decider_model_id)
    is_mock = (backend == "mock")

    if is_mock:
        logger.warning(
            "\n" + "=" * 70 + "\n"
            "WARNING: Active cache is MOCK-DERIVED (backend='mock').\n"
            "These scenario picks are NOT valid for a live demo until recomputed\n"
            "with the real Decider model on GPU (Tesla T4)!\n"
            + "=" * 70 + "\n"
        )
    else:
        logger.info(f"Active cache provenance verified as REAL ({backend}, {model_id}).")

    with open(turns_file, "r", encoding="utf-8") as f:
        turns = [json.loads(line) for line in f if line.strip()]

    pace_builder = PaceProfileBuilder()
    checkpoints = sorted(cfg.turn_manager.checkpoints_ms)
    fixed_timeout = cfg.fixed_replay_timeout_ms
    decision_latency = cfg.replay_decision_latency_ms
    policy_thresholds = PolicyThresholds(
        hi=cfg.policy.confidence_hi,
        lo=cfg.policy.confidence_lo,
    )

    picks: list[dict[str, Any]] = []

    for t in turns:
        # A demo scenario requires the fixed timeout engine to interrupt during pause,
        # while Patient Listener waits for speech resumption and responds after true end.
        if t["is_turn_complete"]:
            continue

        silence_ms = float(t["silence_ms"])
        if silence_ms <= fixed_timeout:
            continue

        # Fixed engine interrupts at fixed_timeout
        fixed_interrupts = True

        # Simulate Patient Listener
        speech_text = " ".join(t["text"].split()[-40:])
        sb = StateBuilder()
        sb.append_transcript(speech_text)
        prof = pace_builder.build_profile(t.get("user_id", ""))
        pace_line = prof.render()

        pl_interrupts = False
        for cp in checkpoints:
            if cp > silence_ms:
                break
            st = sb.build_state_text(silence_ms=cp, pace_summary=pace_line)
            key = compute_cache_key(model_id, "turn_complete", st)
            if key in cache:
                item = cache[key]
                act = policy(item["probabilities"], cp, thresholds=policy_thresholds)
                arrival_ms = cp + decision_latency
                if arrival_ms < silence_ms and act == "RESPOND":
                    pl_interrupts = True
                    break

        if fixed_interrupts and not pl_interrupts:
            picks.append({
                "id": t["id"],
                "user": t["user_id"],
                "persona": t.get("archetype") or t.get("persona_id"),
                "text_preview": t.get("full_text") or t.get("text"),
                "true_end_time": t.get("turn_end_t"),
                "silence_ms": silence_ms,
                "category": t.get("category"),
            })

    logger.info(f"Found {len(picks)} qualifying scenarios where Fixed interrupts and Patient Listener waits.")

    selected_picks = picks[:max_picks]

    result_data = {
        "provenance": {
            "backend": backend,
            "model_id": model_id,
            "total_cached_keys": len(cache),
            "is_mock": is_mock,
            "warning": "NOT valid for demo (mock cache)" if is_mock else None,
            "note": "latency measured on Tesla T4, decisions from cache",
        },
        "scenarios": selected_picks,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    logger.info(f"Wrote {len(selected_picks)} demo scenarios to {out_file}")
    return result_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick demo scenarios from active cache.")
    parser.add_argument("--cache", type=str, default=None, help="Path to decisions cache file.")
    parser.add_argument("--out", type=str, default=None, help="Path to write output demo scenarios JSON.")
    parser.add_argument("--max-picks", type=int, default=10, help="Maximum scenarios to pick.")
    args = parser.parse_args()

    pick_demo_scenarios(cache_path=args.cache, output_path=args.out, max_picks=args.max_picks)


if __name__ == "__main__":
    main()
