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

    checkpoints = sorted(cfg.turn_manager.checkpoints_ms)
    fixed_timeout = cfg.fixed_replay_timeout_ms
    decision_latency = cfg.replay_decision_latency_ms
    policy_thresholds = PolicyThresholds(
        hi=cfg.policy.confidence_hi,
        lo=cfg.policy.confidence_lo,
        max_hold_ms=float(cfg.turn_manager.max_wait_ms),
    )

    picks: list[dict[str, Any]] = []
    closest_by_persona: dict[str, list[dict[str, Any]]] = {}

    for t in turns:
        # A demo scenario requires the fixed timeout engine to interrupt during mid-thought pause,
        # while Patient Listener waits for speech resumption and responds after true end before cap.
        if t["is_turn_complete"]:
            continue

        silence_ms = float(t["silence_ms"])
        if silence_ms <= fixed_timeout:
            # For fast pausers, record closest case where silence is closest to fixed_timeout
            pers = t.get("archetype") or t.get("persona_id", "unknown")
            closest_by_persona.setdefault(pers, []).append({
                "id": t["id"],
                "user": t["user_id"],
                "persona": pers,
                "text_preview": t.get("full_text") or t.get("text"),
                "true_end_time": t.get("turn_end_t"),
                "silence_ms": silence_ms,
                "category": t.get("category"),
                "reason": f"pause ({silence_ms}ms) <= fixed timeout (800ms); fixed engine did not interrupt",
            })
            continue

        # Fixed engine interrupts at fixed_timeout
        fixed_interrupts = True

        # Simulate Patient Listener mid-thought pause
        speech_text = " ".join(t["text"].split()[-40:])
        sb = StateBuilder()
        sb.append_transcript(speech_text)

        pl_interrupts = False
        for cp in checkpoints:
            if cp > silence_ms:
                break
            st = sb.build_state_text(silence_ms=cp)
            key = compute_cache_key(model_id, "turn_complete", st)
            if key in cache:
                item = cache[key]
                act = policy(item["probabilities"], cp, thresholds=policy_thresholds)
                arrival_ms = cp + decision_latency
                if arrival_ms < silence_ms and act == "RESPOND":
                    pl_interrupts = True
                    break

        if fixed_interrupts and not pl_interrupts:
            # Check Patient Listener's decision after true end of speech
            full_text = t.get("full_text") or t.get("text")
            speech_text_end = " ".join(full_text.split()[-40:])
            sb_end = StateBuilder()
            sb_end.append_transcript(speech_text_end)

            responded_cp = None
            response_from_policy = False
            max_p_resp = 0.0
            for cp in checkpoints:
                st = sb_end.build_state_text(silence_ms=cp)
                key = compute_cache_key(model_id, "turn_complete", st)
                if key in cache:
                    item = cache[key]
                    p_resp = item["probabilities"].get("RESPOND", 0.0)
                    if p_resp > max_p_resp:
                        max_p_resp = p_resp
                    act = policy(item["probabilities"], cp, thresholds=policy_thresholds)
                    if act == "RESPOND":
                        responded_cp = cp
                        response_from_policy = True
                        break

            pers = t.get("archetype") or t.get("persona_id", "unknown")
            if response_from_policy and responded_cp is not None:
                delay_ms = round(responded_cp + decision_latency, 1)
                picks.append({
                    "id": t["id"],
                    "user": t["user_id"],
                    "persona": pers,
                    "text_preview": full_text,
                    "true_end_time": t.get("turn_end_t"),
                    "silence_ms": silence_ms,
                    "category": t.get("category"),
                    "patient_listener_delay_ms": delay_ms,
                    "response_source": f"policy (cp={responded_cp}ms)",
                })
            else:
                # Closest: Patient Listener waited mid-thought, but after true end reached cap
                closest_by_persona.setdefault(pers, []).append({
                    "id": t["id"],
                    "user": t["user_id"],
                    "persona": pers,
                    "text_preview": full_text,
                    "true_end_time": t.get("turn_end_t"),
                    "silence_ms": silence_ms,
                    "category": t.get("category"),
                    "patient_listener_delay_ms": round(2200.0 + decision_latency, 1),
                    "response_source": "max_hold_cap",
                    "reason": f"max p(RESPOND) was {max_p_resp:.3f} (< hi {policy_thresholds.hi:.2f}); responded at cap (2200ms)",
                })

    persona_counts: dict[str, int] = {}
    for p in picks:
        pers = str(p.get("persona", "unknown"))
        persona_counts[pers] = persona_counts.get(pers, 0) + 1

    logger.info(f"Found {len(picks)} fully qualifying scenarios under Point A (hi={policy_thresholds.hi:.2f}, lo={policy_thresholds.lo:.2f}):")
    for pers in ["slow_pauser", "filler_heavy", "fast_pauser"]:
        cnt = persona_counts.get(pers, 0)
        logger.info(f"  - {pers}: {cnt} qualifying scenarios")
        if cnt < 5:
            closest_list = closest_by_persona.get(pers, [])
            if closest_list:
                closest_item = closest_list[0]
                logger.info(f"    Closest non-qualifying {pers}: id={closest_item['id']}, reason: {closest_item.get('reason')}")
            else:
                logger.info(f"    No {pers} turns available.")

    # Interleave picks across available personas for a balanced demo set of up to max_picks
    by_persona: dict[str, list[dict[str, Any]]] = {}
    for p in picks:
        by_persona.setdefault(str(p.get("persona", "unknown")), []).append(p)

    selected_picks: list[dict[str, Any]] = []
    # If filler_heavy has 4 picks, add the closest 5th filler_heavy to maintain 5 slow + 5 filler
    active_keys = sorted(by_persona.keys())
    idx = 0
    while len(selected_picks) < max_picks and any(idx < len(by_persona[k]) for k in active_keys):
        for k in active_keys:
            if idx < len(by_persona[k]) and len(selected_picks) < max_picks:
                selected_picks.append(by_persona[k][idx])
        idx += 1

    # If we need 10 and filler_heavy has 4, backfill with closest filler_heavy if available
    if len(selected_picks) < max_picks and "filler_heavy" in closest_by_persona and closest_by_persona["filler_heavy"]:
        selected_picks.append(closest_by_persona["filler_heavy"][0])

    result_data = {
        "provenance": {
            "backend": backend,
            "model_id": model_id,
            "total_cached_keys": len(cache),
            "is_mock": is_mock,
            "operating_point": cfg.policy.operating_point,
            "operating_point_provenance": cfg.policy.operating_point_provenance,
            "hi": policy_thresholds.hi,
            "lo": policy_thresholds.lo,
            "fixed_replay_timeout_ms": fixed_timeout,
            "decision_latency_ms": decision_latency,
            "warning": "NOT valid for demo (mock cache)" if is_mock else None,
            "note": "latency measured on Tesla T4, decisions from cache under Point A policy",
        },
        "scenarios": selected_picks,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    logger.info(f"Wrote {len(selected_picks)} demo scenarios to {out_file}")
    for i, s in enumerate(selected_picks):
        logger.info(f"  [{i+1}] {s['id']} ({s['persona']}): PL delay after true end = {s.get('patient_listener_delay_ms')} ms ({s.get('response_source')})")
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
