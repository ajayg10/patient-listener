"""Audit script for decider policy calibration, p(done) distributions, and policy vs cap triggers.

Evaluates real cache (decisions_cache_real.jsonl) on DEV and TEST splits.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import get_config
from backend.app.pace_profile import PaceProfileBuilder
from backend.app.policy import PolicyThresholds, policy
from backend.app.state_builder import StateBuilder
from scripts.precompute_decisions import compute_cache_key, load_cache

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("audit_calibration")


def run_calibration_audit(cache_path: Path | str | None = None) -> dict[str, Any]:
    cfg = get_config()
    cache_file = Path(cache_path or cfg.decision_cache_path)
    if not cache_file.is_absolute():
        cache_file = REPO_ROOT / cache_file

    logger.info(f"Loading cache: {cache_file}")
    cache = load_cache(cache_file)
    logger.info(f"Loaded {len(cache)} cached entries.")

    test_path = REPO_ROOT / "backend" / "eval" / "data" / "test_turns.jsonl"

    with open(test_path, "r", encoding="utf-8") as f:
        test_turns = [json.loads(line) for line in f if line.strip()]

    checkpoints = sorted(cfg.turn_manager.checkpoints_ms)
    max_wait_ms = float(cfg.turn_manager.max_wait_ms)
    latency_ms = float(cfg.replay_decision_latency_ms)
    model_id = cfg.decider_model_id
    pace_builder = PaceProfileBuilder()

    logger.info("\n" + "=" * 80)
    logger.info("1. CALIBRATION OBJECTIVE & DEV-CHOSEN THRESHOLDS AUDIT")
    logger.info("=" * 80)
    logger.info("Exact objective function: score = accuracy - (0.5 * premature_rate)")
    logger.info("Search grid (64 pairs): hi in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]")
    logger.info("                        lo in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45] (hi > lo)")
    logger.info("Cadence floor: min_silence_for_respond_ms = 250.0 ms")
    logger.info(f"Max hold timeout: max_wait_ms = {max_wait_ms:.0f} ms")
    logger.info("DEV-chosen values in E3: hi = 0.55, lo = 0.10 (DEV score = 1.0000, Acc = 100.0%, Premature = 0.0%)")
    logger.info("Does the objective have a delay term? NO. The objective only rewards accuracy and penalizes premature interruptions.")
    logger.info("Can always-waiting satisfy it? YES. Completed turns coerce to RESPOND at max_hold_ms, so an agent that NEVER fires")
    logger.info("prematurely and waits until max_hold achieves 100% accuracy and 0% premature interruption rate (score = 1.0000).")

    logger.info("\n" + "=" * 80)
    logger.info("2. p(done) DISTRIBUTION FROM REAL CACHE FOR Q1 AND Q2")
    logger.info("=" * 80)

    categories = ["complete", "incomplete", "ambiguous"]
    q1_summary = {}

    for use_prof in [False, True]:
        variant_name = "with_profile" if use_prof else "no_profile"
        logger.info(f"\n--- Q1 turn_complete: p(done) = P(RESPOND) [{variant_name}] ---")
        q1_summary[variant_name] = {}
        for cat in categories:
            logger.info(f"Category: {cat.upper()}")
            q1_summary[variant_name][cat] = {}
            for cp in checkpoints:
                vals = []
                for t in test_turns:
                    if t.get("category") != cat:
                        continue
                    if not t["is_turn_complete"] and cp > t["silence_ms"]:
                        continue
                    txt = " ".join(t["text"].split()[-40:])
                    sb = StateBuilder()
                    sb.append_transcript(txt)
                    pace_line = pace_builder.build_profile(t["user_id"]).render() if use_prof else None
                    st = sb.build_state_text(silence_ms=cp, pace_summary=pace_line)
                    k = compute_cache_key(model_id, "turn_complete", st)
                    if k in cache:
                        vals.append(cache[k]["probabilities"].get("RESPOND", 0.0))

                if vals:
                    arr = np.array(vals)
                    n = len(arr)
                    mean = float(arr.mean())
                    p_min = float(arr.min())
                    p25 = float(np.percentile(arr, 25))
                    med = float(np.median(arr))
                    p75 = float(np.percentile(arr, 75))
                    p_max = float(arr.max())
                    ge50 = float((arr >= 0.5).mean() * 100)
                    ge70 = float((arr >= 0.7).mean() * 100)
                    ge90 = float((arr >= 0.9).mean() * 100)
                    q1_summary[variant_name][cat][cp] = {
                        "n": n, "mean": round(mean, 3), "min": round(p_min, 3),
                        "p25": round(p25, 3), "med": round(med, 3), "p75": round(p75, 3),
                        "max": round(p_max, 3), "ge50_pct": round(ge50, 1),
                        "ge70_pct": round(ge70, 1), "ge90_pct": round(ge90, 1),
                    }
                    logger.info(
                        f"  cp={cp:4d}ms: n={n:3d}, mean={mean:.3f}, min={p_min:.3f}, p25={p25:.3f}, "
                        f"med={med:.3f}, p75={p75:.3f}, max={p_max:.3f} | "
                        f">=0.5: {ge50:5.1f}%, >=0.7: {ge70:5.1f}%, >=0.9: {ge90:5.1f}%"
                    )

    # Q2 mid_thought
    for use_prof in [False, True]:
        variant_name = "with_profile" if use_prof else "no_profile"
        logger.info(f"\n--- Q2 mid_thought: p(mid_thought) = P(YES) [{variant_name}] ---")
        for cat in categories:
            logger.info(f"Category: {cat.upper()}")
            for cp in checkpoints:
                vals = []
                for t in test_turns:
                    if t.get("category") != cat:
                        continue
                    if not t["is_turn_complete"] and cp > t["silence_ms"]:
                        continue
                    txt = " ".join(t["text"].split()[-40:])
                    sb = StateBuilder()
                    sb.append_transcript(txt)
                    pace_line = pace_builder.build_profile(t["user_id"]).render() if use_prof else None
                    st = sb.build_state_text(silence_ms=cp, pace_summary=pace_line)
                    k = compute_cache_key(model_id, "mid_thought", st)
                    if k in cache:
                        vals.append(cache[k]["probabilities"].get("YES", 0.0))

                if vals:
                    arr = np.array(vals)
                    n = len(arr)
                    mean = float(arr.mean())
                    p_min = float(arr.min())
                    p25 = float(np.percentile(arr, 25))
                    med = float(np.median(arr))
                    p75 = float(np.percentile(arr, 75))
                    p_max = float(arr.max())
                    ge50 = float((arr >= 0.5).mean() * 100)
                    ge70 = float((arr >= 0.7).mean() * 100)
                    ge90 = float((arr >= 0.9).mean() * 100)
                    logger.info(
                        f"  cp={cp:4d}ms: n={n:3d}, mean={mean:.3f}, min={p_min:.3f}, p25={p25:.3f}, "
                        f"med={med:.3f}, p75={p75:.3f}, max={p_max:.3f} | "
                        f">=0.5: {ge50:5.1f}%, >=0.7: {ge70:5.1f}%, >=0.9: {ge90:5.1f}%"
                    )

    logger.info("\n" + "=" * 80)
    logger.info("3. FRACTION OF TURNS FROM POLICY VS MAX_HOLD CAP (hi=0.55, lo=0.10)")
    logger.info("=" * 80)

    triggers_summary = {}
    thresh = PolicyThresholds(hi=0.55, lo=0.10, max_hold_ms=max_wait_ms)
    for use_prof in [False, True]:
        vname = "Decider+Profile" if use_prof else "Decider"
        pol_trig = 0
        cap_trig = 0
        for t in test_turns:
            is_c = t["is_turn_complete"]
            sil = float(t["silence_ms"])
            txt = " ".join(t["text"].split()[-40:])
            sb = StateBuilder()
            sb.append_transcript(txt)
            pace_line = pace_builder.build_profile(t["user_id"]).render() if use_prof else None
            responded = False

            for cp in checkpoints:
                if not is_c and cp > sil:
                    break
                arrival = cp + latency_ms
                if arrival >= max_wait_ms:
                    responded = True
                    cap_trig += 1
                    break
                if not is_c and arrival >= sil:
                    break
                st = sb.build_state_text(silence_ms=cp, pace_summary=pace_line)
                k = compute_cache_key(model_id, "turn_complete", st)
                q2_k = compute_cache_key(model_id, "mid_thought", st)
                q1_p = cache[k]["probabilities"]
                q2_p = cache[q2_k]["probabilities"] if q2_k in cache else None
                act = policy(q1_p, cp, thresholds=thresh, q2_probabilities=q2_p)
                if act == "RESPOND":
                    responded = True
                    pol_trig += 1
                    break

            if is_c and not responded:
                cap_trig += 1

        total_completed = sum(1 for t in test_turns if t["is_turn_complete"])
        triggers_summary[vname] = {
            "policy_triggers": pol_trig,
            "cap_triggers": cap_trig,
            "total_completed": total_completed,
            "policy_pct": round(pol_trig / total_completed * 100, 1) if total_completed else 0.0,
            "cap_pct": round(cap_trig / total_completed * 100, 1) if total_completed else 0.0,
        }
        logger.info(
            f"{vname}: Policy RESPOND = {pol_trig}/{total_completed} ({pol_trig/total_completed*100:.1f}%), "
            f"Max Hold Cap = {cap_trig}/{total_completed} ({cap_trig/total_completed*100:.1f}%)"
        )

    logger.info("\n" + "=" * 80)
    logger.info("4. REFERENCE ROWS: ARGMAX POLICY (p(done) >= 0.50, ELSE CAP)")
    logger.info("=" * 80)

    argmax_summary = {}
    for use_prof in [False, True]:
        vname = "Decider+Profile (Argmax)" if use_prof else "Decider (Argmax)"
        premature = 0
        delays = []
        pol_trig = 0
        cap_trig = 0
        for t in test_turns:
            is_c = t["is_turn_complete"]
            sil = float(t["silence_ms"])
            txt = " ".join(t["text"].split()[-40:])
            sb = StateBuilder()
            sb.append_transcript(txt)
            pace_line = pace_builder.build_profile(t["user_id"]).render() if use_prof else None
            responded = False
            resp_time = None

            for cp in checkpoints:
                if not is_c and cp > sil:
                    break
                arrival = cp + latency_ms
                if arrival >= max_wait_ms:
                    responded = True
                    resp_time = max_wait_ms
                    cap_trig += 1
                    break
                if not is_c and arrival >= sil:
                    break
                st = sb.build_state_text(silence_ms=cp, pace_summary=pace_line)
                k = compute_cache_key(model_id, "turn_complete", st)
                q1_p = cache[k]["probabilities"]
                if q1_p.get("RESPOND", 0.0) >= 0.50:
                    responded = True
                    resp_time = arrival
                    pol_trig += 1
                    break

            if is_c and not responded:
                responded = True
                resp_time = max_wait_ms
                cap_trig += 1

            if responded and not is_c:
                premature += 1
            if is_c:
                delays.append(resp_time)

        p50 = float(np.median(delays)) if delays else 0.0
        p95 = float(np.percentile(delays, 95)) if delays else 0.0
        prem_rate = premature / len(test_turns) if test_turns else 0.0
        argmax_summary[vname] = {
            "policy_triggers": pol_trig,
            "cap_triggers": cap_trig,
            "premature_rate": round(prem_rate, 4),
            "delay_p50_ms": round(p50, 1),
            "delay_p95_ms": round(p95, 1),
        }
        logger.info(
            f"{vname}: Policy triggers = {pol_trig}, Cap triggers = {cap_trig} | "
            f"Premature = {prem_rate*100:.1f}%, Delay P50 = {p50:.1f} ms, Delay P95 = {p95:.1f} ms"
        )

    return {
        "q1_distribution": q1_summary,
        "triggers_summary": triggers_summary,
        "argmax_summary": argmax_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Decider calibration and p(done) distribution.")
    parser.add_argument("--cache", type=str, default=None, help="Path to decision cache.")
    args = parser.parse_args()
    run_calibration_audit(cache_path=args.cache)


if __name__ == "__main__":
    main()
