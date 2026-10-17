"""Unit tests for confidence-aware turn policy, DEV calibration, and probability reliability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.policy import PolicyThresholds, policy
from backend.eval.calibration import calibrate_policy_on_dev, compute_probability_reliability
from patient_listener_pattern.decision import (
    PolicyThresholds as PatternPolicyThresholds,
)
from patient_listener_pattern.decision import (
    policy as pattern_policy,
)
from scripts.precompute_decisions import load_cache

DATA_DIR = Path(__file__).resolve().parent.parent / "eval" / "data"


def test_single_policy_re_export_parity():
    """Verify backend.app.policy re-exports patient_listener_pattern.decision policy (single policy rule)."""
    assert policy is pattern_policy, "backend.app.policy must re-export patient_listener_pattern.decision.policy"
    assert PolicyThresholds is PatternPolicyThresholds, "PolicyThresholds must match pattern package"


def test_policy_table_all_branches():
    """Exhaustive table test covering every logical branch of the confidence-aware policy."""
    thresh = PolicyThresholds(hi=0.75, lo=0.25, max_hold_ms=2200.0)

    # 1. Cap wins: silence_ms >= max_hold_ms -> ALWAYS RESPOND (even if p=0 or Q2 conflicts)
    assert policy({"RESPOND": 0.05, "WAIT": 0.95}, silence_ms=2200.0, thresholds=thresh) == "RESPOND"
    assert policy({"RESPOND": 0.0, "WAIT": 1.0}, silence_ms=2500.0, thresholds=thresh) == "RESPOND"
    assert policy(
        {"RESPOND": 0.90, "WAIT": 0.10},
        silence_ms=2200.0,
        thresholds=thresh,
        q2_probabilities={"YES": 0.90, "NO": 0.10},  # Mid-thought conflict, but cap wins
    ) == "RESPOND"

    # 2. Cadence floor: silence_ms < 250 -> ALWAYS WAIT (even if p=0.99)
    assert policy({"RESPOND": 0.99, "WAIT": 0.01}, silence_ms=249.0, thresholds=thresh) == "WAIT"
    assert policy({"RESPOND": 0.99, "WAIT": 0.01}, silence_ms=0.0, thresholds=thresh) == "WAIT"

    # 3. Conflict rule: Q1 turn_complete >= hi (done), but Q2 mid_thought >= 0.50 -> WAIT (keep listening)
    assert policy(
        {"RESPOND": 0.85, "WAIT": 0.15},
        silence_ms=500.0,
        thresholds=thresh,
        q2_probabilities={"YES": 0.60, "NO": 0.40},
    ) == "WAIT"

    # 4. High confidence (agreement): Q1 >= hi AND Q2 mid_thought < 0.50 -> RESPOND
    assert policy(
        {"RESPOND": 0.85, "WAIT": 0.15},
        silence_ms=500.0,
        thresholds=thresh,
        q2_probabilities={"YES": 0.20, "NO": 0.80},
    ) == "RESPOND"

    # 5. High confidence without Q2 provided -> RESPOND
    assert policy({"RESPOND": 0.85, "WAIT": 0.15}, silence_ms=500.0, thresholds=thresh, q2_probabilities=None) == "RESPOND"

    # 6. Low confidence: Q1 <= lo -> WAIT
    assert policy({"RESPOND": 0.20, "WAIT": 0.80}, silence_ms=500.0, thresholds=thresh) == "WAIT"
    assert policy({"RESPOND": 0.25, "WAIT": 0.75}, silence_ms=500.0, thresholds=thresh) == "WAIT"

    # 7. Intermediate uncertainty band: lo < p < hi -> WAIT (keep waiting and re-ask at next checkpoint)
    assert policy({"RESPOND": 0.50, "WAIT": 0.50}, silence_ms=500.0, thresholds=thresh) == "WAIT"
    assert policy({"RESPOND": 0.74, "WAIT": 0.26}, silence_ms=500.0, thresholds=thresh) == "WAIT"
    assert policy({"RESPOND": 0.26, "WAIT": 0.74}, silence_ms=500.0, thresholds=thresh) == "WAIT"

    # 8. Empty / missing probabilities fallback -> WAIT
    assert policy({}, silence_ms=500.0, thresholds=thresh) == "WAIT"


def test_calibration_uses_dev_only_and_detects_contamination():
    """Assert calibration executes strictly on DEV split turns and rejects test turns."""
    dev_path = DATA_DIR / "dev_turns.jsonl"
    test_path = DATA_DIR / "test_turns.jsonl"

    assert dev_path.exists(), "dev_turns.jsonl must exist"
    assert test_path.exists(), "test_turns.jsonl must exist"

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_turns = [json.loads(line) for line in f if line.strip()]
    with open(test_path, "r", encoding="utf-8") as f:
        test_turns = [json.loads(line) for line in f if line.strip()]

    cache = load_cache()
    model_id = "StrandsAgents/strands-decider-2B-hobson-v19"
    checkpoints = [250, 500, 750, 1000, 1200, 1500, 1800, 2200]

    # 1. Clean calibration on DEV succeeds and selects valid thresholds
    thresh, stats = calibrate_policy_on_dev(dev_turns, cache, model_id, checkpoints, max_wait_ms=2200)
    assert stats["grid_size"] > 0
    assert thresh.hi > thresh.lo
    assert 0.50 <= thresh.hi <= 0.95
    assert 0.05 <= thresh.lo <= 0.50
    assert "score" in stats["dev_metrics"]

    # 2. Passing contaminated data with test turns MUST raise ValueError
    with pytest.raises(ValueError, match="Contamination detected in dev calibration"):
        calibrate_policy_on_dev(test_turns, cache, model_id, checkpoints, max_wait_ms=2200)

    # 3. Passing mixed list with a single test turn MUST also raise ValueError
    mixed_turns = dev_turns[:10] + [test_turns[0]]
    with pytest.raises(ValueError, match="Contamination detected in dev calibration"):
        calibrate_policy_on_dev(mixed_turns, cache, model_id, checkpoints, max_wait_ms=2200)


def test_probability_reliability_bins_sum_to_n():
    """Assert probability calibration bin counts sum exactly to total evaluated decisions N."""
    dev_path = DATA_DIR / "dev_turns.jsonl"
    test_path = DATA_DIR / "test_turns.jsonl"

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_turns = [json.loads(line) for line in f if line.strip()]
    with open(test_path, "r", encoding="utf-8") as f:
        test_turns = [json.loads(line) for line in f if line.strip()]

    cache = load_cache()
    model_id = "StrandsAgents/strands-decider-2B-hobson-v19"
    checkpoints = [250, 500, 750, 1000, 1200, 1500, 1800, 2200]

    # 1. Dev split reliability
    rel_dev = compute_probability_reliability(dev_turns, cache, model_id, checkpoints, n_bins=10)
    assert rel_dev["sample_count"] > 0
    assert 0.0 <= rel_dev["brier_score"] <= 1.0
    sum_dev_bins = sum(b["count"] for b in rel_dev["bins"])
    assert sum_dev_bins == rel_dev["sample_count"], (
        f"DEV bin count sum ({sum_dev_bins}) != total sample count ({rel_dev['sample_count']})"
    )

    # 2. Test split reliability
    rel_test = compute_probability_reliability(test_turns, cache, model_id, checkpoints, n_bins=10)
    assert rel_test["sample_count"] > 0
    assert 0.0 <= rel_test["brier_score"] <= 1.0
    sum_test_bins = sum(b["count"] for b in rel_test["bins"])
    assert sum_test_bins == rel_test["sample_count"], (
        f"TEST bin count sum ({sum_test_bins}) != total sample count ({rel_test['sample_count']})"
    )

    # 3. Bin edge contiguous check
    for b in rel_test["bins"]:
        low, high = b["range"]
        assert low < high
        assert 0.0 <= b["mean_predicted_prob"] <= 1.0
        assert 0.0 <= b["observed_rate"] <= 1.0
