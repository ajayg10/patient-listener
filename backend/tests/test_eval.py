"""Unit tests for evaluation harness, baselines, metrics, and dataset splits (Mock only)."""

from __future__ import annotations

import json
from pathlib import Path

from backend.eval.baselines import FixedTimeoutBaseline, LLMJudgeBaseline
from backend.eval.metrics import (
    bootstrap_percentile_ci,
    compute_metrics,
    wilson_score_interval,
)
from scripts.precompute_decisions import compute_cache_key

DATA_DIR = Path(__file__).resolve().parent.parent / "eval" / "data"


def test_dev_test_split_zero_leakage():
    """Dev and Test splits must be strictly disjoint by user_id AND by scenario_template_id."""
    dev_path = DATA_DIR / "dev_turns.jsonl"
    test_path = DATA_DIR / "test_turns.jsonl"

    assert dev_path.exists(), "dev_turns.jsonl must exist"
    assert test_path.exists(), "test_turns.jsonl must exist"

    dev_users = set()
    dev_templates = set()
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            dev_users.add(t["user_id"])
            dev_templates.add(t["scenario_template_id"])

    test_users = set()
    test_templates = set()
    test_count = 0
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            test_count += 1
            t = json.loads(line)
            test_users.add(t["user_id"])
            test_templates.add(t["scenario_template_id"])

    # 1. Zero user overlap
    user_overlap = dev_users.intersection(test_users)
    assert not user_overlap, f"Leaked users across splits: {user_overlap}"

    # 2. Zero scenario template overlap
    template_overlap = dev_templates.intersection(test_templates)
    assert not template_overlap, f"Leaked templates across splits: {template_overlap}"

    # 3. Test set size >= 300 turns
    assert test_count >= 300, f"Expected >= 300 test turns, got {test_count}"


def test_persona_pause_distribution_conflict():
    """Verify that a 700ms pause signals finished for fast_pauser, but mid-thought for slow_pauser."""
    import yaml
    with open(DATA_DIR / "personas.yaml", "r", encoding="utf-8") as f:
        personas = yaml.safe_load(f)["archetypes"]

    fast = personas["fast_pauser"]
    slow = personas["slow_pauser"]

    pause_probe_ms = 700

    # For fast_pauser: mid-thought max is 500ms, turn-end min is 550ms.
    # Therefore, 700ms is strictly AFTER mid-thought max and INSIDE turn_end range [550, 1000]
    assert pause_probe_ms > fast["mid_thought_pause"]["max_ms"]
    assert fast["turn_end_pause"]["min_ms"] <= pause_probe_ms <= fast["turn_end_pause"]["max_ms"]

    # For slow_pauser: mid-thought range is [650, 1200]ms, turn-end min is 1200ms.
    # Therefore, 700ms is INSIDE mid-thought pause and well BEFORE turn_end min (1200ms)
    assert slow["mid_thought_pause"]["min_ms"] <= pause_probe_ms <= slow["mid_thought_pause"]["max_ms"]
    assert pause_probe_ms < slow["turn_end_pause"]["min_ms"]


def test_metrics_on_hand_built_timeline():
    """Test metrics computation, Wilson intervals, and bootstrap CIs on a known 4-turn dataset."""
    records = [
        # 1. Correct finished turn (Action: RESPOND, GT: RESPOND, Delay: 0ms)
        {
            "id": "t1",
            "ground_truth": "RESPOND",
            "predicted_action": "RESPOND",
            "turn_end_t": 1000.0,
            "action_timestamp_ms": 1000.0,
            "is_turn_complete": True,
            "decision_latency_ms": 50.0,
        },
        # 2. Premature interruption (Action: RESPOND, GT: WAIT)
        {
            "id": "t2",
            "ground_truth": "WAIT",
            "predicted_action": "RESPOND",
            "turn_end_t": 2000.0,
            "action_timestamp_ms": 800.0,
            "is_turn_complete": False,
            "decision_latency_ms": 50.0,
        },
        # 3. Unnecessary wait (Action: RESPOND, GT: RESPOND, Delay: 900ms > 800ms)
        {
            "id": "t3",
            "ground_truth": "RESPOND",
            "predicted_action": "RESPOND",
            "turn_end_t": 1000.0,
            "action_timestamp_ms": 1900.0,
            "is_turn_complete": True,
            "decision_latency_ms": 50.0,
        },
        # 4. Correct wait (Action: WAIT, GT: WAIT)
        {
            "id": "t4",
            "ground_truth": "WAIT",
            "predicted_action": "WAIT",
            "turn_end_t": 3000.0,
            "action_timestamp_ms": 2500.0,
            "is_turn_complete": False,
            "decision_latency_ms": 50.0,
        },
    ]

    summary = compute_metrics(records, "TestApproach")

    # Accuracy: 3 correct out of 4 (75%)
    assert summary.accuracy == 0.75
    assert summary.accuracy_ci.low < 0.75 < summary.accuracy_ci.high

    # Premature interruption: 1 out of 4 (25%)
    assert summary.premature_interruption_rate == 0.25

    # Unnecessary wait: 1 out of 2 finished turns (50%)
    assert summary.unnecessary_wait_rate == 0.50

    # Delays for finished turns: [0.0, 900.0] -> Median is 450.0 ms
    assert summary.response_delay_p50_ms == 450.0


def test_wilson_score_interval_formula():
    """Verify Wilson 1927 score interval matches reference calculation."""
    # 16/40 = 0.40 -> Wilson 1927 95% interval: [26.4%, 55.4%]
    ci = wilson_score_interval(16, 40)
    assert 0.26 <= ci.low <= 0.27
    assert 0.55 <= ci.high <= 0.56

    # 34/40 = 0.85 -> Wilson 1927 95% interval: [70.9%, 92.9%]
    ci_dec = wilson_score_interval(34, 40)
    assert 0.70 <= ci_dec.low <= 0.72
    assert 0.92 <= ci_dec.high <= 0.94


def test_bootstrap_ci():
    """Verify bootstrap percentile confidence interval produces valid bounds."""
    data = [100.0, 200.0, 300.0, 400.0, 500.0]
    ci = bootstrap_percentile_ci(data, percentile=50.0, n_resamples=500)
    assert ci.low <= 300.0 <= ci.high


def test_decision_cache_key():
    """Verify compute_cache_key produces deterministic sha256 hashes."""
    k1 = compute_cache_key("model_a", "q_turn", "Speaker: \"Hello\"\n[Silence: 500 ms]")
    k2 = compute_cache_key("model_a", "q_turn", "Speaker: \"Hello\"\n[Silence: 500 ms]")
    k3 = compute_cache_key("model_b", "q_turn", "Speaker: \"Hello\"\n[Silence: 500 ms]")

    assert k1 == k2
    assert len(k1) == 64  # sha256 hex string
    assert k1 != k3


def test_fixed_and_llm_judge_baselines():
    """Verify FixedTimeoutBaseline logic and LLMJudgeBaseline mock plumbing."""
    b600 = FixedTimeoutBaseline(600)
    assert b600.decide(500) == "WAIT"
    assert b600.decide(600) == "RESPOND"
    assert b600.decide(800) == "RESPOND"

    judge = LLMJudgeBaseline(is_mock=True)
    assert judge.is_available() is True
    assert judge.decide("Speaker: \"Thank you.\"", silence_ms=800) == "RESPOND"
    assert judge.decide("Speaker: \"Wait a second...\"", silence_ms=300) == "WAIT"


def test_text_heuristic_baseline():
    """Verify TextHeuristicBaseline incomplete detection and threshold gating."""
    from backend.eval.baselines import TextHeuristicBaseline

    baseline = TextHeuristicBaseline(threshold_ms=250)

    # Incomplete phrases
    assert baseline.is_incomplete("I called the pharmacy yesterday about my refill, and...") is True
    assert baseline.is_incomplete("The nurse mentioned the panel might take longer, because...") is True
    assert baseline.is_incomplete("Checking on the status of my prescription refill") is True  # No punctuation
    assert baseline.is_incomplete("I need my lisinopril refill um...") is True

    # Complete phrases
    assert baseline.is_incomplete("Can I get a refill on my blood pressure medication please?") is False
    assert baseline.is_incomplete("Has the doctor reviewed my routine blood work from Wednesday?") is False
    assert baseline.is_incomplete("Checking on the status of my prescription refill.") is False

    # Decisions with silence gating
    assert baseline.decide("Can I get a refill please?", silence_ms=200) == "WAIT"
    assert baseline.decide("Can I get a refill please?", silence_ms=250) == "RESPOND"
    assert baseline.decide("Can I get a refill, and...", silence_ms=1000) == "WAIT"


def test_dev_split_size():
    """Dev split must have at least 150 turns."""
    dev_path = DATA_DIR / "dev_turns.jsonl"
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_count = sum(1 for line in f if line.strip())
    assert dev_count >= 150, f"Expected >= 150 dev turns, got {dev_count}"

