"""Evaluation metrics with Wilson 1927 score intervals and bootstrap percentile CIs.

SYNTHETIC EVALUATION METRICS MODULE.
Formulas:
- Wilson 1927 score interval for binomial proportion p = k/n (z = 1.95996):
    center = (p + z^2 / (2n)) / (1 + z^2 / n)
    spread = (z * sqrt(p*(1-p)/n + z^2 / (4n^2))) / (1 + z^2 / n)
    CI = [max(0.0, center - spread), min(1.0, center + spread)]
- Bootstrap percentile interval (B = 1000):
    Percentile distribution of 1000 resamples with replacement taken at 2.5th and 97.5th percentiles.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float

    def format_pct(self) -> str:
        return f"[{self.low * 100:.1f}%, {self.high * 100:.1f}%]"

    def format_ms(self) -> str:
        return f"[{self.low:.1f}, {self.high:.1f}] ms"


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> ConfidenceInterval:
    """Compute the Wilson 1927 score interval for a binomial proportion."""
    if total <= 0:
        return ConfidenceInterval(0.0, 0.0)

    # Standard normal quantile for 95% confidence (1.95996)
    z = 1.95996 if abs(confidence - 0.95) < 1e-4 else 1.96

    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    spread = (z * math.sqrt((p * (1.0 - p) / total) + ((z * z) / (4.0 * total * total)))) / denom

    low = max(0.0, center - spread)
    high = min(1.0, center + spread)
    return ConfidenceInterval(round(low, 4), round(high, 4))


def bootstrap_percentile_ci(
    values: list[float],
    percentile: float = 50.0,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ConfidenceInterval:
    """Compute non-parametric bootstrap confidence interval for a percentile."""
    if not values:
        return ConfidenceInterval(0.0, 0.0)
    if len(values) == 1:
        return ConfidenceInterval(values[0], values[0])

    rng = random.Random(seed)
    n = len(values)
    boot_stats = []

    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_stats.append(float(np.percentile(sample, percentile)))

    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0

    low = float(np.percentile(boot_stats, low_pct))
    high = float(np.percentile(boot_stats, high_pct))
    return ConfidenceInterval(round(low, 2), round(high, 2))


@dataclass
class EvalSummary:
    """Summary of turn-taking performance metrics across an evaluation split."""

    name: str
    total_turns: int
    accuracy: float
    accuracy_ci: ConfidenceInterval
    premature_interruption_rate: float
    premature_interruption_ci: ConfidenceInterval
    unnecessary_wait_rate: float
    unnecessary_wait_ci: ConfidenceInterval
    response_delay_p50_ms: float
    response_delay_p50_ci: ConfidenceInterval
    response_delay_p95_ms: float
    response_delay_p95_ci: ConfidenceInterval
    decision_latency_p50_ms: float
    decision_latency_p95_ms: float
    inference_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_turns": self.total_turns,
            "accuracy": round(self.accuracy, 4),
            "accuracy_ci": [self.accuracy_ci.low, self.accuracy_ci.high],
            "premature_interruption_rate": round(self.premature_interruption_rate, 4),
            "premature_interruption_ci": [self.premature_interruption_ci.low, self.premature_interruption_ci.high],
            "unnecessary_wait_rate": round(self.unnecessary_wait_rate, 4),
            "unnecessary_wait_ci": [self.unnecessary_wait_ci.low, self.unnecessary_wait_ci.high],
            "response_delay_p50_ms": round(self.response_delay_p50_ms, 2),
            "response_delay_p50_ci": [self.response_delay_p50_ci.low, self.response_delay_p50_ci.high],
            "response_delay_p95_ms": round(self.response_delay_p95_ms, 2),
            "response_delay_p95_ci": [self.response_delay_p95_ci.low, self.response_delay_p95_ci.high],
            "decision_latency_p50_ms": round(self.decision_latency_p50_ms, 2),
            "decision_latency_p95_ms": round(self.decision_latency_p95_ms, 2),
            "inference_cost_usd": round(self.inference_cost_usd, 6),
        }


def compute_metrics(
    eval_records: list[dict[str, Any]],
    approach_name: str,
    inference_cost_per_turn: float = 0.0,
) -> EvalSummary:
    """Compute turn-taking accuracy, interruption rates, delay percentiles, and CIs."""
    total = len(eval_records)
    if total == 0:
        raise ValueError("Cannot compute metrics over an empty record list.")

    correct_count = 0
    premature_count = 0
    unnecessary_wait_count = 0
    finished_turns = 0

    response_delays: list[float] = []
    decision_latencies: list[float] = []

    for r in eval_records:
        gt_action = r["ground_truth"]  # "RESPOND" if finished else "WAIT"
        pred_action = r["predicted_action"]  # "RESPOND" or "WAIT"
        true_end_t = r["turn_end_t"]
        action_t = r["action_timestamp_ms"]
        is_turn_complete = r["is_turn_complete"]

        if pred_action == gt_action:
            correct_count += 1

        # Premature interruption: system responded before true turn completion
        if pred_action == "RESPOND" and not is_turn_complete:
            premature_count += 1

        # Unnecessary wait: turn was finished, but system waited > 800ms or chose WAIT
        if is_turn_complete:
            finished_turns += 1
            delay = action_t - true_end_t
            response_delays.append(max(0.0, delay))
            if delay > 800.0 or pred_action == "WAIT":
                unnecessary_wait_count += 1

        if "decision_latency_ms" in r and r["decision_latency_ms"] is not None:
            decision_latencies.append(float(r["decision_latency_ms"]))

    accuracy = correct_count / total
    acc_ci = wilson_score_interval(correct_count, total)

    prem_rate = premature_count / total
    prem_ci = wilson_score_interval(premature_count, total)

    unnec_rate = unnecessary_wait_count / finished_turns if finished_turns > 0 else 0.0
    unnec_ci = wilson_score_interval(unnecessary_wait_count, finished_turns) if finished_turns > 0 else ConfidenceInterval(0.0, 0.0)

    delay_p50 = float(np.median(response_delays)) if response_delays else 0.0
    delay_p50_ci = bootstrap_percentile_ci(response_delays, percentile=50.0) if response_delays else ConfidenceInterval(0.0, 0.0)

    delay_p95 = float(np.percentile(response_delays, 95.0)) if response_delays else 0.0
    delay_p95_ci = bootstrap_percentile_ci(response_delays, percentile=95.0) if response_delays else ConfidenceInterval(0.0, 0.0)

    dec_p50 = float(np.median(decision_latencies)) if decision_latencies else 0.0
    dec_p95 = float(np.percentile(decision_latencies, 95.0)) if decision_latencies else 0.0

    total_cost = total * inference_cost_per_turn

    return EvalSummary(
        name=approach_name,
        total_turns=total,
        accuracy=accuracy,
        accuracy_ci=acc_ci,
        premature_interruption_rate=prem_rate,
        premature_interruption_ci=prem_ci,
        unnecessary_wait_rate=unnec_rate,
        unnecessary_wait_ci=unnec_ci,
        response_delay_p50_ms=delay_p50,
        response_delay_p50_ci=delay_p50_ci,
        response_delay_p95_ms=delay_p95,
        response_delay_p95_ci=delay_p95_ci,
        decision_latency_p50_ms=dec_p50,
        decision_latency_p95_ms=dec_p95,
        inference_cost_usd=total_cost,
    )
