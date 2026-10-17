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


def cluster_bootstrap_ci(
    records: list[dict[str, Any]],
    metric_type: str,
    cluster_key: str = "user_id",
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ConfidenceInterval:
    """Non-parametric two-stage cluster bootstrap interval.

    Procedure (resample users with replacement, then turns within each sampled user):
    1. Group records into K distinct user clusters C_1, ..., C_K.
    2. For b in 1..B (B=1000 resamples):
       a. Sample K user clusters with replacement: C*_1, ..., C*_K.
       b. Within each resampled cluster C*_k, sample |C*_k| turns with replacement.
       c. Compute metric theta*(b) on the pooled sample.
    3. Return [percentile(theta*, 2.5), percentile(theta*, 97.5)].
    """
    if not records:
        return ConfidenceInterval(0.0, 0.0)

    clusters: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        uid = r.get(cluster_key, "default")
        clusters.setdefault(uid, []).append(r)

    cluster_keys = list(clusters.keys())
    k = len(cluster_keys)
    if k <= 1:
        if metric_type == "accuracy":
            correct = sum(1 for r in records if r["predicted_action"] == r["ground_truth"])
            return wilson_score_interval(correct, len(records))
        elif metric_type == "premature":
            prem = sum(1 for r in records if r["predicted_action"] == "RESPOND" and not r["is_turn_complete"])
            return wilson_score_interval(prem, len(records))
        elif metric_type == "unnecessary_wait":
            completed = [r for r in records if r["is_turn_complete"]]
            if not completed:
                return ConfidenceInterval(0.0, 0.0)
            delayed = sum(1 for r in completed if (r["action_timestamp_ms"] - r["turn_end_t"] > 800.0 or r["predicted_action"] == "WAIT"))
            return wilson_score_interval(delayed, len(completed))

    rng = random.Random(seed)
    boot_stats: list[float] = []

    for _ in range(n_resamples):
        sampled: list[dict[str, Any]] = []
        for _ in range(k):
            chosen = rng.choice(cluster_keys)
            user_recs = clusters[chosen]
            n_recs = len(user_recs)
            sampled.extend([rng.choice(user_recs) for _ in range(n_recs)])

        n_tot = len(sampled)
        if metric_type == "accuracy":
            correct = sum(1 for r in sampled if r["predicted_action"] == r["ground_truth"])
            stat = correct / n_tot if n_tot > 0 else 0.0
        elif metric_type == "premature":
            prem = sum(1 for r in sampled if r["predicted_action"] == "RESPOND" and not r["is_turn_complete"])
            stat = prem / n_tot if n_tot > 0 else 0.0
        elif metric_type == "unnecessary_wait":
            completed = [r for r in sampled if r["is_turn_complete"]]
            if not completed:
                stat = 0.0
            else:
                delayed = sum(1 for r in completed if (r["action_timestamp_ms"] - r["turn_end_t"] > 800.0 or r["predicted_action"] == "WAIT"))
                stat = delayed / len(completed)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        boot_stats.append(stat)

    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0
    low = float(np.percentile(boot_stats, low_pct))
    high = float(np.percentile(boot_stats, high_pct))
    return ConfidenceInterval(round(low, 4), round(high, 4))


def cluster_bootstrap_paired_difference_ci(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    cluster_key: str = "user_id",
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, ConfidenceInterval]:
    """Non-parametric two-stage cluster bootstrap interval for paired difference in accuracy (B - A)."""
    map_a = {r["id"]: (1.0 if r["predicted_action"] == r["ground_truth"] else 0.0) for r in records_a}
    map_b = {r["id"]: (1.0 if r["predicted_action"] == r["ground_truth"] else 0.0) for r in records_b}

    paired_by_user: dict[str, list[tuple[float, float]]] = {}
    for r in records_a:
        tid = r["id"]
        uid = r.get(cluster_key, "default")
        val_a = map_a[tid]
        val_b = map_b.get(tid, 0.0)
        paired_by_user.setdefault(uid, []).append((val_a, val_b))

    all_pairs = [pair for pairs in paired_by_user.values() for pair in pairs]
    observed_diff = (sum(p[1] for p in all_pairs) - sum(p[0] for p in all_pairs)) / len(all_pairs) if all_pairs else 0.0

    cluster_keys = list(paired_by_user.keys())
    k = len(cluster_keys)
    if k <= 1:
        return round(observed_diff, 4), ConfidenceInterval(round(observed_diff, 4), round(observed_diff, 4))

    rng = random.Random(seed)
    boot_diffs: list[float] = []

    for _ in range(n_resamples):
        sampled_pairs: list[tuple[float, float]] = []
        for _ in range(k):
            chosen = rng.choice(cluster_keys)
            user_pairs = paired_by_user[chosen]
            n_p = len(user_pairs)
            sampled_pairs.extend([rng.choice(user_pairs) for _ in range(n_p)])

        n_tot = len(sampled_pairs)
        diff = (sum(p[1] for p in sampled_pairs) - sum(p[0] for p in sampled_pairs)) / n_tot if n_tot > 0 else 0.0
        boot_diffs.append(diff)

    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0
    low = float(np.percentile(boot_diffs, low_pct))
    high = float(np.percentile(boot_diffs, high_pct))
    return round(observed_diff, 4), ConfidenceInterval(round(low, 4), round(high, 4))


@dataclass
class EvalSummary:
    """Summary of turn-taking performance metrics across an evaluation split."""

    name: str
    total_turns: int
    finished_turns: int
    incomplete_turns: int
    premature_count: int
    unnecessary_wait_count: int
    accuracy: float
    accuracy_ci: ConfidenceInterval
    accuracy_cluster_ci: ConfidenceInterval
    premature_interruption_rate: float
    premature_interruption_ci: ConfidenceInterval
    premature_interruption_cluster_ci: ConfidenceInterval
    unnecessary_wait_rate: float
    unnecessary_wait_ci: ConfidenceInterval
    unnecessary_wait_cluster_ci: ConfidenceInterval
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
            "finished_turns": self.finished_turns,
            "incomplete_turns": self.incomplete_turns,
            "premature_count": self.premature_count,
            "unnecessary_wait_count": self.unnecessary_wait_count,
            "accuracy": round(self.accuracy, 4),
            "accuracy_ci": [self.accuracy_ci.low, self.accuracy_ci.high],
            "accuracy_cluster_ci": [self.accuracy_cluster_ci.low, self.accuracy_cluster_ci.high],
            "premature_interruption_rate": round(self.premature_interruption_rate, 4),
            "premature_interruption_ci": [self.premature_interruption_ci.low, self.premature_interruption_ci.high],
            "premature_interruption_cluster_ci": [self.premature_interruption_cluster_ci.low, self.premature_interruption_cluster_ci.high],
            "unnecessary_wait_rate": round(self.unnecessary_wait_rate, 4),
            "unnecessary_wait_ci": [self.unnecessary_wait_ci.low, self.unnecessary_wait_ci.high],
            "unnecessary_wait_cluster_ci": [self.unnecessary_wait_cluster_ci.low, self.unnecessary_wait_cluster_ci.high],
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
    acc_cluster_ci = cluster_bootstrap_ci(eval_records, "accuracy")

    prem_rate = premature_count / total
    prem_ci = wilson_score_interval(premature_count, total)
    prem_cluster_ci = cluster_bootstrap_ci(eval_records, "premature")

    unnec_rate = unnecessary_wait_count / finished_turns if finished_turns > 0 else 0.0
    unnec_ci = wilson_score_interval(unnecessary_wait_count, finished_turns) if finished_turns > 0 else ConfidenceInterval(0.0, 0.0)
    unnec_cluster_ci = cluster_bootstrap_ci(eval_records, "unnecessary_wait") if finished_turns > 0 else ConfidenceInterval(0.0, 0.0)

    delay_p50 = float(np.median(response_delays)) if response_delays else 0.0
    delay_p50_ci = bootstrap_percentile_ci(response_delays, percentile=50.0) if response_delays else ConfidenceInterval(0.0, 0.0)

    delay_p95 = float(np.percentile(response_delays, 95.0)) if response_delays else 0.0
    delay_p95_ci = bootstrap_percentile_ci(response_delays, percentile=95.0) if response_delays else ConfidenceInterval(0.0, 0.0)

    dec_p50 = float(np.median(decision_latencies)) if decision_latencies else 0.0
    dec_p95 = float(np.percentile(decision_latencies, 95.0)) if decision_latencies else 0.0

    total_cost = total * inference_cost_per_turn

    incomplete_turns = total - finished_turns

    return EvalSummary(
        name=approach_name,
        total_turns=total,
        finished_turns=finished_turns,
        incomplete_turns=incomplete_turns,
        premature_count=premature_count,
        unnecessary_wait_count=unnecessary_wait_count,
        accuracy=accuracy,
        accuracy_ci=acc_ci,
        accuracy_cluster_ci=acc_cluster_ci,
        premature_interruption_rate=prem_rate,
        premature_interruption_ci=prem_ci,
        premature_interruption_cluster_ci=prem_cluster_ci,
        unnecessary_wait_rate=unnec_rate,
        unnecessary_wait_ci=unnec_ci,
        unnecessary_wait_cluster_ci=unnec_cluster_ci,
        response_delay_p50_ms=delay_p50,
        response_delay_p50_ci=delay_p50_ci,
        response_delay_p95_ms=delay_p95,
        response_delay_p95_ci=delay_p95_ci,
        decision_latency_p50_ms=dec_p50,
        decision_latency_p95_ms=dec_p95,
        inference_cost_usd=total_cost,
    )

