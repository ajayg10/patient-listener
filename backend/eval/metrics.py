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
    """Compute non-parametric bootstrap confidence interval for a percentile (vectorized with numpy)."""
    if not values:
        return ConfidenceInterval(0.0, 0.0)
    if len(values) == 1:
        return ConfidenceInterval(values[0], values[0])

    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    samples = arr[indices]
    boot_stats = np.percentile(samples, percentile, axis=1)

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
    """Non-parametric two-stage cluster bootstrap interval (vectorized with numpy).

    Procedure:
    1. Group records into K distinct user clusters C_1, ..., C_K.
    2. Aggregate per-user numerators and denominators.
    3. For B resamples:
       a. Sample K user clusters with replacement.
       b. Resample turns within each sampled user using binomial draws.
       c. Compute metric theta*(b) on the pooled sample.
    4. Return [percentile(theta*, 2.5), percentile(theta*, 97.5)].
    """
    if not records:
        return ConfidenceInterval(0.0, 0.0)

    clusters: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        uid = r.get(cluster_key, "default")
        clusters.setdefault(uid, []).append(r)

    cluster_keys = sorted(clusters.keys())
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

    user_n_list: list[int] = []
    user_s_list: list[int] = []

    for uid in cluster_keys:
        user_recs = clusters[uid]
        if metric_type == "accuracy":
            n_u = len(user_recs)
            s_u = sum(1 for r in user_recs if r["predicted_action"] == r["ground_truth"])
        elif metric_type == "premature":
            n_u = len(user_recs)
            s_u = sum(1 for r in user_recs if r["predicted_action"] == "RESPOND" and not r["is_turn_complete"])
        elif metric_type == "unnecessary_wait":
            completed = [r for r in user_recs if r["is_turn_complete"]]
            n_u = len(completed)
            s_u = sum(1 for r in completed if (r["action_timestamp_ms"] - r["turn_end_t"] > 800.0 or r["predicted_action"] == "WAIT"))
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        user_n_list.append(n_u)
        user_s_list.append(s_u)

    user_n = np.array(user_n_list, dtype=np.int64)
    user_s = np.array(user_s_list, dtype=np.int64)
    user_p = np.where(user_n > 0, user_s / user_n, 0.0)

    rng = np.random.default_rng(seed)
    u_idx = rng.integers(0, k, size=(n_resamples, k))

    sampled_n = user_n[u_idx]
    sampled_p = user_p[u_idx]

    # Draw sampled success counts using vectorized binomial: Binomial(N_u, P_u)
    sampled_s = rng.binomial(sampled_n, sampled_p)

    tot_num = np.sum(sampled_s, axis=1)
    tot_den = np.sum(sampled_n, axis=1)
    boot_stats = np.where(tot_den > 0, tot_num / tot_den, 0.0)

    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0
    low = float(np.percentile(boot_stats, low_pct))
    high = float(np.percentile(boot_stats, high_pct))
    return ConfidenceInterval(round(low, 4), round(high, 4))


def cluster_bootstrap_paired_difference_ci(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    cluster_key: str = "user_id",
    metric: str = "accuracy",
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, ConfidenceInterval]:
    """Non-parametric two-stage cluster bootstrap interval for paired difference (B - A). Vectorized."""
    def extract_val(r: dict[str, Any]) -> float:
        if metric == "premature":
            return 1.0 if (r["predicted_action"] == "RESPOND" and not r["is_turn_complete"]) else 0.0
        elif metric == "delay":
            return max(0.0, float(r["action_timestamp_ms"] - r["turn_end_t"]))
        return 1.0 if r["predicted_action"] == r["ground_truth"] else 0.0

    map_a = {r["id"]: r for r in records_a}
    map_b = {r["id"]: r for r in records_b}

    paired_by_user: dict[str, list[float]] = {}
    for tid, r_a in map_a.items():
        if tid not in map_b:
            continue
        r_b = map_b[tid]
        if metric == "delay" and not r_a.get("is_turn_complete", False):
            continue
        uid = r_a.get(cluster_key, "default")
        diff_val = extract_val(r_b) - extract_val(r_a)
        paired_by_user.setdefault(uid, []).append(diff_val)

    all_diffs = [d for diffs in paired_by_user.values() for d in diffs]
    observed_diff = float(np.mean(all_diffs)) if all_diffs else 0.0

    cluster_keys = sorted(paired_by_user.keys())
    k = len(cluster_keys)
    if k <= 1:
        return round(observed_diff, 4), ConfidenceInterval(round(observed_diff, 4), round(observed_diff, 4))

    user_diff_arrays = [np.array(paired_by_user[uid], dtype=np.float64) for uid in cluster_keys]
    user_lens = np.array([len(arr) for arr in user_diff_arrays], dtype=np.int64)

    rng = np.random.default_rng(seed)
    equal_lens = bool(np.all(user_lens == user_lens[0]))

    if equal_lens and len(user_lens) > 0 and user_lens[0] > 0:
        m = user_lens[0]
        d_matrix = np.vstack(user_diff_arrays)
        u_idx = rng.integers(0, k, size=(n_resamples, k))
        t_idx = rng.integers(0, m, size=(n_resamples, k, m))
        sampled_d = d_matrix[u_idx[:, :, None], t_idx]
        boot_diffs = np.mean(sampled_d, axis=(1, 2))
    else:
        max_m = int(np.max(user_lens)) if len(user_lens) > 0 else 0
        if max_m == 0:
            return round(observed_diff, 4), ConfidenceInterval(round(observed_diff, 4), round(observed_diff, 4))
        padded_d = np.zeros((k, max_m), dtype=np.float64)
        for i, arr in enumerate(user_diff_arrays):
            padded_d[i, :len(arr)] = arr

        u_idx = rng.integers(0, k, size=(n_resamples, k))
        sampled_lens = user_lens[u_idx]
        t_idx = (rng.random(size=(n_resamples, k, max_m)) * sampled_lens[:, :, None]).astype(np.int64)
        sampled_d = padded_d[u_idx[:, :, None], t_idx]
        mask = np.arange(max_m)[None, None, :] < sampled_lens[:, :, None]
        tot_diff = np.sum(sampled_d * mask, axis=(1, 2))
        tot_count = np.sum(sampled_lens, axis=1)
        boot_diffs = np.where(tot_count > 0, tot_diff / tot_count, 0.0)

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
    n_resamples: int = 1000,
    compute_ci: bool = True,
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
    prem_rate = premature_count / total
    unnec_rate = unnecessary_wait_count / finished_turns if finished_turns > 0 else 0.0
    delay_p50 = float(np.median(response_delays)) if response_delays else 0.0
    delay_p95 = float(np.percentile(response_delays, 95.0)) if response_delays else 0.0
    dec_p50 = float(np.median(decision_latencies)) if decision_latencies else 0.0
    dec_p95 = float(np.percentile(decision_latencies, 95.0)) if decision_latencies else 0.0

    if compute_ci:
        acc_ci = wilson_score_interval(correct_count, total)
        acc_cluster_ci = cluster_bootstrap_ci(eval_records, "accuracy", n_resamples=n_resamples)

        prem_ci = wilson_score_interval(premature_count, total)
        prem_cluster_ci = cluster_bootstrap_ci(eval_records, "premature", n_resamples=n_resamples)

        unnec_ci = wilson_score_interval(unnecessary_wait_count, finished_turns) if finished_turns > 0 else ConfidenceInterval(0.0, 0.0)
        unnec_cluster_ci = cluster_bootstrap_ci(eval_records, "unnecessary_wait", n_resamples=n_resamples) if finished_turns > 0 else ConfidenceInterval(0.0, 0.0)

        delay_p50_ci = bootstrap_percentile_ci(response_delays, percentile=50.0, n_resamples=n_resamples) if response_delays else ConfidenceInterval(0.0, 0.0)
        delay_p95_ci = bootstrap_percentile_ci(response_delays, percentile=95.0, n_resamples=n_resamples) if response_delays else ConfidenceInterval(0.0, 0.0)
    else:
        empty_ci = ConfidenceInterval(0.0, 0.0)
        acc_ci = acc_cluster_ci = prem_ci = prem_cluster_ci = unnec_ci = unnec_cluster_ci = delay_p50_ci = delay_p95_ci = empty_ci

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

