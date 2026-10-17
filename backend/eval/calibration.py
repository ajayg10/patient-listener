"""Policy calibration, probability reliability data, and Pareto sweep visualization.

Pure evaluation module:
- Calibrates policy thresholds (hi, lo) strictly on DEV split only.
- Computes probability calibration bins and Brier scores on PROBABILITIES (not confidence).
- Evaluates Pareto frontiers (premature interruption rate vs response delay P50/P95).
- Generates publication-ready PNG plots via matplotlib with provenance watermarking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend.app.policy import PolicyThresholds, policy
from scripts.precompute_decisions import compute_cache_key


def calibrate_policy_on_dev(
    dev_turns: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    model_id: str,
    checkpoints: list[int],
    max_wait_ms: int,
) -> tuple[PolicyThresholds, dict[str, Any]]:
    """Grid search (hi, lo) thresholds on DEV split turns ONLY (asserting test is untouched).

    Returns optimal PolicyThresholds and search diagnostic statistics.
    """
    assert len(dev_turns) > 0, "dev_turns must not be empty"
    for t in dev_turns:
        user_id = t.get("user_id", "")
        turn_id = t.get("id", "")
        tmpl_id = t.get("scenario_template_id", "")
        if "test" in user_id or "test" in turn_id or "test" in tmpl_id:
            raise ValueError(f"Contamination detected in dev calibration! Found test element: {turn_id}")

    hi_candidates = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    lo_candidates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

    grid_pairs = [(round(hi, 2), round(lo, 2)) for hi in hi_candidates for lo in lo_candidates if hi > lo]

    best_score = -1.0
    best_hi = 0.70
    best_lo = 0.30
    best_metrics: dict[str, float] = {}

    for hi, lo in grid_pairs:
        thresh = PolicyThresholds(hi=hi, lo=lo, max_hold_ms=float(max_wait_ms))
        correct = 0
        total = len(dev_turns)
        premature = 0

        for t in dev_turns:
            silence_ms = t["silence_ms"]
            is_complete = t["is_turn_complete"]
            gt = t["ground_truth"]

            pred_action = "WAIT"

            for cp in checkpoints:
                if not is_complete and cp > silence_ms:
                    break
                if cp >= max_wait_ms:
                    break

                speech_text = " ".join(t["text"].split()[-40:])
                st = f'Speaker: "{speech_text}"\n[Silence: {cp} ms]'
                q1_key = compute_cache_key(model_id, "turn_complete", st)
                q2_key = compute_cache_key(model_id, "mid_thought", st)

                cached_q1 = cache.get(q1_key)
                if not cached_q1:
                    continue

                q1_probs = cached_q1["probabilities"]
                q2_probs = cache.get(q2_key, {}).get("probabilities") if q2_key in cache else None

                act = policy(q1_probs, cp, thresholds=thresh, q2_probabilities=q2_probs)
                if act == "RESPOND":
                    pred_action = "RESPOND"
                    break

            if is_complete and pred_action == "WAIT":
                pred_action = "RESPOND"

            if pred_action == gt:
                correct += 1
            if pred_action == "RESPOND" and not is_complete:
                premature += 1

        accuracy = correct / total if total > 0 else 0.0
        # Multi-objective score: accuracy penalized by premature interruptions
        prem_rate = premature / total if total > 0 else 0.0
        score = accuracy - (0.5 * prem_rate)

        if score > best_score:
            best_score = score
            best_hi = hi
            best_lo = lo
            best_metrics = {
                "accuracy": round(accuracy, 4),
                "premature_rate": round(prem_rate, 4),
                "score": round(score, 4),
            }

    stats = {
        "grid_size": len(grid_pairs),
        "dev_chosen_hi": best_hi,
        "dev_chosen_lo": best_lo,
        "dev_metrics": best_metrics,
        "dev_sample_count": len(dev_turns),
    }
    return PolicyThresholds(hi=best_hi, lo=best_lo, max_hold_ms=float(max_wait_ms)), stats


def compute_probability_reliability(
    turns: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    model_id: str,
    checkpoints: list[int],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute reliability data on model PROBABILITIES (not confidence).

    Bins probability p(RESPOND), calculates mean predicted vs empirical observed frequency,
    and returns Brier score. Asserts bin counts sum exactly to total decisions N.
    """
    probs: list[float] = []
    labels: list[int] = []

    for t in turns:
        is_complete = t["is_turn_complete"]
        silence_ms = t["silence_ms"]
        speech_text = " ".join(t["text"].split()[-40:])

        for cp in checkpoints:
            if not is_complete and cp > silence_ms:
                break
            st = f'Speaker: "{speech_text}"\n[Silence: {cp} ms]'
            key = compute_cache_key(model_id, "turn_complete", st)
            if key in cache:
                p_resp = cache[key]["probabilities"].get("RESPOND", 0.0)
                probs.append(float(p_resp))
                labels.append(1 if is_complete else 0)

    n_samples = len(probs)
    if n_samples == 0:
        return {"brier_score": 0.0, "sample_count": 0, "bins": []}

    # Brier score: 1/N * sum((p - y)^2)
    brier = float(np.mean([(p - y) ** 2 for p, y in zip(probs, labels)]))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data: list[dict[str, Any]] = []

    total_counted = 0
    for i in range(n_bins):
        low_edge = bin_edges[i]
        high_edge = bin_edges[i + 1]

        if i == n_bins - 1:
            indices = [idx for idx, p in enumerate(probs) if low_edge <= p <= high_edge]
        else:
            indices = [idx for idx, p in enumerate(probs) if low_edge <= p < high_edge]

        bin_count = len(indices)
        total_counted += bin_count

        if bin_count > 0:
            mean_pred = float(np.mean([probs[idx] for idx in indices]))
            obs_rate = float(np.mean([labels[idx] for idx in indices]))
        else:
            mean_pred = float((low_edge + high_edge) / 2.0)
            obs_rate = 0.0

        bins_data.append({
            "bin_index": i,
            "range": [round(float(low_edge), 2), round(float(high_edge), 2)],
            "count": bin_count,
            "mean_predicted_prob": round(mean_pred, 4),
            "observed_rate": round(obs_rate, 4),
        })

    # Strict invariant: bin counts must sum to N
    assert total_counted == n_samples, f"Bin count sum ({total_counted}) != total samples ({n_samples})"

    return {
        "brier_score": round(brier, 4),
        "sample_count": n_samples,
        "bins": bins_data,
    }


def render_reliability_plot(
    dev_reliability: dict[str, Any],
    test_reliability: dict[str, Any],
    output_png: Path,
    is_mock: bool = True,
) -> None:
    """Render probability calibration / reliability curve to PNG."""
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    # Reference diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.6)

    # Plot DEV
    dev_bins = dev_reliability.get("bins", [])
    if dev_bins:
        x_dev = [b["mean_predicted_prob"] for b in dev_bins if b["count"] > 0]
        y_dev = [b["observed_rate"] for b in dev_bins if b["count"] > 0]
        ax.plot(x_dev, y_dev, "s-", color="#1f77b4", label=f"DEV Split (Brier: {dev_reliability['brier_score']:.4f})")

    # Plot TEST
    test_bins = test_reliability.get("bins", [])
    if test_bins:
        x_test = [b["mean_predicted_prob"] for b in test_bins if b["count"] > 0]
        y_test = [b["observed_rate"] for b in test_bins if b["count"] > 0]
        ax.plot(x_test, y_test, "o-", color="#ff7f0e", label=f"TEST Split (Brier: {test_reliability['brier_score']:.4f})")

    ax.set_xlabel("Mean Predicted Probability P(RESPOND)", fontsize=11)
    ax.set_ylabel("Empirical Turn Completion Rate", fontsize=11)
    ax.set_title("Probability Calibration Curve (Reliability Diagram)", fontsize=13, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    if is_mock:
        fig.text(
            0.5, 0.5, "MOCK-DECIDER", fontsize=36, color="gray",
            alpha=0.18, ha="center", va="center", rotation=30, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close(fig)


def render_pareto_sweeps(
    turns: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    model_id: str,
    checkpoints: list[int],
    max_wait_ms: int,
    output_png: Path,
    output_json: Path,
    is_mock: bool = True,
    gpu_latency_ms: float = 235.6,
    cpu_latency_ms: float = 3190.1,
) -> dict[str, Any]:
    """Sweep premature interruption rate vs response delay across hardware latencies."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    from backend.eval.baselines import FixedTimeoutBaseline, TextHeuristicBaseline

    # 1. Fixed Timeout Sweep: 300ms to 2000ms
    fixed_timeouts = [300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 1800, 2000]
    fixed_curve: list[dict[str, Any]] = []

    for f_to in fixed_timeouts:
        base = FixedTimeoutBaseline(timeout_ms=f_to)
        prem = 0
        delays = []
        for t in turns:
            sil = t["silence_ms"]
            is_c = t["is_turn_complete"]
            act = base.decide(sil)
            if act == "RESPOND" and not is_c:
                prem += 1
            if is_c:
                delay = f_to if sil >= f_to else sil
                delays.append(float(delay))
        prem_rate = prem / len(turns) if turns else 0.0
        p50 = float(np.median(delays)) if delays else 0.0
        p95 = float(np.percentile(delays, 95)) if delays else 0.0
        fixed_curve.append({
            "param": f_to,
            "premature_rate": round(prem_rate, 4),
            "delay_p50_ms": round(p50, 1),
            "delay_p95_ms": round(p95, 1),
        })

    # 2. Text Heuristic Threshold Sweep: 100ms to 1200ms
    heuristic_thresholds = [100, 200, 250, 350, 500, 700, 900, 1200]
    heuristic_curve: list[dict[str, Any]] = []

    for h_th in heuristic_thresholds:
        base = TextHeuristicBaseline(threshold_ms=h_th)
        prem = 0
        delays = []
        for t in turns:
            sil = t["silence_ms"]
            is_c = t["is_turn_complete"]
            act = base.decide(t["text"], sil)
            if act == "RESPOND" and not is_c:
                prem += 1
            if is_c:
                delay = h_th if act == "RESPOND" else sil
                delays.append(float(delay))
        prem_rate = prem / len(turns) if turns else 0.0
        p50 = float(np.median(delays)) if delays else 0.0
        p95 = float(np.percentile(delays, 95)) if delays else 0.0
        heuristic_curve.append({
            "param": h_th,
            "premature_rate": round(prem_rate, 4),
            "delay_p50_ms": round(p50, 1),
            "delay_p95_ms": round(p95, 1),
        })

    # 3. Decider Policy hi Sweep at 3 Latencies (0ms, GPU 235.6ms, CPU 3190.1ms)
    hi_values = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    latencies = {
        "0ms_oracle": 0.0,
        f"gpu_{gpu_latency_ms:.0f}ms_T4": gpu_latency_ms,
        f"cpu_{cpu_latency_ms:.0f}ms": cpu_latency_ms,
    }

    decider_curves: dict[str, list[dict[str, Any]]] = {}

    for lat_name, lat_val in latencies.items():
        curve: list[dict[str, Any]] = []
        for hi in hi_values:
            thresh = PolicyThresholds(hi=hi, lo=0.30, max_hold_ms=float(max_wait_ms))
            prem = 0
            delays = []
            for t in turns:
                sil = t["silence_ms"]
                is_c = t["is_turn_complete"]

                pred_action = "WAIT"
                action_delay = float(max_wait_ms)

                for cp in checkpoints:
                    if not is_c and cp > sil:
                        break
                    if cp >= max_wait_ms:
                        break

                    st = f'Speaker: "{" ".join(t["text"].split()[-40:])}"\n[Silence: {cp} ms]'
                    q1_key = compute_cache_key(model_id, "turn_complete", st)
                    q2_key = compute_cache_key(model_id, "mid_thought", st)

                    if q1_key not in cache:
                        continue

                    arrival = cp + lat_val
                    if not is_c and arrival >= sil:
                        pred_action = "WAIT"
                        action_delay = sil
                        break

                    if arrival >= max_wait_ms:
                        pred_action = "RESPOND"
                        action_delay = float(max_wait_ms)
                        break

                    q1_p = cache[q1_key]["probabilities"]
                    q2_p = cache.get(q2_key, {}).get("probabilities") if q2_key in cache else None

                    act = policy(q1_p, cp, thresholds=thresh, q2_probabilities=q2_p)
                    if act == "RESPOND":
                        pred_action = "RESPOND"
                        action_delay = arrival
                        break

                if is_c and pred_action == "WAIT":
                    pred_action = "RESPOND"
                    action_delay = float(max_wait_ms)

                if pred_action == "RESPOND" and not is_c:
                    prem += 1
                if is_c:
                    delays.append(action_delay)

            prem_rate = prem / len(turns) if turns else 0.0
            p50 = float(np.median(delays)) if delays else 0.0
            p95 = float(np.percentile(delays, 95)) if delays else 0.0
            curve.append({
                "param_hi": hi,
                "premature_rate": round(prem_rate, 4),
                "delay_p50_ms": round(p50, 1),
                "delay_p95_ms": round(p95, 1),
            })
        decider_curves[lat_name] = curve

    pareto_data = {
        "provenance": "MOCK-DECIDER" if is_mock else "REAL-DECIDER (Tesla T4)",
        "fixed_timeout_sweep": fixed_curve,
        "text_heuristic_sweep": heuristic_curve,
        "decider_hi_sweeps": decider_curves,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pareto_data, f, indent=2)

    # Render Pareto Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    prov_tag = "[MOCK-DECIDER]" if is_mock else "[REAL-DECIDER]"

    # Subplot 1: Premature Interruption Rate vs Delay P50
    ax1.plot(
        [pt["delay_p50_ms"] for pt in fixed_curve],
        [pt["premature_rate"] * 100 for pt in fixed_curve],
        "k:o", label="Fixed Timeout (300-2000ms) [Baseline]", alpha=0.7
    )
    ax1.plot(
        [pt["delay_p50_ms"] for pt in heuristic_curve],
        [pt["premature_rate"] * 100 for pt in heuristic_curve],
        "m--s", label="Text Heuristic (100-1200ms) [Baseline]", alpha=0.7
    )
    ax1.plot(
        [pt["delay_p50_ms"] for pt in decider_curves["0ms_oracle"]],
        [pt["premature_rate"] * 100 for pt in decider_curves["0ms_oracle"]],
        "g-^", label=f"Decider (0ms latency, hi sweep) {prov_tag}"
    )
    ax1.plot(
        [pt["delay_p50_ms"] for pt in decider_curves[f"gpu_{gpu_latency_ms:.0f}ms_T4"]],
        [pt["premature_rate"] * 100 for pt in decider_curves[f"gpu_{gpu_latency_ms:.0f}ms_T4"]],
        "b-o", label=f"Decider ({gpu_latency_ms:.0f}ms GPU T4, hi sweep) {prov_tag}"
    )
    ax1.plot(
        [pt["delay_p50_ms"] for pt in decider_curves[f"cpu_{cpu_latency_ms:.0f}ms"]],
        [pt["premature_rate"] * 100 for pt in decider_curves[f"cpu_{cpu_latency_ms:.0f}ms"]],
        "r-x", label=f"Decider ({cpu_latency_ms:.0f}ms CPU) {prov_tag}"
    )

    ax1.set_xlabel("Response Delay P50 (ms)", fontsize=11)
    ax1.set_ylabel("Premature Interruption Rate (%)", fontsize=11)
    ax1.set_title("Pareto Trade-off: Premature vs Delay P50", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.0)

    # Subplot 2: Premature Interruption Rate vs Delay P95
    ax2.plot(
        [pt["delay_p95_ms"] for pt in fixed_curve],
        [pt["premature_rate"] * 100 for pt in fixed_curve],
        "k:o", label="Fixed Timeout (300-2000ms) [Baseline]", alpha=0.7
    )
    ax2.plot(
        [pt["delay_p95_ms"] for pt in heuristic_curve],
        [pt["premature_rate"] * 100 for pt in heuristic_curve],
        "m--s", label="Text Heuristic (100-1200ms) [Baseline]", alpha=0.7
    )
    ax2.plot(
        [pt["delay_p95_ms"] for pt in decider_curves["0ms_oracle"]],
        [pt["premature_rate"] * 100 for pt in decider_curves["0ms_oracle"]],
        "g-^", label=f"Decider (0ms latency, hi sweep) {prov_tag}"
    )
    ax2.plot(
        [pt["delay_p95_ms"] for pt in decider_curves[f"gpu_{gpu_latency_ms:.0f}ms_T4"]],
        [pt["premature_rate"] * 100 for pt in decider_curves[f"gpu_{gpu_latency_ms:.0f}ms_T4"]],
        "b-o", label=f"Decider ({gpu_latency_ms:.0f}ms GPU T4, hi sweep) {prov_tag}"
    )
    ax2.plot(
        [pt["delay_p95_ms"] for pt in decider_curves[f"cpu_{cpu_latency_ms:.0f}ms"]],
        [pt["premature_rate"] * 100 for pt in decider_curves[f"cpu_{cpu_latency_ms:.0f}ms"]],
        "r-x", label=f"Decider ({cpu_latency_ms:.0f}ms CPU) {prov_tag}"
    )

    ax2.set_xlabel("Response Delay P95 (ms)", fontsize=11)
    ax2.set_ylabel("Premature Interruption Rate (%)", fontsize=11)
    ax2.set_title("Pareto Trade-off: Premature vs Delay P95", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.0)

    prov_text = f"Provenance: {pareto_data['provenance']}"
    fig.suptitle(f"Turn-Taking Pareto Frontier ({prov_text})", fontsize=14, fontweight="bold")

    if is_mock:
        fig.text(
            0.5, 0.5, "MOCK-DECIDER", fontsize=48, color="gray",
            alpha=0.15, ha="center", va="center", rotation=25, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close(fig)

    return pareto_data
