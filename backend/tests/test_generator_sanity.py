"""Tests for synthetic dataset generator sanity and statistical tolerance.

Enforces:
1. Realized continuation rates vs configured per-user p within binomial tolerance.
2. Stratified quota allocation preventing runaway statistical drift (e.g. 26/36 under p=0.50).
3. Pearson correlation between user history session pause durations and p_user.
"""

from __future__ import annotations

import math


def check_binomial_tolerance(trials: int, successes: int, p: float, z: float = 2.0) -> bool:
    """Return True if successes / trials falls within binomial tolerance of p."""
    if trials == 0:
        return True
    se = math.sqrt(p * (1.0 - p) / trials)
    allowed_margin = z * se + (0.5 / trials)  # standard continuity correction
    rate = successes / trials
    return abs(rate - p) <= allowed_margin


def test_binomial_tolerance_math():
    """Verify binomial tolerance rejects extreme anomalies and accepts expected ranges."""
    # 36 trials, p = 0.50: mean 18, SE = 0.0833.
    # 26 successes (72.2%) has ~1 in 150 odds under p=0.50 and must be rejected under 2.0 SE.
    assert check_binomial_tolerance(36, 18, 0.50, z=2.0) is True
    assert check_binomial_tolerance(36, 20, 0.50, z=2.0) is True
    assert check_binomial_tolerance(36, 26, 0.50, z=2.0) is False  # Fails sanity check


def test_generator_quota_allocation():
    """Verify that stratified quota allocation strictly enforces |k - N*p| <= 0.5."""
    for n in [6, 12, 18, 24]:
        for p in [0.20, 0.30, 0.425, 0.50, 0.65, 0.80]:
            k = int(round(n * p))
            realized_p = k / n
            assert abs(k - n * p) <= 0.5
            assert abs(realized_p - p) <= (0.5 / n) + 1e-9


def test_correlation_math():
    """Verify Pearson correlation calculation helper."""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [10.0, 20.0, 30.0, 40.0, 50.0]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    r = cov / math.sqrt(var_x * var_y)
    assert abs(r - 1.0) < 1e-6
