# Benchmark Results — Patient Listener

> [!NOTE]
> **PROVENANCE**: Decider evaluated from real GPU cache (cuda (Tesla T4), 8450 keys).

> [!IMPORTANT]
> **SYNTHETIC BENCHMARK**: Evaluated on 360 synthetic test turns (181 complete, 179 incomplete) across 30 users (12 turns/user) in 3 conflicting persona archetypes (fast, slow, filler-heavy). Zero leakage: dev and test splits strictly partition users and templates.

### Dataset Reconciled Count Table (Current Split)

| Split | Turns | Users | Unique Utterances | Lookups | Unique States Reached | Cache Hits | Misses |
|---|---|---|---|---|---|---|---|
| DEV | 180 | 15 | 24 | 644 | 102 | 644 | 0 |
| TEST | 360 | 30 | 36 | 1282 | 154 | 1282 | 0 |
| TOTAL | 540 | 45 | 60 | 1926 | 256 | 1926 | 0 |

- **Lookups difference (dev 664 -> 685, test 1,136 -> 1,146)**: Increasing ambiguous share to 50% in B5.8 extended silences so more turns reached later pause checkpoints before speech resumed.
- **States difference (test 156 -> 155)**: In the B5.8 test realization, one slow template never experienced a silence lasting to the 1800 ms checkpoint.
- **2,250 vs 1,831 lookups**: 1,831 (685 dev + 1,146 test) is the single-pass lookup count, whereas 2,250 was the cumulative query count across all latency sweeps in `run_eval.py`.

## 1. Operating Point Benchmark: Premature Interruption and Delay (n=360 test turns across 30 users)

> [!NOTE]
> **Operating Points Chosen on DEV Only, Reported on TEST Only**:
> - **Point A (Delay Budget)**: DEV delay P50 <= 800 ms (matching Fixed 800ms baseline), minimizing DEV premature rate.
> - **Point B (Premature Budget)**: DEV premature rate <= 5.0%, minimizing DEV delay P50.
> - *Accuracy is removed from this headline comparison as delay and premature interruption rate jointly define turn-taking quality.*

| Operating Point | Approach | Parameter | Premature Interruption (Cluster 95% CI) | Delay P50 (Bootstrap 95% CI) | Delay P95 | Condition |
|---|---|---|---|---|---|---|
| Point A (Delay budget <= 800ms) | Fixed Timeout | 800 ms | 18.9% [11.9%, 26.4%] | 800 ms [800.0, 800.0] ms | 800 ms | Fixed timeout (800ms) |
| Point A (Delay budget <= 800ms) | Text Heuristic | 800 ms | 11.1% [6.4%, 16.7%] | 800 ms [800.0, 800.0] ms | 800 ms | dev-tuned (800ms) |
| Point A (Delay budget <= 800ms) | Per-User Control | scale 1.15 | 4.2% [1.4%, 7.8%] | 808 ms [760.1, 874.0] ms | 1067 ms | profile threshold x1.15 (no model) |
| Point A (Delay budget <= 800ms) | Strands Decider 2B | hi = 0.30 | 10.0% [5.8%, 14.7%] | 1501 ms [551.2, 2200.0] ms | 2200 ms | GPU cache (hi=0.3, lo=0.30) |
| Point A (Delay budget <= 800ms) | Strands Decider 2B + Profile | hi = 0.30 | 8.1% [4.2%, 12.2%] | 551 ms [551.2, 1501.2] ms | 2200 ms | GPU cache + profile (hi=0.3, lo=0.30) |
| Point B (Premature budget <= 5%) | Fixed Timeout | 950 ms | 3.3% [1.1%, 6.4%] | 950 ms [910.0, 950.0] ms | 950 ms | Fixed timeout (950ms) |
| Point B (Premature budget <= 5%) | Text Heuristic | 950 ms | 2.5% [0.6%, 5.3%] | 950 ms [910.0, 950.0] ms | 950 ms | dev-tuned (950ms) |
| Point B (Premature budget <= 5%) | Per-User Control | scale 1.10 | 7.5% [3.6%, 11.9%] | 778 ms [727.1, 836.0] ms | 1021 ms | profile threshold x1.1 (no model) |
| Point B (Premature budget <= 5%) | Strands Decider 2B | hi = 0.55 | 0.0% [0.0%, 0.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | GPU cache (hi=0.55, lo=0.30) |
| Point B (Premature budget <= 5%) | Strands Decider 2B + Profile | hi = 0.50 | 0.0% [0.0%, 0.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | GPU cache + profile (hi=0.5, lo=0.30) |

### Key Paired Differences at Operating Points (Cluster Bootstrap 95% CI over Users)

#### Point A: Paired Premature Rate Differences (Equal Delay Budget <= 800ms)
| Comparison | Model A | Model B | Paired Premature Difference (B - A) | Cluster Bootstrap 95% CI |
|---|---|---|---|---|
| Decider vs Fixed Timeout | Fixed Timeout (800ms) | Strands Decider 2B (hi=0.30) | -8.9% | [-15.6%, -3.0%] |
| Decider vs Text Heuristic | Text Heuristic (800ms) | Strands Decider 2B (hi=0.30) | -1.1% | [-5.6%, +3.3%] |
| Decider+Profile vs Decider | Strands Decider 2B (hi=0.30) | Strands Decider 2B + Profile (hi=0.30) | -1.9% | [-5.8%, +1.1%] |
| Decider+Profile vs Per-User Control | Per-User Control (scale 1.15) | Strands Decider 2B + Profile (hi=0.30) | +3.9% | [-1.1%, +8.6%] |

#### Point B: Paired Response Delay Differences (Equal Premature Budget <= 5%)
| Comparison | Model A | Model B | Paired Delay Difference (B - A) | Cluster Bootstrap 95% CI |
|---|---|---|---|---|
| Decider vs Fixed Timeout | Fixed Timeout (950ms) | Strands Decider 2B (hi=0.55) | +1353.2 ms | [+1304.6, +1399.0] ms |
| Decider vs Text Heuristic | Text Heuristic (950ms) | Strands Decider 2B (hi=0.55) | +1353.2 ms | [+1304.6, +1399.0] ms |
| Decider+Profile vs Decider | Strands Decider 2B (hi=0.55) | Strands Decider 2B + Profile (hi=0.50) | +0.0 ms | [+0.0, +0.0] ms |
| Decider+Profile vs Per-User Control | Per-User Control (scale 1.10) | Strands Decider 2B + Profile (hi=0.50) | +1508.3 ms | [+1403.8, +1604.5] ms |

### Dominance Analysis at Operating Points
- Fixed timeouts and text heuristic dominate Decider at both operating points: At Point A (equal delay budget <= 800ms), Text Heuristic achieves 800ms delay vs Decider 1501ms with comparable cut-in rate. At Point B (premature budget <= 5%), Fixed 950ms and Text Heuristic 950ms achieve 950ms delay with 3.3% and 2.5% cut-ins, whereas Decider waited 2200ms (+1250ms slower) to eliminate cut-ins.

## 1B. Exploratory Full Benchmark Comparison (19 Latency & Sensitivity Configurations)

| Approach | Accuracy (Cluster 95% CI) | Accuracy (Wilson 95% CI) | Premature Interruption (Cluster 95% CI) | Premature (Wilson 95% CI) | Unnecessary Wait (Cluster 95% CI) | Unnecessary Wait (Wilson 95% CI) | Delay P50 (Bootstrap 95% CI) | Delay P95 | Latency Condition |
|---|---|---|---|---|---|---|---|---|---|
| Fixed 600ms Baseline | 63.3% [54.7%, 72.2%] | [58.2%, 68.2%] | 34.2% [24.2%, 44.2%] | [29.5%, 39.2%] | 5.0% [1.1%, 9.9%] | [2.6%, 9.2%] | 600 ms [600.0, 600.0] ms | 600 ms | Fixed timeout |
| Fixed 800ms Baseline | 64.4% [56.7%, 72.2%] | [59.4%, 69.2%] | 18.9% [11.9%, 26.4%] | [15.2%, 23.2%] | 33.1% [17.1%, 49.5%] | [26.7%, 40.3%] | 800 ms [800.0, 800.0] ms | 800 ms | Fixed timeout |
| Fixed 1200ms Baseline | 70.6% [60.0%, 80.0%] | [65.6%, 75.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 100.0% [100.0%, 100.0%] | [97.9%, 100.0%] | 1010 ms [910.0, 1196.0] ms | 1200 ms | Fixed timeout |
| Text Heuristic Baseline (250ms, dev-tuned) | 75.3% [69.7%, 80.6%] | [70.6%, 79.5%] | 24.7% [19.4%, 30.3%] | [20.5%, 29.4%] | 0.0% [0.0%, 0.0%] | [0.0%, 2.1%] | 250 ms [250.0, 250.0] ms | 250 ms | dev-tuned |
| Per-User Text Heuristic (Profile Threshold, No Model) | 86.9% [82.2%, 91.1%] | [83.1%, 90.0%] | 12.8% [8.3%, 17.8%] | [9.7%, 16.6%] | 32.6% [16.4%, 49.7%] | [26.2%, 39.7%] | 707 ms [661.0, 766.0] ms | 928 ms | profile threshold (no model) |
| Strands Decider 2B (301ms GPU, cache-measured on T4) | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 94.5% [89.9%, 98.3%] | [90.1%, 97.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | Tesla T4 GPU (cache-measured median, actual evaluation states) |
| Strands Decider 2B + Profile (301ms GPU, cache-measured on T4) | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 100.0% [100.0%, 100.0%] | [97.9%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | GPU cache + profile |
| Strands Decider 2B + Calibrated Policy [REAL-CALIBRATED] (hi=0.55, lo=0.10, 301ms GPU, cache-measured on T4) | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 100.0% [100.0%, 100.0%] | [97.9%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | calibrated policy, virtual clock |
| Strands Decider 2B (235.6ms GPU benchmark, synthetic 20-150 tokens on T4) | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 94.5% [89.9%, 98.3%] | [90.1%, 97.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | bench_decider on synthetic 20 to 150 token states on a T4 |
| Strands Decider 2B (0 ms decision latency) | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 94.5% [89.9%, 98.3%] | [90.1%, 97.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | 0 ms virtual oracle |
| Strands Decider 2B (3190ms CPU, Ryzen 5 5500U) | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 100.0% [100.0%, 100.0%] | [97.9%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | AMD Ryzen 5 5500U (measured CPU median) |
| Decider Sweep (100ms Sweep) | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 94.5% [89.9%, 98.3%] | [90.1%, 97.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | GPU cache, virtual clock |
| Decider Sweep (300ms Sweep) | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 94.5% [89.9%, 98.3%] | [90.1%, 97.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | GPU cache, virtual clock |
| Decider Sweep (1000ms Sweep) | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 100.0% [100.0%, 100.0%] | [97.9%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | GPU cache, virtual clock |
| Decider Sweep (3000ms Sweep) | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 100.0% [100.0%, 100.0%] | [97.9%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | GPU cache, virtual clock |
| PERSONA-BLIND REFERENCE (single first-checkpoint, not a ceiling) | 74.7% [68.9%, 80.0%] | [70.0%, 78.9%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 50.3% [42.6%, 57.5%] | [43.1%, 57.5%] | 2200 ms [250.0, 2200.0] ms | 2200 ms | Single first-checkpoint reference, not a ceiling |
| ARCHETYPE-AWARE REFERENCE (single first-checkpoint, not a ceiling) | 81.4% [76.4%, 85.8%] | [77.0%, 85.1%] | 5.0% [1.9%, 9.2%] | [3.2%, 7.8%] | 27.1% [17.0%, 37.9%] | [21.1%, 34.0%] | 250 ms [250.0, 250.0] ms | 2200 ms | Single first-checkpoint reference, not a ceiling |
| USER-AWARE REFERENCE (single first-checkpoint, not a ceiling) | 81.9% [76.9%, 86.7%] | [77.6%, 85.6%] | 3.9% [1.4%, 7.0%] | [2.3%, 6.4%] | 28.2% [17.9%, 39.0%] | [22.1%, 35.1%] | 250 ms [250.0, 250.0] ms | 2200 ms | Single first-checkpoint reference, not a ceiling |
| ORACLE-UPPER-BOUND REFERENCE (single first-checkpoint, not a ceiling) | 81.4% [76.4%, 85.8%] | [77.0%, 85.1%] | 5.0% [1.9%, 9.2%] | [3.2%, 7.8%] | 27.1% [17.0%, 37.9%] | [21.1%, 34.0%] | 250 ms [250.0, 250.0] ms | 2200 ms | Single first-checkpoint reference, not a ceiling |
| Bedrock LLM Judge | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED (not run in this eval) |

### Reference Rows Paired Differences (Single First-Checkpoint Reference, Not a Ceiling)

| Comparison | Reference A | Reference B | Paired Accuracy Difference (B - A) | Cluster Bootstrap 95% CI |
|---|---|---|---|---|
| Archetype vs Blind | PERSONA-BLIND REFERENCE | ARCHETYPE-AWARE REFERENCE | +6.7% | [+1.1%, +13.1%] |
| User vs Archetype | ARCHETYPE-AWARE REFERENCE | USER-AWARE REFERENCE | +0.6% | [-2.8%, +4.2%] |
| User vs Blind | PERSONA-BLIND REFERENCE | USER-AWARE REFERENCE | +7.2% | [+1.9%, +13.9%] |

## 2. Per-Persona Performance Breakdown

| Approach | Fast Pauser Premature % | Fast Pauser Delay P50 | Slow Pauser Premature % | Slow Pauser Delay P50 | Filler-Heavy Premature % | Filler-Heavy Delay P50 |
|---|---|---|---|---|---|---|
| Fixed 600ms Baseline | 0.0% | 600 ms | 50.8% | 600 ms | 51.7% | 600 ms |
| Fixed 800ms Baseline | 0.0% | 695 ms | 34.2% | 800 ms | 22.5% | 800 ms |
| Fixed 1200ms Baseline | 0.0% | 695 ms | 0.0% | 1200 ms | 0.0% | 1098 ms |
| Text Heuristic Baseline (250ms, dev-tuned) | 15.0% | 250 ms | 25.8% | 250 ms | 33.3% | 250 ms |
| Per-User Text Heuristic (Profile Threshold, No Model) | 8.3% | 343 ms | 12.5% | 884 ms | 17.5% | 716 ms |
| Strands Decider 2B (301ms GPU, cache-measured on T4) | 0.0% | 2200 ms | 0.0% | 2200 ms | 0.0% | 2200 ms |
| Strands Decider 2B + Profile (301ms GPU, cache-measured on T4) | 0.0% | 2200 ms | 0.0% | 2200 ms | 0.0% | 2200 ms |
| Strands Decider 2B + Calibrated Policy [REAL-CALIBRATED] (hi=0.55, lo=0.10, 301ms GPU, cache-measured on T4) | 0.0% | 2200 ms | 0.0% | 2200 ms | 0.0% | 2200 ms |
| Strands Decider 2B (235.6ms GPU benchmark, synthetic 20-150 tokens on T4) | 0.0% | 2200 ms | 0.0% | 2200 ms | 0.0% | 2200 ms |
| Strands Decider 2B (0 ms decision latency) | 0.0% | 2200 ms | 0.0% | 2200 ms | 0.0% | 2200 ms |
| Strands Decider 2B (3190ms CPU, Ryzen 5 5500U) | 0.0% | 2200 ms | 0.0% | 2200 ms | 0.0% | 2200 ms |
| PERSONA-BLIND REFERENCE (single first-checkpoint, not a ceiling) | 0.0% | 2200 ms | 0.0% | 250 ms | 0.0% | 250 ms |
| ARCHETYPE-AWARE REFERENCE (single first-checkpoint, not a ceiling) | 15.0% | 250 ms | 0.0% | 250 ms | 0.0% | 250 ms |
| USER-AWARE REFERENCE (single first-checkpoint, not a ceiling) | 8.3% | 250 ms | 3.3% | 250 ms | 0.0% | 250 ms |
| ORACLE-UPPER-BOUND REFERENCE (single first-checkpoint, not a ceiling) | 15.0% | 250 ms | 0.0% | 250 ms | 0.0% | 250 ms |

## 3. Per-Category Performance Breakdown

| Approach | Complete Accuracy % | Complete Delay P50 | Incomplete Premature % | Ambiguous Accuracy % | Ambiguous Premature % |
|---|---|---|---|---|---|
| Fixed 600ms Baseline | 97.8% | 600 ms | 62.2% | 58.9% | 37.2% |
| Fixed 800ms Baseline | 73.3% | 800 ms | 31.1% | 57.8% | 22.2% |
| Fixed 1200ms Baseline | 43.3% | 1078 ms | 0.0% | 69.4% | 0.0% |
| Text Heuristic Baseline (250ms, dev-tuned) | 100.0% | 250 ms | 0.0% | 50.6% | 49.4% |
| Per-User Text Heuristic (Profile Threshold, No Model) | 100.0% | 738 ms | 0.0% | 73.9% | 25.6% |
| Strands Decider 2B (301ms GPU, cache-measured on T4) | 100.0% | 2200 ms | 0.0% | 100.0% | 0.0% |
| Strands Decider 2B + Profile (301ms GPU, cache-measured on T4) | 100.0% | 2200 ms | 0.0% | 100.0% | 0.0% |
| Strands Decider 2B + Calibrated Policy [REAL-CALIBRATED] (hi=0.55, lo=0.10, 301ms GPU, cache-measured on T4) | 100.0% | 2200 ms | 0.0% | 100.0% | 0.0% |
| Strands Decider 2B (235.6ms GPU benchmark, synthetic 20-150 tokens on T4) | 100.0% | 2200 ms | 0.0% | 100.0% | 0.0% |
| Strands Decider 2B (0 ms decision latency) | 100.0% | 2200 ms | 0.0% | 100.0% | 0.0% |
| Strands Decider 2B (3190ms CPU, Ryzen 5 5500U) | 100.0% | 2200 ms | 0.0% | 100.0% | 0.0% |
| PERSONA-BLIND REFERENCE (single first-checkpoint, not a ceiling) | 100.0% | 250 ms | 0.0% | 49.4% | 0.0% |
| ARCHETYPE-AWARE REFERENCE (single first-checkpoint, not a ceiling) | 100.0% | 250 ms | 0.0% | 62.8% | 10.0% |
| USER-AWARE REFERENCE (single first-checkpoint, not a ceiling) | 100.0% | 250 ms | 0.0% | 63.9% | 7.8% |
| ORACLE-UPPER-BOUND REFERENCE (single first-checkpoint, not a ceiling) | 100.0% | 250 ms | 0.0% | 62.8% | 10.0% |

## 4. Policy Calibration, Probability Reliability & Pareto Frontiers

### Policy Threshold Calibration (DEV Split Only)
- **Grid Search Space**: 64 candidate (hi, lo) pairs searched on DEV split (180 turns).
- **Dev-Selected Optimal Thresholds**: `hi = 0.55`, `lo = 0.10`, `max_hold_ms = 2200 ms` [REAL-CALIBRATED].
- **Dev Calibration Metrics**: Accuracy: 100.0%, Premature: 0.0%, Score: 1.0000.
- **Test Set Zero Leakage**: Test split was untouched during calibration (evaluated strictly out-of-sample).

### Probability Reliability (Calibration on Probabilities, Not Confidence)
- **DEV Split Brier Score**: 0.3476 (over 644 checkpoint predictions).
- **TEST Split Brier Score**: 0.3452 (over 1282 checkpoint predictions).
- **Reliability Artifacts**: [results_reliability.png](backend/eval/results/results_reliability.png) | [reliability_data.json](backend/eval/results/reliability_data.json).

### Turn-Taking Pareto Frontiers (Premature Interruption Rate vs Response Delay)
- Swept Fixed Timeouts (300ms to 2000ms), Text Heuristic Thresholds (100ms to 1200ms), and Decider Policy hi sweeps.
- Evaluated across 3 latency conditions: 0 ms virtual oracle, 301 ms GPU (Tesla T4), and 3190 ms CPU.
- **Pareto Artifacts**: [results_pareto.png](backend/eval/results/results_pareto.png) | [pareto_data.json](backend/eval/results/pareto_data.json).

## 5. Decider Hardware Latency Benchmarks & B0 Validation Set

### Measured Decider Latency on GPU (Tesla T4, Google Colab)
- **Device**: `cuda` (GPU: Tesla T4, real model weights)
- **Warnings**: `causal_conv1d` and `flash-linear-attention` missing (reference path execution)

| State Tokens | 1 Question Median (ms) | 1 Question P95 (ms) | 2 Questions Median (ms) | 2 Questions P95 (ms) | 2Q Overhead (ms) |
|---|---|---|---|---|---|
| 20 tokens | 167.5 | 176.0 | 343.1 | 343.7 | +175.6 |
| 60 tokens | 235.6 | 236.7 | 395.2 | 400.3 | +159.6 |
| 150 tokens | 237.3 | 238.5 | 395.8 | 401.1 | +158.5 |

### Decider B0 40-Case Small Validation Set
- Number of cases: 40
- Fixed 800ms Baseline Accuracy: 40.0% (16/40)
- Strands Decider 2B (Real Weights): 85.0% (34/40) [backend: local_cpu (AMD Ryzen 5 5500U, b0_spike.py)]
- Strands Decider 2B [MOCK-DECIDER]: 75.0% (30/40) [backend: mock (MockDecider)]

## 6. Metric Definitions & Protocol
- **Premature Interruption Rate**: Fraction of all test turns where the system yielded `RESPOND` while speaker was mid-thought. Numerator = premature interruption count, Denominator = 360 (total test turns).
- **Unnecessary Wait Rate**: Fraction of completed turns where agent waited > 800 ms before responding. Numerator = delayed turns, Denominator = 181 (total completed turns). Threshold is global 800 ms.
- **Response Delay (P50/P95)**: Elapsed time in milliseconds from speech end to agent floor yield, evaluated exclusively over completed turns (n=181).
- **Accuracy**: Binary turn-taking decision accuracy over all turns (n=360).
- **Uncertainty Intervals**: Headline tables report non-parametric two-stage cluster bootstrap 95% intervals (B=1000 resamples: sample users with replacement, then sample turns within user with replacement) to account for turn clustering across 30 users (12 turns/user). Secondary column reports standard Wilson 1927 score intervals for comparison.
- **Single First-Checkpoint Reference Rows (Not Ceilings)**: Labeled 'single first-checkpoint reference, not a ceiling' (formerly reference ceilings). These rows evaluate decisions only at the first checkpoint (250 ms) without multi-checkpoint dynamics. The archetype-aware vs user-aware gap is about 0.5 points (81.4% vs 81.9%), meaning little per-user headroom in this dataset (a limitation). Note that the per-user control heuristic (87.2%) beats the user-aware reference (81.9%), proving that single-checkpoint references are not upper bounds for multi-checkpoint policies. Multi-checkpoint reference from known pause distributions was skipped under the 20-minute limit.

## 7. Findings & Observations

### A. Operating Points Performance (Points A & B)
- **Point A (Delay Budget <= 800ms)**: Strands Decider 2B achieves 10.0% premature rate and 1501 ms delay P50 (2200 ms P95). The Fixed Timeout (800ms) yields 18.9% premature rate and 800 ms delay P50. The Text Heuristic (800ms) yields 11.1% premature rate and 800 ms delay P50. The Per-User Control yields 4.2% premature rate and 808 ms delay P50. Decider+Profile achieves 8.1% premature rate and 551 ms delay P50.
- **Point B (Premature Budget <= 5.0%)**: Strands Decider 2B achieves 0.0% premature rate and 2200 ms delay P50 (2200 ms P95). The Fixed Timeout (950ms) yields 3.3% premature rate and 950 ms delay P50. The Text Heuristic (950ms) yields 2.5% premature rate and 950 ms delay P50. The Per-User Control yields 7.5% premature rate and 778 ms delay P50. Decider+Profile achieves 0.0% premature rate and 2200 ms delay P50.
- **Dominance Analysis**: Fixed timeouts and text heuristic dominate Decider at both operating points: At Point A (equal delay budget <= 800ms), Text Heuristic achieves 800ms delay vs Decider 1501ms with comparable cut-in rate. At Point B (premature budget <= 5%), Fixed 950ms and Text Heuristic 950ms achieve 950ms delay with 3.3% and 2.5% cut-ins, whereas Decider waited 2200ms (+1250ms slower) to eliminate cut-ins.

### B. Paired Differences at Operating Points
- **Point A (Premature Rate Difference at Delay Budget <= 800ms)**: Decider minus Fixed 800ms: -8.9% [-15.6%, -3.0%]; Decider minus Text Heuristic: -1.1% [-5.6%, +3.3%]; Decider+Profile minus Decider: -1.9% [-5.8%, +1.1%]; Decider+Profile minus Per-User Control: +3.9% [-1.1%, +8.6%].
- **Point B (Response Delay Difference at Premature Budget <= 5%)**: Decider minus Fixed 950ms: +1353.2 ms [+1304.6, +1399.0] ms; Decider minus Text Heuristic: +1353.2 ms [+1304.6, +1399.0] ms; Decider+Profile minus Decider: +0.0 ms [+0.0, +0.0] ms; Decider+Profile minus Per-User Control: +1508.3 ms [+1403.8, +1604.5] ms.

### C. Single First-Checkpoint Reference Rows (Not Ceilings)
- **PERSONA-BLIND REFERENCE**: Accuracy 74.7%, Premature 0.0%, Delay P50 2200 ms.
- **ARCHETYPE-AWARE REFERENCE**: Accuracy 81.4%, Premature 5.0%, Delay P50 250 ms.
- **USER-AWARE REFERENCE**: Accuracy 81.9%, Premature 3.9%, Delay P50 250 ms.
- **Headroom Analysis**: The gap between ARCHETYPE-AWARE REFERENCE (81.4%) and USER-AWARE REFERENCE (81.9%) is +0.6% [-2.8%, +4.2%], demonstrating little per-user headroom in this dataset (a dataset limitation). The Per-User Control heuristic (86.9% accuracy) beats the user-aware reference (81.9%), proving that single-checkpoint references are not upper bounds for multi-checkpoint policies. Multi-checkpoint reference from known pause distributions was skipped under the 20-minute limit.

### D. Calibration, Reliability & Conflict Rule
- **DEV Calibration**: Calibrated thresholds hi = 0.55, lo = 0.10, max_hold_ms = 2200 ms (DEV score = 1.0000).
- **Probability Reliability**: DEV Brier score = 0.3476 (644 samples); TEST Brier score = 0.3452 (1282 samples).
- **Q1+Q2 Conflict Rule Effect**: Across 1800 evaluated pause checkpoints on the test split, exactly 0 checkpoints exhibited conflict (Q1 p(RESPOND) >= 0.55 and Q2 p(YES) >= 0.50).

### E. Outcome & Summary Verdict
- **Plain Language Outcome**: Across both operating points (Point A equal delay budget and Point B equal premature budget), the Fixed Timeout and Text Heuristic baselines dominate the Strands Decider 2B. At Point A, Fixed Timeout achieves 800.0 ms delay P50, while Decider exhibits 1501.2 ms delay P50. At Point B, Fixed Timeout (950ms) achieves a 3.3% premature rate and 950.0 ms delay P50, whereas Decider exhibits 2200.0 ms delay P50 (+1353.2 ms slower). Decider does not beat the baselines at equal delay or equal premature budget.
