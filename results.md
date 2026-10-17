# Benchmark Results — Patient Listener

> [!WARNING]
> **PROVENANCE & MOCK NOTICE**: All Decider rows below were generated using `MockDecider` cache entries (`backend=mock`, 448 keys, compute time 0.00 ms). They are marked `[MOCK-DECIDER]` and must NOT be cited as measured performance of the real weights. On ambiguous turns, the text heuristic baseline achieves matching accuracy to the mock decider; no real Decider run yet.

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

## 1. Test Split Turn-Taking Comparison (n=360 turns across 30 users, 12 turns/user)

| Approach | Accuracy (Cluster 95% CI) | Accuracy (Wilson 95% CI) | Premature Interruption (Cluster 95% CI) | Premature (Wilson 95% CI) | Unnecessary Wait (Cluster 95% CI) | Unnecessary Wait (Wilson 95% CI) | Delay P50 (Bootstrap 95% CI) | Delay P95 | Latency Condition |
|---|---|---|---|---|---|---|---|---|---|
| Fixed 600ms Baseline | 63.3% [54.2%, 72.2%] | [58.2%, 68.2%] | 34.2% [24.4%, 43.9%] | [29.5%, 39.2%] | 5.0% [1.1%, 10.5%] | [2.6%, 9.2%] | 600 ms [600.0, 600.0] ms | 600 ms | Fixed |
| Fixed 800ms Baseline | 64.4% [56.9%, 71.4%] | [59.4%, 69.2%] | 18.9% [11.9%, 26.1%] | [15.2%, 23.2%] | 33.1% [17.5%, 50.3%] | [26.7%, 40.3%] | 800 ms [800.0, 800.0] ms | 800 ms | Fixed |
| Fixed 1200ms Baseline | 70.6% [60.0%, 80.0%] | [65.6%, 75.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 100.0% [100.0%, 100.0%] | [97.9%, 100.0%] | 1010 ms [910.0, 1182.0] ms | 1200 ms | Fixed |
| Text Heuristic Baseline (250ms, dev-tuned) | 75.3% [70.0%, 80.6%] | [70.6%, 79.5%] | 24.7% [19.4%, 30.0%] | [20.5%, 29.4%] | 0.0% [0.0%, 0.0%] | [0.0%, 2.1%] | 250 ms [250.0, 250.0] ms | 250 ms | dev-tuned |
| Strands Decider 2B (0 ms decision latency) [MOCK-DECIDER] | 75.3% [70.0%, 80.6%] | [70.6%, 79.5%] | 24.7% [19.4%, 30.0%] | [20.5%, 29.4%] | 0.0% [0.0%, 0.0%] | [0.0%, 2.1%] | 250 ms [250.0, 250.0] ms | 250 ms | mock cache, virtual clock |
| Strands Decider 2B (236ms GPU, Tesla T4) [MOCK-DECIDER] | 80.3% [73.6%, 86.9%] | [75.9%, 84.1%] | 19.7% [13.1%, 26.4%] | [15.9%, 24.1%] | 0.0% [0.0%, 0.0%] | [0.0%, 2.1%] | 486 ms [485.6, 485.6] ms | 486 ms | mock cache, virtual clock |
| Strands Decider 2B (3190ms CPU) [MOCK-DECIDER] | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 100.0% [100.0%, 100.0%] | [97.9%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | mock cache, virtual clock |
| Decider Sweep (100ms Sweep) [MOCK-DECIDER] | 76.9% [71.1%, 83.1%] | [72.3%, 81.0%] | 23.1% [16.9%, 28.9%] | [19.0%, 27.7%] | 0.0% [0.0%, 0.0%] | [0.0%, 2.1%] | 350 ms [350.0, 350.0] ms | 350 ms | mock cache, virtual clock |
| Decider Sweep (300ms Sweep) [MOCK-DECIDER] | 81.4% [75.0%, 87.8%] | [77.0%, 85.1%] | 18.6% [12.2%, 25.0%] | [14.9%, 23.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 2.1%] | 550 ms [550.0, 550.0] ms | 550 ms | mock cache, virtual clock |
| Decider Sweep (1000ms Sweep) [MOCK-DECIDER] | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 100.0% [100.0%, 100.0%] | [97.9%, 100.0%] | 1250 ms [1250.0, 1250.0] ms | 1250 ms | mock cache, virtual clock |
| Decider Sweep (3000ms Sweep) [MOCK-DECIDER] | 100.0% [100.0%, 100.0%] | [98.9%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 100.0% [100.0%, 100.0%] | [97.9%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | mock cache, virtual clock |
| PERSONA-BLIND CEILING | 74.7% [68.6%, 80.3%] | [70.0%, 78.9%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.1%] | 50.3% [41.9%, 59.2%] | [43.1%, 57.5%] | 2200 ms [250.0, 2200.0] ms | 2200 ms | Reference ceiling (blind) |
| ORACLE-UPPER-BOUND (Persona-Aware) | 81.4% [76.4%, 86.4%] | [77.0%, 85.1%] | 5.0% [1.9%, 9.2%] | [3.2%, 7.8%] | 27.1% [16.6%, 38.4%] | [21.1%, 34.0%] | 250 ms [250.0, 250.0] ms | 2200 ms | Reference ceiling (persona-aware) |
| Bedrock LLM Judge | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED (no AWS credentials) |

## 2. Per-Persona Performance Breakdown

| Approach | Fast Pauser Premature % | Fast Pauser Delay P50 | Slow Pauser Premature % | Slow Pauser Delay P50 | Filler-Heavy Premature % | Filler-Heavy Delay P50 |
|---|---|---|---|---|---|---|
| Fixed 600ms Baseline | 0.0% | 600 ms | 50.8% | 600 ms | 51.7% | 600 ms |
| Fixed 800ms Baseline | 0.0% | 695 ms | 34.2% | 800 ms | 22.5% | 800 ms |
| Fixed 1200ms Baseline | 0.0% | 695 ms | 0.0% | 1200 ms | 0.0% | 1098 ms |
| Text Heuristic Baseline (250ms, dev-tuned) | 15.0% | 250 ms | 25.8% | 250 ms | 33.3% | 250 ms |
| Strands Decider 2B (0 ms decision latency) [MOCK-DECIDER] | 15.0% | 250 ms | 25.8% | 250 ms | 33.3% | 250 ms |
| Strands Decider 2B (236ms GPU, Tesla T4) [MOCK-DECIDER] | 0.0% | 486 ms | 25.8% | 486 ms | 33.3% | 486 ms |
| Strands Decider 2B (3190ms CPU) [MOCK-DECIDER] | 0.0% | 2200 ms | 0.0% | 2200 ms | 0.0% | 2200 ms |
| PERSONA-BLIND CEILING | 0.0% | 2200 ms | 0.0% | 250 ms | 0.0% | 250 ms |
| ORACLE-UPPER-BOUND (Persona-Aware) | 15.0% | 250 ms | 0.0% | 250 ms | 0.0% | 250 ms |

## 3. Per-Category Performance Breakdown

| Approach | Complete Accuracy % | Complete Delay P50 | Incomplete Premature % | Ambiguous Accuracy % | Ambiguous Premature % |
|---|---|---|---|---|---|
| Fixed 600ms Baseline | 97.8% | 600 ms | 62.2% | 58.9% | 37.2% |
| Fixed 800ms Baseline | 73.3% | 800 ms | 31.1% | 57.8% | 22.2% |
| Fixed 1200ms Baseline | 43.3% | 1078 ms | 0.0% | 69.4% | 0.0% |
| Text Heuristic Baseline (250ms, dev-tuned) | 100.0% | 250 ms | 0.0% | 50.6% | 49.4% |
| Strands Decider 2B (0 ms decision latency) [MOCK-DECIDER] | 100.0% | 250 ms | 0.0% | 50.6% | 49.4% |
| Strands Decider 2B (236ms GPU, Tesla T4) [MOCK-DECIDER] | 100.0% | 486 ms | 0.0% | 60.6% | 39.4% |
| Strands Decider 2B (3190ms CPU) [MOCK-DECIDER] | 100.0% | 2200 ms | 0.0% | 100.0% | 0.0% |
| PERSONA-BLIND CEILING | 100.0% | 250 ms | 0.0% | 49.4% | 0.0% |
| ORACLE-UPPER-BOUND (Persona-Aware) | 100.0% | 250 ms | 0.0% | 62.8% | 10.0% |

## 4. Decider Hardware Latency Benchmarks & B0 Validation Set

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
- Strands Decider 2B [MOCK-DECIDER]: 82.5% (33/40) [backend: mock (MockDecider)]

## 5. Metric Definitions & Protocol
- **Premature Interruption Rate**: Fraction of all test turns where the system yielded `RESPOND` while speaker was mid-thought. Numerator = premature interruption count, Denominator = 360 (total test turns).
- **Unnecessary Wait Rate**: Fraction of completed turns where agent waited > 800 ms before responding. Numerator = delayed turns, Denominator = 181 (total completed turns). Threshold is global 800 ms.
- **Response Delay (P50/P95)**: Elapsed time in milliseconds from speech end to agent floor yield, evaluated exclusively over completed turns (n=181).
- **Accuracy**: Binary turn-taking decision accuracy over all turns (n=360).
- **Uncertainty Intervals**: Headline tables report non-parametric two-stage cluster bootstrap 95% intervals (B=1000 resamples: sample users with replacement, then sample turns within user with replacement) to account for turn clustering across 9 users (36 turns/user). Secondary column reports standard Wilson 1927 score intervals for comparison.
- **Reference Ceilings**: `PERSONA-BLIND CEILING` uses text category and global majority continuation (WAIT), achieving 77.2% (250/324). `ORACLE-UPPER-BOUND (Persona-Aware)` uses archetype parameters to achieve 82.1% (266/324). Neither is an achievable baseline; both serve as theoretical bounds.

## 6. Findings & Observations
- **Ambiguity Impact**: On ambiguous utterances where complete-sounding clauses have persona-sampled continuations, the Text Heuristic achieves 50.6% accuracy with 49.4% premature interruptions, matching the mock decider; no real Decider run yet.
- **Persona Pause Dilemma**: Fixed 600ms causes 50.8% premature interruptions on slow pausers versus 0.0% on fast pausers.
- **CPU Latency Degeneration**: At 3190 ms CPU latency, speculative queries arrive after the 2200 ms max_wait timeout, degenerating into a fixed 2200 ms timeout with 100.0% unnecessary wait.
- **Cache & Replay Protocol**: Replay verified with 2904 cache hits and 0 cache misses (zero fallback).
