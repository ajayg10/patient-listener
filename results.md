# Benchmark Results — Patient Listener

> [!WARNING]
> **PROVENANCE & MOCK NOTICE**: All Decider rows below were generated using `MockDecider` cache entries (`backend=mock`, 448 keys, compute time 0.00 ms). They are marked `[MOCK-DECIDER]` and must NOT be cited as measured performance of the real weights. On ambiguous turns, the text heuristic baseline achieves matching accuracy to the mock decider; no real Decider run yet.

> [!IMPORTANT]
> **SYNTHETIC BENCHMARK**: Evaluated on 324 synthetic test turns (154 complete, 170 incomplete) across 9 users (36 turns/user) in 3 conflicting persona archetypes (fast, slow, filler-heavy). Zero leakage: dev and test splits strictly partition users and templates.

### Dataset Reconciled Count Table (Current Split)

| Split | Turns | Users | Unique Utterances | Lookups | Unique States Reached | Cache Hits | Misses |
|---|---|---|---|---|---|---|---|
| DEV | 192 | 12 | 24 | 685 | 104 | 685 | 0 |
| TEST | 324 | 9 | 36 | 1146 | 155 | 1146 | 0 |
| TOTAL | 516 | 21 | 60 | 1831 | 259 | 1831 | 0 |

- **Lookups difference (dev 664 -> 685, test 1,136 -> 1,146)**: Increasing ambiguous share to 50% in B5.8 extended silences so more turns reached later pause checkpoints before speech resumed.
- **States difference (test 156 -> 155)**: In the B5.8 test realization, one slow template never experienced a silence lasting to the 1800 ms checkpoint.
- **2,250 vs 1,831 lookups**: 1,831 (685 dev + 1,146 test) is the single-pass lookup count, whereas 2,250 was the cumulative query count across all latency sweeps in `run_eval.py`.

## 1. Test Split Turn-Taking Comparison (n=324 turns across 9 users, 36 turns/user)

| Approach | Accuracy (Cluster 95% CI) | Accuracy (Wilson 95% CI) | Premature Interruption (Cluster 95% CI) | Premature (Wilson 95% CI) | Unnecessary Wait (Cluster 95% CI) | Unnecessary Wait (Wilson 95% CI) | Delay P50 (Bootstrap 95% CI) | Delay P95 | Latency Condition |
|---|---|---|---|---|---|---|---|---|---|
| Fixed 600ms Baseline | 58.3% [44.8%, 72.8%] | [52.9%, 63.6%] | 37.3% [20.1%, 54.3%] | [32.3%, 42.7%] | 9.1% [0.0%, 18.3%] | [5.5%, 14.7%] | 600 ms [600.0, 600.0] ms | 600 ms | Fixed |
| Fixed 800ms Baseline | 59.6% [50.0%, 69.4%] | [54.1%, 64.8%] | 22.2% [10.2%, 34.6%] | [18.0%, 27.1%] | 38.3% [8.9%, 64.0%] | [31.0%, 46.2%] | 800 ms [800.0, 800.0] ms | 800 ms | Fixed |
| Fixed 1200ms Baseline | 70.7% [51.8%, 88.3%] | [65.5%, 75.4%] | 0.3% [0.0%, 1.2%] | [0.1%, 1.7%] | 100.0% [100.0%, 100.0%] | [97.6%, 100.0%] | 972 ms [821.0, 1162.0] ms | 1200 ms | Fixed |
| Text Heuristic Baseline (250ms, dev-tuned) | 72.5% [61.7%, 83.0%] | [67.4%, 77.1%] | 27.5% [17.0%, 38.3%] | [22.9%, 32.6%] | 0.0% [0.0%, 0.0%] | [0.0%, 2.4%] | 250 ms [250.0, 250.0] ms | 250 ms | dev-tuned |
| Strands Decider 2B (0 ms decision latency) [MOCK-DECIDER] | 72.5% [61.7%, 83.0%] | [67.4%, 77.1%] | 27.5% [17.0%, 38.3%] | [22.9%, 32.6%] | 0.0% [0.0%, 0.0%] | [0.0%, 2.4%] | 250 ms [250.0, 250.0] ms | 250 ms | mock cache, virtual clock |
| Strands Decider 2B (236ms GPU, Tesla T4) [MOCK-DECIDER] | 75.6% [63.0%, 88.0%] | [70.7%, 80.0%] | 24.4% [12.0%, 37.0%] | [20.0%, 29.3%] | 0.0% [0.0%, 0.0%] | [0.0%, 2.4%] | 486 ms [485.6, 485.6] ms | 486 ms | mock cache, virtual clock |
| Strands Decider 2B (3190ms CPU) [MOCK-DECIDER] | 100.0% [100.0%, 100.0%] | [98.8%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.2%] | 100.0% [100.0%, 100.0%] | [97.6%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | mock cache, virtual clock |
| Decider Sweep (100ms Sweep) [MOCK-DECIDER] | 73.5% [62.0%, 84.3%] | [68.4%, 78.0%] | 26.5% [15.7%, 38.0%] | [22.0%, 31.6%] | 0.0% [0.0%, 0.0%] | [0.0%, 2.4%] | 350 ms [350.0, 350.0] ms | 350 ms | mock cache, virtual clock |
| Decider Sweep (300ms Sweep) [MOCK-DECIDER] | 76.9% [64.2%, 88.6%] | [72.0%, 81.1%] | 23.1% [11.4%, 35.8%] | [18.9%, 28.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 2.4%] | 550 ms [550.0, 550.0] ms | 550 ms | mock cache, virtual clock |
| Decider Sweep (1000ms Sweep) [MOCK-DECIDER] | 100.0% [100.0%, 100.0%] | [98.8%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.2%] | 100.0% [100.0%, 100.0%] | [97.6%, 100.0%] | 1250 ms [1250.0, 1250.0] ms | 1250 ms | mock cache, virtual clock |
| Decider Sweep (3000ms Sweep) [MOCK-DECIDER] | 100.0% [100.0%, 100.0%] | [98.8%, 100.0%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.2%] | 100.0% [100.0%, 100.0%] | [97.6%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | mock cache, virtual clock |
| PERSONA-BLIND CEILING | 77.5% [66.7%, 88.3%] | [72.6%, 81.7%] | 0.0% [0.0%, 0.0%] | [0.0%, 1.2%] | 47.4% [30.9%, 59.5%] | [39.7%, 55.3%] | 250 ms [250.0, 2200.0] ms | 2200 ms | Reference ceiling (blind) |
| ORACLE-UPPER-BOUND (Persona-Aware) | 87.3% [79.6%, 93.8%] | [83.3%, 90.5%] | 3.4% [0.0%, 7.7%] | [1.9%, 6.0%] | 19.5% [4.6%, 38.4%] | [14.0%, 26.5%] | 250 ms [250.0, 250.0] ms | 2200 ms | Reference ceiling (persona-aware) |
| Bedrock LLM Judge | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED (no AWS credentials) |

## 2. Per-Persona Performance Breakdown

| Approach | Fast Pauser Premature % | Fast Pauser Delay P50 | Slow Pauser Premature % | Slow Pauser Delay P50 | Filler-Heavy Premature % | Filler-Heavy Delay P50 |
|---|---|---|---|---|---|---|
| Fixed 600ms Baseline | 0.0% | 600 ms | 53.7% | 600 ms | 58.3% | 600 ms |
| Fixed 800ms Baseline | 0.0% | 668 ms | 39.8% | 800 ms | 26.9% | 800 ms |
| Fixed 1200ms Baseline | 0.0% | 668 ms | 0.0% | 1200 ms | 0.9% | 1122 ms |
| Text Heuristic Baseline (250ms, dev-tuned) | 10.2% | 250 ms | 28.7% | 250 ms | 43.5% | 250 ms |
| Strands Decider 2B (0 ms decision latency) [MOCK-DECIDER] | 10.2% | 250 ms | 28.7% | 250 ms | 43.5% | 250 ms |
| Strands Decider 2B (236ms GPU, Tesla T4) [MOCK-DECIDER] | 0.9% | 486 ms | 28.7% | 486 ms | 43.5% | 486 ms |
| Strands Decider 2B (3190ms CPU) [MOCK-DECIDER] | 0.0% | 2200 ms | 0.0% | 2200 ms | 0.0% | 2200 ms |
| PERSONA-BLIND CEILING | 0.0% | 2200 ms | 0.0% | 250 ms | 0.0% | 250 ms |
| ORACLE-UPPER-BOUND (Persona-Aware) | 10.2% | 250 ms | 0.0% | 250 ms | 0.0% | 250 ms |

## 3. Per-Category Performance Breakdown

| Approach | Complete Accuracy % | Complete Delay P50 | Incomplete Premature % | Ambiguous Accuracy % | Ambiguous Premature % |
|---|---|---|---|---|---|
| Fixed 600ms Baseline | 93.8% | 600 ms | 63.0% | 51.2% | 43.2% |
| Fixed 800ms Baseline | 75.3% | 800 ms | 42.0% | 52.5% | 23.5% |
| Fixed 1200ms Baseline | 43.2% | 1117 ms | 1.2% | 70.4% | 0.0% |
| Text Heuristic Baseline (250ms, dev-tuned) | 100.0% | 250 ms | 0.0% | 45.1% | 54.9% |
| Strands Decider 2B (0 ms decision latency) [MOCK-DECIDER] | 100.0% | 250 ms | 0.0% | 45.1% | 54.9% |
| Strands Decider 2B (236ms GPU, Tesla T4) [MOCK-DECIDER] | 100.0% | 486 ms | 0.0% | 51.2% | 48.8% |
| Strands Decider 2B (3190ms CPU) [MOCK-DECIDER] | 100.0% | 2200 ms | 0.0% | 100.0% | 0.0% |
| PERSONA-BLIND CEILING | 100.0% | 250 ms | 0.0% | 54.9% | 0.0% |
| ORACLE-UPPER-BOUND (Persona-Aware) | 100.0% | 250 ms | 0.0% | 74.7% | 6.8% |

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
- **Premature Interruption Rate**: Fraction of all test turns where the system yielded `RESPOND` while speaker was mid-thought. Numerator = premature interruption count, Denominator = 324 (total test turns).
- **Unnecessary Wait Rate**: Fraction of completed turns where agent waited > 800 ms before responding. Numerator = delayed turns, Denominator = 154 (total completed turns). Threshold is global 800 ms.
- **Response Delay (P50/P95)**: Elapsed time in milliseconds from speech end to agent floor yield, evaluated exclusively over completed turns (n=154).
- **Accuracy**: Binary turn-taking decision accuracy over all turns (n=324).
- **Uncertainty Intervals**: Headline tables report non-parametric two-stage cluster bootstrap 95% intervals (B=1000 resamples: sample users with replacement, then sample turns within user with replacement) to account for turn clustering across 9 users (36 turns/user). Secondary column reports standard Wilson 1927 score intervals for comparison.
- **Reference Ceilings**: `PERSONA-BLIND CEILING` uses text category and global majority continuation (WAIT), achieving 77.2% (250/324). `ORACLE-UPPER-BOUND (Persona-Aware)` uses archetype parameters to achieve 82.1% (266/324). Neither is an achievable baseline; both serve as theoretical bounds.

## 6. Findings & Observations
- **Ambiguity Impact**: On ambiguous utterances where complete-sounding clauses have persona-sampled continuations, the Text Heuristic achieves 45.1% accuracy with 54.9% premature interruptions, matching the mock decider; no real Decider run yet.
- **Persona Pause Dilemma**: Fixed 600ms causes 53.7% premature interruptions on slow pausers versus 0.0% on fast pausers.
- **CPU Latency Degeneration**: At 3190 ms CPU latency, speculative queries arrive after the 2200 ms max_wait timeout, degenerating into a fixed 2200 ms timeout with 100.0% unnecessary wait.
- **Cache & Replay Protocol**: Replay verified with 2663 cache hits and 0 cache misses (zero fallback).
