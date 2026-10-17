# Benchmark Results — Patient Listener

> [!WARNING]
> **PROVENANCE & MOCK NOTICE**: All Decider rows below were generated using `MockDecider` cache entries (`backend=mock`, 448 keys, compute time 0.00 ms). They are marked `[MOCK-DECIDER]` and must NOT be cited as measured performance of the real weights. On ambiguous turns, the text heuristic baseline achieves matching accuracy to the mock decider; no real Decider run yet.

> [!IMPORTANT]
> **SYNTHETIC BENCHMARK**: Evaluated on 324 synthetic test turns (157 complete, 167 incomplete) across 3 conflicting persona archetypes (fast, slow, filler-heavy). Zero leakage: dev and test splits strictly partition users and templates.

## 1. Test Split Turn-Taking Comparison (n=324 turns)

| Approach | Accuracy (Wilson 95% CI) | Premature Interruption (Wilson 95% CI) | Unnecessary Wait (Wilson 95% CI) | Delay P50 (Bootstrap 95% CI) | Delay P95 | Latency Condition |
|---|---|---|---|---|---|---|
| Fixed 600ms Baseline | 61.7% [56.3%, 66.8%] | 35.2% [30.2%, 40.5%] | 6.4% [3.5%, 11.3%] | 600 ms [600.0, 600.0] ms | 600 ms | Fixed |
| Fixed 800ms Baseline | 63.0% [57.6%, 68.0%] | 19.8% [15.8%, 24.4%] | 35.7% [28.6%, 43.4%] | 800 ms [800.0, 800.0] ms | 800 ms | Fixed |
| Fixed 1200ms Baseline | 69.8% [64.5%, 74.5%] | 0.0% [0.0%, 1.2%] | 100.0% [97.6%, 100.0%] | 1006 ms [890.0, 1121.0] ms | 1200 ms | Fixed |
| Text Heuristic Baseline (250ms, dev-tuned) | 81.8% [77.2%, 85.6%] | 18.2% [14.4%, 22.8%] | 0.0% [0.0%, 2.4%] | 250 ms [250.0, 250.0] ms | 250 ms | dev-tuned |
| Strands Decider 2B (0 ms decision latency) [MOCK-DECIDER] | 81.8% [77.2%, 85.6%] | 18.2% [14.4%, 22.8%] | 0.0% [0.0%, 2.4%] | 250 ms [250.0, 250.0] ms | 250 ms | mock cache, virtual clock |
| Strands Decider 2B (3190ms CPU) [MOCK-DECIDER] | 100.0% [98.8%, 100.0%] | 0.0% [0.0%, 1.2%] | 100.0% [97.6%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | mock cache, virtual clock |
| Decider Sweep (100ms Sweep) [MOCK-DECIDER] | 83.0% [78.6%, 86.7%] | 17.0% [13.3%, 21.4%] | 0.0% [0.0%, 2.4%] | 350 ms [350.0, 350.0] ms | 350 ms | mock cache, virtual clock |
| Decider Sweep (300ms Sweep) [MOCK-DECIDER] | 84.6% [80.2%, 88.1%] | 15.4% [11.9%, 19.8%] | 0.0% [0.0%, 2.4%] | 550 ms [550.0, 550.0] ms | 550 ms | mock cache, virtual clock |
| Decider Sweep (1000ms Sweep) [MOCK-DECIDER] | 100.0% [98.8%, 100.0%] | 0.0% [0.0%, 1.2%] | 100.0% [97.6%, 100.0%] | 1250 ms [1250.0, 1250.0] ms | 1250 ms | mock cache, virtual clock |
| Decider Sweep (3000ms Sweep) [MOCK-DECIDER] | 100.0% [98.8%, 100.0%] | 0.0% [0.0%, 1.2%] | 100.0% [97.6%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | mock cache, virtual clock |
| ORACLE-UPPER-BOUND | 91.7% [88.1%, 94.2%] | 2.2% [1.1%, 4.4%] | 12.7% [8.4%, 18.9%] | 250 ms [250.0, 250.0] ms | 2200 ms | Oracle bound |
| Bedrock LLM Judge | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED (no AWS credentials) |
| Strands Decider 2B (GPU) | not measured yet | not measured yet | not measured yet | not measured yet | not measured yet | not measured yet |

## 2. Per-Persona Performance Breakdown

| Approach | Fast Pauser Premature % | Fast Pauser Delay P50 | Slow Pauser Premature % | Slow Pauser Delay P50 | Filler-Heavy Premature % | Filler-Heavy Delay P50 |
|---|---|---|---|---|---|---|
| Fixed 600ms Baseline | 0.0% | 600 ms | 57.4% | 600 ms | 48.1% | 600 ms |
| Fixed 800ms Baseline | 0.0% | 693 ms | 35.2% | 800 ms | 24.1% | 800 ms |
| Fixed 1200ms Baseline | 0.0% | 693 ms | 0.0% | 1200 ms | 0.0% | 1110 ms |
| Text Heuristic Baseline (250ms, dev-tuned) | 6.5% | 250 ms | 24.1% | 250 ms | 24.1% | 250 ms |
| Strands Decider 2B (0 ms decision latency) [MOCK-DECIDER] | 6.5% | 250 ms | 24.1% | 250 ms | 24.1% | 250 ms |
| Strands Decider 2B (3190ms CPU) [MOCK-DECIDER] | 0.0% | 2200 ms | 0.0% | 2200 ms | 0.0% | 2200 ms |
| ORACLE-UPPER-BOUND | 6.5% | 250 ms | 0.0% | 250 ms | 0.0% | 250 ms |

## 3. Per-Category Performance Breakdown

| Approach | Complete Accuracy % | Complete Delay P50 | Incomplete Premature % | Ambiguous Accuracy % | Ambiguous Premature % |
|---|---|---|---|---|---|
| Fixed 600ms Baseline | 94.4% | 600 ms | 60.2% | 50.9% | 45.4% |
| Fixed 800ms Baseline | 71.3% | 800 ms | 35.2% | 52.8% | 24.1% |
| Fixed 1200ms Baseline | 43.5% | 1116 ms | 0.0% | 65.7% | 0.0% |
| Text Heuristic Baseline (250ms, dev-tuned) | 100.0% | 250 ms | 0.0% | 45.4% | 54.6% |
| Strands Decider 2B (0 ms decision latency) [MOCK-DECIDER] | 100.0% | 250 ms | 0.0% | 45.4% | 54.6% |
| Strands Decider 2B (3190ms CPU) [MOCK-DECIDER] | 100.0% | 2200 ms | 0.0% | 100.0% | 0.0% |
| ORACLE-UPPER-BOUND | 100.0% | 250 ms | 0.0% | 75.0% | 6.5% |

## 4. Decider B0 40-Case Small Validation Set
- Number of cases: 40
- Fixed 800ms Baseline Accuracy: 40.0% (16/40)
- Strands Decider 2B (Real Weights): 85.0% (34/40) [backend: local_cpu (AMD Ryzen 5 5500U, b0_spike.py)]
- Strands Decider 2B [MOCK-DECIDER]: 82.5% (33/40) [backend: mock (MockDecider)]

## 5. Metric Definitions & Protocol
- **Premature Interruption Rate**: Fraction of all test turns where the system yielded `RESPOND` while speaker was mid-thought. Numerator = premature interruption count, Denominator = 324 (total test turns).
- **Unnecessary Wait Rate**: Fraction of completed turns where agent waited > 800 ms before responding. Numerator = delayed turns, Denominator = 157 (total completed turns). Threshold is global 800 ms.
- **Response Delay (P50/P95)**: Elapsed time in milliseconds from speech end to agent floor yield, evaluated exclusively over completed turns (n=157).
- **Accuracy**: Binary turn-taking decision accuracy over all turns (n=324).

## 6. Findings & Observations
- **Ambiguity Impact**: On ambiguous utterances where complete-sounding clauses have persona-sampled continuations, the Text Heuristic achieves 45.4% accuracy with 54.6% premature interruptions, matching the mock decider; no real Decider run yet.
- **Persona Pause Dilemma**: Fixed 600ms causes 57.4% premature interruptions on slow pausers versus 0.0% on fast pausers.
- **CPU Latency Degeneration**: At 3190 ms CPU latency, speculative queries arrive after the 2200 ms max_wait timeout, degenerating into a fixed 2200 ms timeout with 100.0% unnecessary wait.
- **Cache & Replay Protocol**: Replay verified with 2281 cache hits and 0 cache misses (zero fallback).
