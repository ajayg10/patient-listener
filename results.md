# Benchmark Results — Patient Listener

> [!WARNING]
> **PROVENANCE & MOCK NOTICE**: All Decider rows below were generated using `MockDecider` cache entries (`backend=mock`, 325 keys, total compute time 0.00 ms). They are marked `[MOCK-DECIDER]` and must NOT be cited as measured performance of the real weights. A simple text heuristic baseline (`TextHeuristicBaseline`) achieves 100% accuracy on these synthetic templates, matching or exceeding the mock decider without neural inference.

> [!IMPORTANT]
> **SYNTHETIC BENCHMARK**: Evaluated on 324 synthetic test turns across 3 conflicting persona archetypes (fast, slow, filler-heavy). Zero leakage: dev and test splits strictly partition users and templates.

## 1. Test Split Turn-Taking Comparison (n=324 turns)

| Approach | Accuracy (Wilson 95% CI) | Premature Interruption (Wilson 95% CI) | Unnecessary Wait (Wilson 95% CI) | Delay P50 (Bootstrap 95% CI) | Delay P95 | Latency Condition |
|---|---|---|---|---|---|---|
| Fixed 600ms Baseline | 67.6% [62.3%, 72.5%] | 30.6% [25.8%, 35.8%] | 3.7% [1.7%, 7.8%] | 600 ms [600.0, 600.0] ms | 600 ms | Fixed |
| Fixed 800ms Baseline | 68.5% [63.3%, 73.3%] | 17.0% [13.3%, 21.4%] | 29.0% [22.6%, 36.4%] | 800 ms [800.0, 800.0] ms | 800 ms | Fixed |
| Fixed 1200ms Baseline | 73.5% [68.4%, 78.0%] | 0.0% [0.0%, 1.2%] | 100.0% [97.7%, 100.0%] | 1134 ms [1051.0, 1200.0] ms | 1200 ms | Fixed |
| Text Heuristic Baseline (250ms, dev-tuned) | 100.0% [98.8%, 100.0%] | 0.0% [0.0%, 1.2%] | 0.0% [0.0%, 2.3%] | 250 ms [250.0, 250.0] ms | 250 ms | Heuristic |
| Strands Decider 2B (0ms Oracle) [MOCK-DECIDER] | 95.4% [92.5%, 97.2%] | 4.6% [2.8%, 7.5%] | 0.0% [0.0%, 2.3%] | 250 ms [250.0, 250.0] ms | 250 ms | Evaluated |
| Strands Decider 2B (3190ms CPU) [MOCK-DECIDER] | 100.0% [98.8%, 100.0%] | 0.0% [0.0%, 1.2%] | 100.0% [97.7%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | Evaluated |
| Decider Sweep (100ms Sweep) [MOCK-DECIDER] | 95.7% [92.9%, 97.4%] | 4.3% [2.6%, 7.1%] | 0.0% [0.0%, 2.3%] | 350 ms [350.0, 350.0] ms | 350 ms | Sweep |
| Decider Sweep (300ms Sweep) [MOCK-DECIDER] | 96.9% [94.4%, 98.3%] | 3.1% [1.7%, 5.6%] | 0.0% [0.0%, 2.3%] | 550 ms [550.0, 550.0] ms | 550 ms | Sweep |
| Decider Sweep (1000ms Sweep) [MOCK-DECIDER] | 100.0% [98.8%, 100.0%] | 0.0% [0.0%, 1.2%] | 100.0% [97.7%, 100.0%] | 1250 ms [1250.0, 1250.0] ms | 1250 ms | Sweep |
| Decider Sweep (3000ms Sweep) [MOCK-DECIDER] | 100.0% [98.8%, 100.0%] | 0.0% [0.0%, 1.2%] | 100.0% [97.7%, 100.0%] | 2200 ms [2200.0, 2200.0] ms | 2200 ms | Sweep |
| Bedrock LLM Judge (Claude 3 Haiku) | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED (no AWS credentials) |
| Strands Decider 2B (GPU ~35ms) | *not measured yet* | *not measured yet* | *not measured yet* | *not measured yet* | *not measured yet* | Pending GPU Host |

## 2. Decider B0 40-Case Small Validation Set
- Number of cases: 40
- Fixed 800ms Baseline Accuracy: 40.0%
- Strands Decider 2B Accuracy: 82.5%

## 3. Findings & Observations
- **Heuristic Parity**: The simple regex-based `TextHeuristicBaseline` (tuned on dev) achieves 100.0% accuracy with 0.0% premature interruptions and 250 ms delay, outperforming neural models on these synthetic templates because template completions cleanly correlate with terminal punctuation and connectors.
- **CPU Latency Degeneration**: At 3190 ms CPU decision time, speculative queries arrive after the 2200 ms max_wait timeout. The system degenerates entirely into a fixed 2200 ms timeout with 100% unnecessary wait rate and 2200 ms response delay.
- **Persona Conflict Reality**: Fixed timeouts create an unavoidable tradeoff: 600ms causes 31.2% premature interruptions (mostly on slow pausers), while 1200ms eliminates premature interruptions but inflicts 100% unnecessary wait.
- **Provenance & Honesty**: All Decider rows in this table used MockDecider cache entries. They illustrate policy timing dynamics on a virtual clock, but do not represent validated neural inference on these templates.
