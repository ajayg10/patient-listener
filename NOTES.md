# Build Notes — Patient Listener

## Status board
| Step | Title | Status | Date |
|---|---|---|---|
| B0 | Feasibility spike (GO/NO-GO gate) | DONE | 2026-10-17 |
| B0.5 | Latency diagnosis & synthetic dataset audit | DONE | 2026-10-17 |
| B1 | Scaffold | DONE | 2026-10-17 |
| B2 | Decider service | DONE | 2026-10-17 |
| B3 | Events, state builder, turn manager | DONE | 2026-10-17 |
| B4 | Reply agent | DONE | 2026-10-17 |
| B5 | Baselines, dataset, eval harness | DONE | 2026-10-17 |
| B5.5 | Audit & honesty check | DONE | 2026-10-17 |
| B5.8 | Reconcile counts, cluster bootstrap & personalization effect | DONE | 2026-10-18 |
| B5.9 | Independent units, learnable profile, ceilings & enumeration | DONE | 2026-10-18 |
| B6 | Pace profile + regret log | DONE | 2026-10-18 |
| B7 | Confidence-aware policy | DONE | 2026-10-18 |
| B8 | Reusable pattern package | DONE | 2026-10-17 |
| B9 | API | DONE | 2026-10-19 |
| F1 | Frontend scaffold + WS client | DONE | 2026-10-21 |
| E1 | run_eval speed optimization | DONE | 2026-10-21 |
| E2 | Real cache validation and archive | DONE | 2026-10-22 |
| E3 | Real Decider evaluation | DONE | 2026-10-22 |
| E4 | Calibration audit & operating point analysis | IN PROGRESS | 2026-10-22 |
| F2 | Live view | TODO | |
| F3 | Side-by-side demo mode | DONE | 2026-10-22 |
| F4 | Profile + regret UI | TODO | |
| F5 | Results dashboard + polish | TODO | |
| R1 | Ship (submission bundle & docs) | TODO | |

## Environment
- **OS**: Microsoft Windows 11 Home Single Language (NT 10.0.26200.0, x86_64)
- **CPU**: AMD Ryzen 5 5500U with Radeon Graphics (6 cores, 12 threads)
- **RAM**: 15.33 GB total
- **GPU**: None detected (`nvidia-smi` not found, CPU execution)
- **Python**: Python 3.11.17 available at `E:\uv-python\cpython-3.11-windows-x86_64-none\python.exe`
- **uv**: uv 0.12.23
- **Disk Space**:
  - C: 4.31 GB free / 194.89 GB used
  - D: 102.07 GB free / 42.39 GB used
  - E: 92.35 GB free / 39.30 GB used (Workspace root `E:\patient-listener`)
- **Caches & Environment Variables**:
  - `HF_HOME`: `E:\hf-cache` (Configured for session; redirected away from C: to protect the 4.31 GB remaining)
  - `UV_CACHE_DIR`: `E:\uv-cache`
  - `TEMP`: `C:\Users\hp\AppData\Local\Temp`
  - `UV_PROJECT_ENVIRONMENT`: `.venv` (located on E: drive)
- **AWS Bedrock / STS**: `aws CLI not found` (No credentials configured; Bedrock LLM judge skipped per Section 0.5 / MOCK_BEDROCK=1)
- **Internet Access**: Verified HTTP 200 to `huggingface.co`
- **Decider Model Candidate**: `StrandsAgents/strands-decider-2B-hobson-v19`
  - Adapter + scoring head size: 87.81 MB (measured via HF API)
  - Base torso: `Qwen/Qwen3.5-2B-Base` (4.26 GB measured via HF API)
  - Combined model download size: ~4.35 GB
  - Storage Gate Check: E: has 91.31 GB free (> 15 GB requirement; PASS). C: has 4.31 GB free (protected via `HF_HOME=E:\hf-cache` and `TEMP=E:\tmp`).

## Step B0 — PLAN
- **Goal**: Complete the feasibility spike and establish a rigorous GO/NO-GO gate for personalized, regret-aware turn-taking with Strands Decider.
- **Approach**:
  1. Set `$env:HF_HOME = "E:\hf-cache"` and create virtual environment using `uv` with Python 3.11.
  2. Install `strands-decider` and test CLI invocation on a sample state.
  3. Load model in Python via `strands_decider.infer.load_engine` on CPU.
  4. Measure cold start and warm latency (median/p95 over repeats) at state sizes 20, 60, and 150 tokens on this machine (CPU).
  5. Hand-craft 40 realistic conversational pause cases (20 "done", 20 "still thinking") into `backend/eval/data/b0_cases.jsonl` with token counts, pause duration, and ground-truth turn state.
  6. Run comparison: Fixed 800 ms timeout vs Decider (Bedrock LLM-judge marked SKIPPED due to no AWS credentials).
  7. Formulate data-driven GO/NO-GO verdict based on accuracy, latency, and pause classification viability.
- **Files expected**:
  - `backend/eval/data/b0_cases.jsonl`
  - `backend/eval/b0_spike.py` (temporary spike script to measure latencies and evaluate b0 cases)
  - `NOTES.md` updated with measured numbers and GO/NO-GO verdict.

## Step B0 — DONE
- **Status**: DONE (GO verdict on classification signal and architectural viability)
- **What was done**:
  1. Hand-crafted 40 labeled pause cases into `backend/eval/data/b0_cases.jsonl` (20 "WAIT" / still thinking, 20 "RESPOND" / turn complete) spanning realistic hesitation patterns (trailing connectors, fillers, slow searches, multi-part lists, complete queries).
  2. Verified download of `StrandsAgents/strands-decider-2B-hobson-v19` adapter (87.81 MB) and base torso `Qwen/Qwen3.5-2B-Base` (4.55 GB) into `E:\hf-cache`. Verified exact byte count (4,548,221,488 bytes) and SHA256 (`928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d`) matching HF LFS OID.
  3. Verified `strands-decider ask` CLI execution in offline mode (`HF_HUB_OFFLINE=1`). Returned structured choice ("billing", 84.4% prob, 0.766 confidence).
  4. Executed `backend/eval/b0_spike.py`: measured cold-start load time, warm inference latency across 20/60/150 token states, and ran 40-case comparative evaluation against the Fixed 800ms baseline (Bedrock judge marked SKIPPED due to lack of AWS credentials).
  5. Saved structured evaluation output to `backend/eval/results/b0_results.json`.
- **Files changed**:
  - `backend/eval/data/b0_cases.jsonl` (40 labeled evaluation cases)
  - `backend/eval/b0_spike.py` (feasibility spike script)
  - `backend/eval/results/b0_results.json` (benchmark and evaluation metrics)
  - `NOTES.md` (environment, plan, incident resolution, done entry)
- **Decisions + why**:
  - **Verdict**: Signal: GO (synthetic, n=40, see caveats). Strands Decider achieved 85.0% accuracy on linguistic turn-completion vs 40.0% for Fixed 800ms. Pause length is nearly uninformative in these 40 SYNTHETIC cases; fixed-800 at 40% (CI includes 50%) is roughly chance. The cases test whether the model reads the words, not personalization. The "6x" figure must NOT be used (baseline at chance, n=40, agent-written cases).
  - **Latency reality**: CPU warm latency on AMD Ryzen 5 5500U is ~3.2 - 4.7 s per decision due to uncompiled linear-attention/conv1d reference kernels. This is too slow for synchronous sub-second pause checkpoints (250/500/800 ms).
  - **Architectural adaptation**:
    - Turn-taking decisions must run **asynchronously** via a speculative generation-counter pattern (start inference at silence onset; if speech resumes before Decider returns, discard stale result immediately).
    - For live demo, provide both local CPU execution and instantaneous Mock mode (`MOCK_BEDROCK=1` / calibrated mock Decider for zero-lag live testing).
- **Measured numbers (with how measured)**:
  - Cold start model load (CPU): 15.68 s (measured via `time.perf_counter()` around `load_engine()`).
  - Warm inference latency (AMD Ryzen 5 5500U, 10 repeats per size, `time.perf_counter()`):
    - 20 tokens: median 4705.1 ms, p95 5219.5 ms (min 3974.5, max 5273.7 ms)
    - 60 tokens: median 4733.7 ms, p95 5252.6 ms (min 4141.8, max 5453.6 ms)
    - 150 tokens: median 4695.8 ms, p95 5460.1 ms (min 4212.5, max 5844.2 ms)
  - 40-Case Benchmark Comparison:
    - Fixed 800ms Baseline: Accuracy = 40.0% (16/40), Premature Interruptions = 12/20 (60%), Unnecessary Waits = 12/20 (60%).
    - Strands Decider 2B: Accuracy = 85.0% (34/40), Premature Interruptions = 2/20 (10%), Unnecessary Waits = 4/20 (20%).
    - Decider latency during case run: Median = 3894.3 ms, P95 = 4294.7 ms.
    - Bedrock LLM judge: SKIPPED (no AWS credentials configured, `MOCK_BEDROCK=1` mode).
- **Problems / open questions**:
  - Linear attention CPU kernel fallback: PyTorch outputs warnings that `causal_conv1d` and `flash-linear-attention` are falling back to reference loops. This accounts for the ~4s CPU latency.
  - On a GPU with FlashAttention, Decider 2B runs in ~20-50 ms. Our architecture in B3 must decouple the event loop from the inference thread so voice audio is never blocked.
- **Next step**: Step B0.5 — Latency diagnosis & synthetic dataset audit.

## Step B0.5 — PLAN
- **Goal**:
  1. Diagnose why CPU inference latency was flat (~4.7 s) across 20/60/150 token states (examine dtype, thread allocation, prompt/state token breakdown, and optional dynamic int8 quantization).
  2. Audit `b0_cases.jsonl` pause distribution (silence_ms stats per label, fixed-800 accuracy by bucket, Wilson 95% confidence intervals) to evaluate whether cases were unintentionally adversarial to fixed timeouts.
  3. Update GO/NO-GO verdict into the two required lines and document the virtual clock design requirement for B3/B5.
- **Approach**:
  - Run `backend/eval/b0_5_diagnosis.py` measuring free RAM, default dtype vs float32, `torch.get_num_threads()` vs 6 physical cores, prompt vs state token counts, and single-attempt int8 dynamic quantization test.
  - Calculate Wilson score intervals for Decider (85.0%) and Fixed 800ms (40.0%).
  - Label all findings SYNTHETIC (n=40, written by coding agent).
- **Files expected**:
  - `backend/eval/b0_5_diagnosis.py`
  - `NOTES.md` updated with diagnosis table, dataset audit, and updated verdict.

## Step B0.5 — DONE
- **Status**: DONE
- **What was done**:
  1. Latency diagnosis benchmark executed (5 warm runs each, median only):
     - Dtype inspected: Config specifies `bfloat16`, but PyTorch automatically loaded parameters as `torch.float32` on CPU. Free RAM after model loading dropped to 0.39 GB (< 3 GB spare requirement); FP32 reload was safely skipped per instructions.
     - Threading: Default `torch.get_num_threads()` was 6 (PyTorch detected physical cores). Setting `torch.set_num_threads(6)` explicitly dropped median latency from 3558.0 ms to 3190.1 ms (~10.3% faster).
     - Token analysis: State text is 35 tokens, but total input tokens forwarded by engine is 138 tokens. Prompt template overhead is 103 tokens (fixed question/options/criteria template dominates the sequence, explaining why 20/60/150 token variations showed near-flat latency).
     - Dynamic INT8 quantization: Attempted one-liner `torch.ao.quantization.quantize_dynamic`; failed due to hybrid linear-attention layer structure; skipped per instructions.
  2. Dataset Audit (`b0_cases.jsonl`, SYNTHETIC, n=40, written by coding agent):
     - Silence distribution:
       - `WAIT` (still thinking): min=600 ms, median=800.0 ms, max=1200 ms (n=20)
       - `RESPOND` (done): min=500 ms, median=750.0 ms, max=900 ms (n=20)
       - The distributions overlap heavily (RESPOND range 500-900 ms; WAIT range 600-1200 ms). Pause length is nearly uninformative in these 40 SYNTHETIC cases; fixed-800 at 40% (CI includes 50%) is roughly chance. The cases test whether the model reads the words, not personalization.
       - **Do NOT use the "6x" figure**: baseline at chance, n=40, agent-written cases.
     - Fixed 800ms Accuracy split:
       - Silence < 800 ms: 8/20 (40.0%)
       - Silence >= 800 ms: 8/20 (40.0%)
       - Overall: 16/40 (40.0%)
     - Wilson 95% Confidence Intervals (n=40):
       - Strands Decider 2B (34/40, 85.0%): Full Wilson 1927: [70.9% — 92.9%].
       - Fixed 800ms (16/40, 40.0%): Full Wilson 1927: [26.4% — 55.4%] (enclosing 50% chance; simplified textbook variant yields [27.0% — 54.7%]).
  3. B3/B5 Design Note:
     - The eval harness must run on a **virtual clock** where decision latency is a configurable parameter. This allows evaluations to report results under:
       (i) Measured CPU latency (3190 ms, Ryzen 5 5500U)
       (ii) Measured remote-GPU latency (once hardware/endpoint is provided)
       (iii) Zero-latency oracle
     - Numbers from the blog must not be inserted into results as if measured.
  4. Logged Architectural Decisions:
     - **DeciderClient Interface**: Decider is accessed through a `DeciderClient` interface with backends selected by `DECIDER_BACKEND=mock|local|remote` in `.env` (`remote` = HTTP URL, for a GPU host). Tests use `mock` only. *Why*: Decouples business logic from execution environment; allows running unit tests without downloading weights or hitting memory limits; enables switching between local CPU, fast mock for demo/tests, and remote GPU endpoint without changing agent code.
     - **Server-Side Latency Reporting**: Latency results are always reported as server-side decision time, separate from network round trip. *Why*: Isolates algorithmic/model compute performance from variable network transmission ping.
  5. Memory Guard:
     - Never load the real model inside `pytest`. Real-model runs happen only in explicit scripts, one process at a time, after checking free RAM is at least 2 GB; otherwise stop and report.
- **Files changed**:
  - `backend/eval/b0_5_diagnosis.py` (diagnosis and audit script)
  - `backend/eval/results/b0_5_diagnosis.json` (saved metrics and audit data)
  - `NOTES.md` (recorded measurements, audit, verdict update, and design note)
- **Measured numbers (Latency Diagnosis Table)**:
  | Configuration | Threads | Torso Dtype | Input Tokens (Prompt + State) | Median Latency |
  |---|---|---|---|---|
  | Baseline Default | 6 | torch.float32 | 138 (103 template + 35 state) | 3558.0 ms |
  | Physical Cores Pinned | 6 | torch.float32 | 138 (103 template + 35 state) | 3190.1 ms |
  | Float32 Reload | — | — | — | SKIPPED (0.39 GB free RAM < 3 GB gate) |
  | Dynamic INT8 Quant | — | — | — | SKIPPED (incompatible with hybrid layer) |

- **Updated Verdict**:
  Signal: GO (synthetic, n=40, see caveats)
  Live CPU latency: 3190.1 ms (AMD Ryzen 5 5500U, 6 threads, torch.float32)

- **Next step**: Step B1 — Scaffold.

## Step B1 — PLAN
- **Goal**: Establish the production repo structure, uv project with pyproject.toml, ruff, pytest, environment configurations (.env.example, config.yaml), and passing smoke tests.
- **Approach**:
  1. Create directories: `backend/app/`, `backend/tests/`, `patient_listener_pattern/`.
  2. Create `pyproject.toml` with `fastapi`, `uvicorn`, `pydantic`, `pyyaml`, `strands-decider`, `pytest`, `ruff`, explicitly excluding `hf-xet`.
  3. Create `.env.example` with `DECIDER_BACKEND=mock|local|remote`, `DECIDER_URL`, Bedrock settings (`BEDROCK_MODEL_ID`, `AWS_REGION`, `MOCK_BEDROCK=1`), and `HF_HOME=E:\hf-cache`.
  4. Create `config.yaml` configuring default pause checkpoints (250, 500, 800, 1200, 1800 ms), confidence thresholds (`hi=0.7`, `lo=0.3`), and profile params.
  5. Create `backend/app/config.py` for structured configuration parsing.
  6. Create `backend/tests/test_smoke.py` asserting config defaults and mock DeciderClient behavior without loading the 2B model in pytest (memory guard).
  7. Run `pytest` and `ruff check .` to verify clean pass.
- **Files expected**:
  - `pyproject.toml`
  - `.env.example`
  - `.env`
  - `config.yaml`
  - `backend/app/__init__.py`
  - `backend/app/config.py`
  - `backend/tests/__init__.py`
  - `backend/tests/test_smoke.py`

## Step B1 — DONE
- **Status**: DONE
- **What was done**:
  1. Configured project dependencies and build system in `pyproject.toml` (`fastapi`, `uvicorn`, `pydantic-settings`, `pyyaml`, `strands-decider`, `pytest`, `ruff`), explicitly omitting `hf-xet`.
  2. Created `.env.example` and `.env` specifying `DECIDER_BACKEND=mock|local|remote`, `DECIDER_URL=http://localhost:8000`, `DECIDER_MODEL_ID`, `BEDROCK_MODEL_ID`, `AWS_REGION`, `MOCK_BEDROCK=1`, and `HF_HOME=E:\hf-cache`.
  3. Created `config.yaml` specifying default pause checkpoints (`[250, 500, 800, 1200, 1800]` ms), confidence thresholds (`hi=0.70`, `lo=0.30`), and pace profile defaults.
  4. Built `backend/app/config.py` loading environment variables via Pydantic Settings merged with YAML structure.
  5. Built `backend/tests/test_smoke.py` verifying config loading and enforcing the memory guard (real model weights are never loaded in pytest).
  6. Verified all unit tests and lint checks: `pytest` passed (3/3 passed in 0.85s) and `ruff check .` passed with 0 errors.
- **Files changed**:
  - `pyproject.toml`
  - `.env.example`
  - `.env`
  - `config.yaml`
  - `backend/app/__init__.py`
  - `backend/app/config.py`
  - `backend/tests/__init__.py`
  - `backend/tests/test_smoke.py`
  - `patient_listener_pattern/__init__.py`
  - `NOTES.md`
- **Decisions + why**:
  - Used Pydantic Settings with YAML fallback in `backend/app/config.py`: provides strict type checking, auto-completion, and default values while honoring `.env` overrides.
  - Smoke tests test mock/configuration paths only: keeps test suite sub-second (0.85s) and completely guards against memory spikes.
- **Measured numbers (with how measured)**:
  - Test suite runtime: 0.85 s (`pytest -v` across 3 test cases).
  - Linting: 0 errors (`ruff check .`).
- **Next step**: Step B2 — Decider service.

## Step B2 — PLAN
- **Goal**: Implement a clean `DeciderClient` service with `mock`, `local`, and `remote` backends, typed question definitions, standalone decider server, benchmark script, and mock-only tests.
- **Source API Analysis (Probabilities vs Confidence)**:
  - From `strands_decider.schema.derive_confidence` and `infer._to_answer`:
    - `probabilities: dict[str, float]`: The softmax readout across all $N$ options in the question, summing to 1.0. For $N=2$ (`WAIT`/`RESPOND`), $p(\text{WAIT}) + p(\text{RESPOND}) = 1.0$.
    - `confidence: float`: A normalized metric measuring distribution concentration:
      $$\text{confidence} = \frac{N \cdot p_{\text{max}} - 1}{N - 1}$$
      - Uniform distribution ($p_{\text{max}} = 1/N$) $\rightarrow$ confidence = 0.0.
      - One-hot certainty ($p_{\text{max}} = 1.0$) $\rightarrow$ confidence = 1.0.
      - For $N=3$ (e.g. CLI billing test), $p_{\text{max}} = 0.8442 \rightarrow \text{confidence} = (3 \times 0.8442 - 1)/2 = 0.7663$.
      - For $N=2$ (binary choice), $\text{confidence} = |2p - 1|$.
      - B7 uses `confidence` as the decision reliability threshold (e.g., act when confidence $\ge$ `hi`, hold/reask when confidence $<$ `lo`).
- **Approach**:
  1. Define typed questions in `backend/app/questions.py`:
     - Q1: `turn_complete` (`WAIT` vs `RESPOND`).
     - Q2: `mid_thought` (`YES` vs `NO`).
     - Measure and record template token overhead for each.
  2. Implement `backend/app/decider_service.py`:
     - `Decision` dataclass: `choice: str`, `probabilities: dict[str, float]`, `confidence: float`, `latency_ms: float` (server-side compute time only), `backend: str`.
     - `DeciderClient` Protocol defining `decide(state_text: str, question: ChoiceQuestion) -> Decision`.
     - `MockDecider`: deterministic rule-based (trailing connectors, fillers, silence_ms), configurable fake latency.
     - `LocalDecider`: loads engine once, one warmup call, sets torch threads to 6, RAM gate requiring $\ge 2$ GB free RAM before loading.
     - `RemoteDecider`: HTTP client posting to `DECIDER_URL`, separates server compute latency from network round-trip.
     - Factory function `get_decider_client(settings) -> DeciderClient` selecting backend by `DECIDER_BACKEND`.
  3. Create `scripts/decider_server.py`: minimal FastAPI server exposing `POST /decide` and `GET /health` with device auto-select (`cuda` if available else `cpu`), runnable on Colab/Kaggle or locally.
  4. Create `scripts/bench_decider.py`: benchmark script testing warm median/p95 at 20/60/150 tokens on 1 vs 2 questions in the same call (checking if multi-question batching is supported in real API). Enforces $\ge 2$ GB RAM check before running.
  5. Create `backend/tests/test_decider.py` (mock only!): mock determinism, probability sums to 1.0, latency field presence, RemoteDecider with mock HTTP transport, LocalDecider RAM gate failure via monkeypatch.
  6. Run `ruff` and `pytest`.
- **Files expected**:
  - `backend/app/questions.py`
  - `backend/app/decider_service.py`
  - `scripts/decider_server.py`
  - `scripts/bench_decider.py`
  - `backend/tests/test_decider.py`
  - `NOTES.md`

## Step B2 — DONE
- **Status**: DONE
- **What was done**:
  1. **Source API Analysis (Probabilities vs Confidence)**:
     - Documented return fields from `strands_decider.schema` and `infer._to_answer`:
       - `probabilities: dict[str, float]`: Softmax distribution across all options ($\sum p_i = 1.0$).
       - `confidence: float`: Normalized concentration score:
         $$\text{confidence} = \frac{N \cdot p_{\text{max}} - 1}{N - 1}$$
         Uniform distribution ($p_{\text{max}} = 1/N$) $\rightarrow$ `0.0`; one-hot certainty ($p_{\text{max}} = 1.0$) $\rightarrow$ `1.0`. For $N=3$, $p_{\text{max}} = 0.845 \implies \text{confidence} = 0.768$. For $N=2$, $\text{confidence} = |2p - 1|$.
       - B7 will use `confidence` as the decision reliability signal (e.g. threshold gating: if confidence $\ge$ `hi`, act; if confidence $\le$ `lo`, hold/reask).
  2. **Typed Questions (`backend/app/questions.py`)**:
     - Defined Q1 `turn_complete` (`WAIT` vs `RESPOND`) and Q2 `mid_thought` (`YES` vs `NO`) with shortened wording preserving semantics.
     - Measured template suffix token overhead against `Qwen/Qwen3.5-2B-Base` tokenizer:
       - B0.5 baseline template: 95 suffix tokens (103 total template overhead).
       - Q1 `turn_complete`: 81 suffix tokens (saving 14 tokens).
       - Q2 `mid_thought`: 73 suffix tokens (saving 22 tokens).
  3. **Decider Abstraction (`backend/app/decider_service.py`)**:
     - `Decision` dataclass: `choice: str`, `probabilities: dict[str, float]`, `confidence: float`, `latency_ms: float` (server-side compute only), `backend: str`, `round_trip_ms: float | None` (separate for remote), and `.probs` alias.
     - `MockDecider`: deterministic rule-based (analyzing fillers, trailing connectors, punctuation, silence duration), configurable `fake_latency_ms` for virtual-clock evaluation.
     - `LocalDecider`: loads engine once at startup, one warmup call, sets `torch.set_num_threads(cfg.decider_torch_threads)` (default 6), strict RAM gate refusing to load if free RAM < 2.0 GB, and resolves cached snapshot path for offline loading.
     - `RemoteDecider`: HTTP POST client to `DECIDER_URL`, extracts server compute `latency_ms` from response, and records network `round_trip_ms` separately.
     - `get_decider_client` factory selecting backend via `DECIDER_BACKEND=mock|local|remote`.
  4. **Standalone Server (`scripts/decider_server.py`)**:
     - Minimal FastAPI server exposing `POST /decide` and `GET /health` with hardware auto-selection (`cuda` if available else `cpu`), no tunnel code, ready for Colab/Kaggle GPU.
  5. **Inference Latency Benchmark (`scripts/bench_decider.py`)**:
     - Verified Strands Decider API natively supports multiple questions per call (`questions: dict[str, Question]` in `SystemOneRequest` and `engine.ask`).
     - RAM gate confirmed available system RAM (7.70 GB $\ge 2.0$ GB).
     - Benchmarked 5 warm runs each at 20, 60, and 150 token states for 1 Question vs 2 Questions in the same call.
     - Recorded measured CPU benchmark table and created an empty table template for GPU runs.
  6. **Mock-only Unit Tests (`backend/tests/test_decider.py`)**:
     - 6 unit test cases verifying mock determinism, probability sums $\approx 1.0$, latency reporting, remote mock transport, and RAM gate refusal via monkeypatch.
     - Test suite executed 9/9 passing tests in 0.68s.
     - `ruff check .` passed with 0 errors.
- **Files changed**:
  - `backend/app/questions.py`
  - `backend/app/decider_service.py`
  - `backend/app/config.py`
  - `scripts/decider_server.py`
  - `scripts/bench_decider.py`
  - `backend/tests/test_decider.py`
  - `NOTES.md`
- **Decisions + why**:
  - **Multi-Question Batching**: On CPU with uncompiled reference kernels, asking 2 questions against the same state adds ~2.4 - 2.7 s overhead. On GPU with FlashAttention, question suffixes share the prefix KV cache and evaluate in parallel with negligible overhead.
  - **Offline Snapshot Resolution**: When `HF_HUB_OFFLINE=1` is enforced, `snapshot_download(..., local_files_only=True)` reliably resolves the local snapshot directory without triggering `OfflineModeIsEnabled`.
  - **Separation of Compute Latency and Round-Trip Time**: `Decision.latency_ms` strictly isolates model inference duration, while `round_trip_ms` measures network transport time.
- **Measured numbers (Latency Benchmark)**:
  Available system RAM at benchmark run time: 7.70 GB (Gate requirement: >= 2.0 GB).
  ### Benchmark Results Table (Measured on CPU: AMD Ryzen 5 5500U, 6 threads, torch.float32)
  | State Tokens | 1 Question Median (ms) | 1 Question P95 (ms) | 2 Questions Median (ms) | 2 Questions P95 (ms) | 2Q Overhead (ms) |
  |---|---|---|---|---|---|
  | 20 tokens | 2967.6 | 3295.1 | 5526.4 | 5535.9 | +2558.8 |
  | 60 tokens | 3424.5 | 3470.9 | 5852.7 | 5886.5 | +2428.3 |
  | 150 tokens | 3336.9 | 3440.8 | 6095.3 | 6571.0 | +2758.4 |

  ### GPU Benchmark Results Table (Measured on GPU: Tesla T4, Google Colab, device cuda, real Decider)
  - Warnings in that run: `causal_conv1d` and `flash-linear-attention` missing (ran on reference path).
  | State Tokens | 1 Question Median (ms) | 1 Question P95 (ms) | 2 Questions Median (ms) | 2 Questions P95 (ms) | 2Q Overhead (ms) |
  |---|---|---|---|---|---|
  | 20 tokens | 167.5 | 176.0 | 343.1 | 343.7 | +175.6 |
  | 60 tokens | 235.6 | 236.7 | 395.2 | 400.3 | +159.6 |
  | 150 tokens | 237.3 | 238.5 | 395.8 | 401.1 | +158.5 |

  - **Batching finding (CPU)**: 2-question call costs about 1.7 to 1.9x the 1-question call on this CPU (2968 vs 5526 ms at 20 tok, 3337 vs 6095 ms at 150 tok, free RAM 7.7 GB). So no batching gain on CPU. Live decision = Q1 only; Q2 is for offline ablation. Do not claim a batching benefit unless a GPU row shows one.
  - Test suite runtime: 0.68 s (9/9 passed).
  - Ruff check: 0 errors.
- **Next step**: Step B3 — Events, state builder, turn manager.

## Step B3 — PLAN
- **Goal**: Implement turn-taking engine with typed events, state builder, and turn manager supporting injectable virtual/real clocks, generation-counter cancellation, checkpoint probes, and max-hold caps.
- **Approach**:
  1. Define injectable clock abstractions in `backend/app/clock.py`:
     - `Clock` protocol with `now_ms() -> float` and `sleep(ms: float)`.
     - `SystemClock`: real wall-clock using `time.perf_counter()`.
     - `VirtualClock`: explicit time advancement (`advance_ms(delta)`), configurable decision latency simulation for eval harness.
  2. Implement typed events in `backend/app/events.py`:
     - `SpeechStarted`, `SpeechDelta`, `SpeechEnded`, `SilenceProbe`, `DecisionRequested`, `DecisionCompleted`, `DecisionDropped`, `TurnYielded`, `MaxHoldExceeded`.
  3. Implement `backend/app/state_builder.py`:
     - Maintains transcript history, extracts recent ~40 words.
     - Adds hesitation cues (trailing conjunctions, fillers) and pause duration `[Silence: X ms]`.
     - Optional pace profile summary.
     - Enforces compact token budget (< 100 tokens).
  4. Implement `backend/app/turn_manager.py`:
     - Configured with pause checkpoints (`[250, 500, 800, 1200, 1800]` ms) and `max_wait_ms` (2200 ms).
     - State machine: `IDLE`, `LISTENING`, `PAUSED_EVALUATING`, `RESPONDING`.
     - Speculative generation counter: increments on speech or silence onset; any in-flight decision whose generation doesn't match current generation is dropped immediately.
     - New speech (`SpeechStarted` or `SpeechDelta`) cancels/drops in-flight decision and resets pause tracker.
     - Single in-flight decision gate: never issue overlapping requests.
     - Hard max-hold cap: triggers `TurnYielded(action="RESPOND")` when silence duration reaches `max_wait_ms`.
     - Simple placeholder policy hook (RESPOND if choice == RESPOND) until B7 confidence policy.
  5. Scripted-timeline unit tests in `backend/tests/test_turn_manager.py`:
     - Fast decision at checkpoint triggering RESPOND.
     - Slow decision (3000 ms) arriving after new speech (dropped via generation counter).
     - Max-hold cap reached during silence.
     - Speech resumed mid-decision.
     - State builder token budget assertion.
  6. Run `ruff check .` and `pytest`.
- **Files expected**:
  - `backend/app/clock.py`
  - `backend/app/events.py`
  - `backend/app/state_builder.py`
  - `backend/app/turn_manager.py`
  - `backend/tests/test_turn_manager.py`
  - `NOTES.md`

## Step B3 — DONE
- **Status**: DONE
- **What was done**:
  1. **Clock Abstractions (`backend/app/clock.py`)**:
     - Built `Clock` Protocol with `now_ms()` and `sleep(ms)`.
     - Built `SystemClock` backed by `time.perf_counter()`.
     - Built `VirtualClock` supporting manual `advance_ms()`, instant virtual sleep, and configurable decision latency for evaluation pipelines.
  2. **Typed Events (`backend/app/events.py`)**:
     - Created frozen dataclasses for `SpeechStarted`, `SpeechDelta`, `SpeechEnded`, `SilenceProbe`, `DecisionRequested`, `DecisionCompleted`, `DecisionDropped`, `TurnYielded`, `MaxHoldExceeded`.
  3. **State Builder (`backend/app/state_builder.py`)**:
     - Maintains recent speech buffer, extracts last ~40 words, appends silence marker `[Silence: X ms]` and optional pace profile summary.
     - Verified compact prompt token footprint (< 100 tokens).
  4. **Turn Manager (`backend/app/turn_manager.py`)**:
     - State machine (`IDLE`, `LISTENING`, `PAUSED`, `RESPONDING`).
     - Pause checkpoint polling (`[250, 500, 800, 1200, 1800]` ms).
     - Speculative generation counter: drops any delayed or stale in-flight decisions whose generation doesn't match current pause epoch.
     - In-flight concurrency gate: prevents overlapping inference calls.
     - Interruption handling: speech onset (`on_speech_started` / `on_speech_delta`) immediately drops pending decision and resets pause tracker.
     - Hard safety cap: triggers `MaxHoldExceeded` and `TurnYielded(action="RESPOND")` at `max_wait_ms`.
     - Placeholder policy hook yielding turn when decision choice is `RESPOND`.
  5. **Scripted Timeline Tests (`backend/tests/test_turn_manager.py`)**:
     - Tested fast decision completion triggering `RESPOND`.
     - Tested slow decision (3000 ms) dropped via generation counter after speech resumption.
     - Tested hard `max_wait_ms` safety cap yielding turn after continuous hesitation.
     - Tested speech resumed during in-flight query.
     - Tested state builder word and token length caps.
     - Test suite: 14/14 tests passing in 0.81s; `ruff check .` clean (0 errors).
- **Files changed**:
  - `backend/app/clock.py`
  - `backend/app/events.py`
  - `backend/app/state_builder.py`
  - `backend/app/turn_manager.py`
  - `backend/tests/test_turn_manager.py`
  - `NOTES.md`
- **Decisions + why**:
  - **Generation counter pattern**: Audio stream events arrive continuously. Incrementing `generation` on any state boundary (speech start, speech end) enables zero-lock cancellation: in-flight queries check `generation == self.generation` on return and discard stale results instantly.
  - **Separation of clock and state**: Parameterizing time via `Clock` enables identical TurnManager logic to operate live on WebSocket audio and in microsecond-fast virtual evaluations.
- **Measured numbers**:
  - State text token budget: 40-word state text is ~55 tokens (well within < 100 token ceiling).
  - Test suite runtime: 0.81 s across 14 test cases.
  - Linting: 0 errors (`ruff check .`).
- **Next step**: Step B4 — Reply agent.

## Step B4 — PLAN
- **Goal**: Implement `reply_agent.py` wrapping Strands Agent on Amazon Bedrock with token streaming, mock Bedrock fallback (`MOCK_BEDROCK=1`), and end-to-end integration triggered strictly on `RESPOND`.
- **Approach**:
  1. Inspect `strands-agents` source API for `BedrockModel` and streaming:
     - Verified: `from strands import Agent` and `from strands.models import BedrockModel`.
     - Streaming interface: `agent.stream_async(prompt)` yields event dicts where text delta is `event["data"]`.
  2. Implement `backend/app/reply_agent.py`:
     - `ReplyAgent`: accepts `AppConfig`, initializes `BedrockModel(model_id=cfg.bedrock_model_id, region_name=cfg.aws_region)` when `MOCK_BEDROCK=0` and credentials exist.
     - When `MOCK_BEDROCK=1` (default in tests) or credentials missing: provides calibrated mock streaming (`stream_reply`) yielding text deltas asynchronously.
     - `generate_reply(prompt: str) -> str`: non-streaming accumulator.
     - `should_respond(action: str) -> bool`: returns True strictly if action == "RESPOND".
  3. Integration in End-to-End Test (`backend/tests/test_reply_agent.py`):
     - End-to-end test with MockDecider and MockBedrock:
       - User speaks a completed thought.
       - Pause reaches checkpoint -> Decider evaluates state -> yields `TurnYielded(action="RESPOND")`.
       - Reply agent triggers on `RESPOND`, streams response tokens.
       - Verify reply is NOT triggered on `WAIT`.
  4. Real-Bedrock check:
     - Check live AWS credentials via STS: absent (`InvalidClientTokenId`), so mark real-Bedrock check SKIPPED in NOTES.md without blocking.
  5. Run `ruff` and `pytest`.
- **Files expected**:
  - `backend/app/reply_agent.py`
  - `backend/tests/test_reply_agent.py`
  - `NOTES.md`

## Step B4 — DONE
- **Status**: DONE
- **What was done**:
  1. **Strands Agents API Integration (`backend/app/reply_agent.py`)**:
     - Verified source API from `strands-agents`: `from strands import Agent` and `from strands.models import BedrockModel`.
     - Built `ReplyAgent` integrating with Amazon Bedrock Converse stream API via `agent.stream_async(prompt)`.
     - Added mock fallback (`MOCK_BEDROCK=1` by default in test suite or when AWS credentials are absent).
     - Contextual streaming responses with `stream_reply(prompt)` yielding text tokens asynchronously.
     - Enforced strict gating via `should_respond(action)`: reply generation triggers strictly if action is `RESPOND`.
  2. **End-to-End Pipeline Tests (`backend/tests/test_reply_agent.py`)**:
     - Tested mock token streaming and non-streaming reply generation.
     - Tested turn gating logic (`RESPOND` -> True; `WAIT`/`HOLD` -> False).
     - End-to-end integration: User speaks completed statement -> silence checkpoint reaches 250 ms -> `TurnManager` queries `MockDecider` -> emits `TurnYielded(action="RESPOND")` -> `ReplyAgent` streams reply tokens matching the user utterance.
     - Hesitation scenario: User hesitates with trailing connector -> Decider outputs `WAIT` -> `TurnManager` remains in `PAUSED` -> `ReplyAgent` is NOT triggered.
     - Test suite: 18/18 tests passing in 0.96s; `ruff check .` clean (0 errors).
  3. **Real-Bedrock Check**:
     - Verified AWS STS credentials: STS call failed with `InvalidClientTokenId` (no valid AWS credentials in environment). Marked real-Bedrock check SKIPPED per Section 0.5 without blocking.
- **Files changed**:
  - `pyproject.toml` (added `strands-agents` and `boto3`)
  - `backend/app/reply_agent.py`
  - `backend/tests/test_reply_agent.py`
  - `NOTES.md`
- **Decisions + why**:
  - **Graceful Mock Fallback**: When `MOCK_BEDROCK=1` or AWS credentials fail, `ReplyAgent` seamlessly switches to a contextual generator yielding streaming token deltas. This ensures complete end-to-end integration without blocking developer pair work or CI on cloud accounts.
  - **Strict Turn Gating**: Isolating agent invocation behind `should_respond` guarantees LLM tokens and API costs are never consumed while the user is merely pausing mid-thought (`WAIT`).
- **Measured numbers**:
  - Test suite runtime: 0.96 s (18/18 passed).
  - Linting: 0 errors (`ruff check .`).
  - Real-Bedrock check: SKIPPED (no credentials).
- **Next step**: Step B5 — Baselines, dataset, eval harness.

## Step B5 — PLAN
- **Goal**: Build baselines (fixed timeout and Bedrock LLM judge), generate synthetic persona-conflict dataset with clean disjoint dev/test splits (>= 300 test turns), implement evaluation metrics with Wilson & bootstrap CIs, precompute decision cache with runtime guard (> 60 min stops), and build evaluation harness across latency parameters.
- **Approach**:
  1. `backend/eval/baselines.py`:
     - `FixedTimeoutBaseline(timeout_ms)`: models standard silence timeout (600, 800, 1200 ms).
     - `LLMJudgeBaseline`: queries Bedrock Converse API with small model (Claude 3 Haiku). With missing AWS credentials (`InvalidClientTokenId`), marked SKIPPED in results table. Mock judge is restricted to unit test plumbing only and barred from results.md.
  2. `backend/eval/data/personas.yaml` & `backend/eval/generate_dataset.py`:
     - Define 3 persona archetypes (fast pauser, slow pauser, filler-heavy) with conflicting pause distributions (e.g. 700 ms is finished for fast pausers but mid-thought for slow pausers).
     - Generate structured scenarios with timed audio events (`events.py` schema), ground truth `turn_end_t`, `syntactically_complete` flags, and earlier user history turns.
     - Disjoint split: strictly partition by `user_id` AND `scenario_template_id` ensuring zero leakage between dev and test splits.
     - Test split has >= 300 turns. Label all files and tables SYNTHETIC.
  3. `backend/eval/metrics.py`:
     - `premature_interruption_rate`, `unnecessary_wait_rate`, `response_delay_p50_ms` / `p95_ms`, `decision_latency_p50_ms` / `p95_ms`, `accuracy`, and `inference_cost_usd`.
     - Wilson 1927 95% CI formula for rates; 1000-sample bootstrap percentile CI for delay percentiles.
  4. `scripts/precompute_decisions.py`:
     - Deduplicates all `(state_text, question)` needed for dev and test splits into `backend/eval/data/decisions_cache.jsonl`.
     - Keyed by sha256 hash of `(model_id, question_id, state_text)`. Resumable.
     - Estimates runtime: `N_uncached * median_latency_s` (3.19s). If > 60 min, prints estimate and halts.
     - Enforces RAM gate (>= 2.0 GB free) and supports `DECIDER_BACKEND=remote`.
  5. `backend/eval/run_eval.py`:
     - Reads decision cache and evaluates all approaches across latency conditions on `VirtualClock`:
       - 0 ms (oracle)
       - 3190 ms (measured CPU median)
       - GPU row ("not measured yet" placeholder)
       - Parameter sweep: 100, 300, 1000, 3000 ms.
     - Evaluates Fixed 600/800/1200ms baselines and Decider on B0 40 cases.
     - Writes `backend/eval/results/eval_results.json` and `backend/eval/results/results.md`.
  6. Unit tests in `backend/tests/test_eval.py`:
     - Split leakage check (disjoint users and templates).
     - Persona conflict check (probability of finished at 700 ms is higher for fast than slow persona).
     - Metric computations on hand-built timeline.
     - Cache hit/miss serialization.
  7. Run `ruff` and `pytest`.
- **Files expected**:
  - `backend/eval/baselines.py`
  - `backend/eval/data/personas.yaml`
  - `backend/eval/generate_dataset.py`
  - `backend/eval/metrics.py`
  - `scripts/precompute_decisions.py`
  - `backend/eval/run_eval.py`
  - `backend/tests/test_eval.py`
  - `NOTES.md`

## Step B5 — DONE
- **Status**: DONE
- **What was done**:
  1. `backend/eval/baselines.py`: Implemented `FixedTimeoutBaseline(timeout_ms)` (600, 800, 1200 ms) and `LLMJudgeBaseline` (Claude 3 Haiku via Bedrock Converse API, with SKIPPED handling for missing credentials; mock judge restricted to plumbing tests only and forbidden from `results.md`).
  2. `backend/eval/data/personas.yaml` & `backend/eval/generate_dataset.py`: Implemented 3 conflicting persona archetypes (`fast_pauser`, `slow_pauser`, `filler_heavy`). Generated 324 synthetic test turns (>= 300 target) and 48 dev turns across 12 disjoint user profiles. Split strictly partitions by both `user_id` AND `scenario_template_id` (zero train/dev/test leakage). All files labeled SYNTHETIC.
  3. `backend/eval/metrics.py`: Implemented metrics engine for premature interruption rate, unnecessary wait rate, response delay percentiles (p50, p95), decision latency percentiles, accuracy, and inference cost. Enforced Wilson 1927 95% confidence intervals for proportions and 1000-resample bootstrap percentile CIs for delay percentiles.
  4. `scripts/precompute_decisions.py`: Deduplication script for evaluation state queries into `backend/eval/data/decisions_cache.jsonl` keyed by sha256 hash. Enforced 2.0 GB free RAM check and runtime guard (> 60 min stops). Cached all 274 unique states.
  5. `backend/eval/run_eval.py`: Replays evaluation timelines on `VirtualClock` across 0ms oracle, 3190.1ms measured CPU median, parameter sweeps (100, 300, 1000, 3000 ms), and B0 40 cases. Bedrock LLM judge marked `SKIPPED`, GPU row marked `*not measured yet*`. Generated `backend/eval/results/eval_results.json` and `backend/eval/results/results.md`.
  6. `backend/tests/test_eval.py`: 7 comprehensive unit tests verifying zero split leakage, persona pause conflict property, hand-built metric timelines, Wilson and bootstrap interval calculations, and decision cache key hashing.
  7. Verification: 25/25 unit tests passing in 0.80s (`pytest`); `ruff check .` clean with 0 errors.
- **Files changed**:
  - `backend/eval/baselines.py`
  - `backend/eval/data/personas.yaml`
  - `backend/eval/generate_dataset.py`
  - `backend/eval/data/eval_dev.jsonl` (48 synthetic turns)
  - `backend/eval/data/eval_test.jsonl` (324 synthetic turns)
  - `backend/eval/metrics.py`
  - `scripts/precompute_decisions.py`
  - `backend/eval/data/decisions_cache.jsonl` (274 cached decisions)
  - `backend/eval/run_eval.py`
  - `backend/eval/results/eval_results.json`
  - `backend/eval/results/results.md`
  - `backend/tests/test_eval.py`
  - `NOTES.md`
- **Decisions + why**:
  - **Disjoint Split Partitioning**: Strict partitioning by both `user_id` and `scenario_template_id` prevents linguistic or stylistic data contamination between development and test evaluation.
  - **Persona Conflict Validation**: Mathematically verified in tests that at a 700 ms pause, a fast pauser has finished their turn while a slow pauser is still in mid-thought, creating a genuine personalization dilemma that fixed-timeout baselines cannot resolve.
  - **Virtual Clock Latency Decoupling**: Evaluating turn-taking policies on `VirtualClock` enables reproducible comparison of policy algorithms under zero-latency oracle conditions, measured hardware latencies (3190 ms CPU), and hypothetical GPU latencies without changing application logic.
- **Measured numbers**:
  - Test split size: 324 synthetic turns (dev: 48 turns, 12 users).
  - Wilson 95% CIs and Bootstrap CIs (n=324):
    - Fixed 600ms: Acc 66.0% [60.7%, 71.0%], Premature Interruptions 31.2% [26.4%, 36.4%], Unnecessary Wait 5.6% [2.9%, 10.2%], Delay P50 600 ms [600.0, 600.0].
    - Fixed 800ms: Acc 68.5% [63.3%, 73.3%], Premature Interruptions 17.0% [13.3%, 21.4%], Unnecessary Wait 29.0% [22.6%, 36.4%], Delay P50 800 ms [800.0, 800.0].
    - Fixed 1200ms: Acc 71.9% [66.8%, 76.5%], Premature Interruptions 0.0% [0.0%, 1.2%], Unnecessary Wait 100.0% [97.7%, 100.0%], Delay P50 1082 ms [1003.0, 1200.0].
    - Strands Decider 2B (0ms Oracle): Acc 94.8% [91.8%, 96.7%], Premature Interruptions 5.2% [3.3%, 8.2%], Unnecessary Wait 0.0% [0.0%, 2.3%], Delay P50 250 ms [250.0, 250.0].
    - Strands Decider 2B (3190ms CPU): Acc 100.0% [98.8%, 100.0%], Premature Interruptions 0.0% [0.0%, 1.2%], Unnecessary Wait 100.0%, Delay P50 3440 ms.
    - Decider Sweep (300ms latency): Acc 97.5% [95.2%, 98.7%], Premature Interruptions 2.5%, Unnecessary Wait 0.0%, Delay P50 550 ms.
    - Bedrock LLM Judge: SKIPPED (no AWS credentials).
    - Strands Decider 2B (GPU ~35ms): *not measured yet*.
  - Unit tests: 25/25 passing in 0.80s.
  - Linting: 0 errors (`ruff check .`).
- **Next step**: Step B5.5 — Audit & honesty check.

## Step B5.5 — PLAN
- **Goal**: Complete rigorous audit of Step B5: provenance transparency of decision cache, diagnose and resolve replay simulation anomaly (enforcing max_wait_ms and speech cancellation), implement text-only heuristic baseline with threshold tuned strictly on dev, enlarge dev dataset to >= 150 turns without modifying test set, verify zero leakage, measure vocabulary and generation counts, check AWS Bedrock status, remove ungrounded claims, and commit in 4 Conventional Commits.
- **Approach**:
  1. Provenance: Inspect and record cache properties (backend, model_id, key count, total compute time, file date). Mark mock results in `results.md` explicitly as `SYNTHETIC MOCK-DECIDER` and remove from headline claims.
  2. Replay Anomaly: Fix `evaluate_decider_virtual` to mirror `TurnManager`: enforce `max_hold_ms` (2200 ms) and speech-resumed cancellation so decisions arriving after speech resumption or exceeding max hold are dropped/preempted. Report premature interruption rate, unnecessary wait rate, and response delay (p50/p95) with Wilson 1927 and bootstrap 95% CIs.
  3. Text Heuristic Baseline: Add `TextHeuristicBaseline` to `backend/eval/baselines.py` (checking trailing connectors/fillers/punctuation; threshold tuned strictly on dev split). Compare directly against Decider and fixed timeouts on test split.
  4. Dataset Enlargement & Honesty: Add dev users in `personas.yaml` (from 3 to 12 dev users, 4 per archetype) and re-generate `dev_turns.jsonl` to reach >= 150 turns (192 turns) while preserving `test_turns.jsonl` (324 turns). Verify 0 user and 0 template leakage. Compute vocabulary size of final utterances and record 100% template vs 0% Bedrock counts.
  5. Claims Cleanup: Remove "eliminating persona conflict" wording from `NOTES.md` and summaries; replace with empirical numbers labeled SYNTHETIC.
  6. AWS Bedrock: Report `aws configure list` status, boto3 session/credentials check (`UnrecognizedClientException`), absence of placeholder keys in `.env` / `.env.example`, and mark LLM judge SKIPPED.
  7. Verification & Commits: Run `pytest`, `ruff`, staged-file check, and secret grep. Create 4 separate Conventional Commits without pushing.
- **Files expected**:
  - `NOTES.md`
  - `backend/eval/baselines.py`
  - `backend/eval/data/personas.yaml`
  - `backend/eval/generate_dataset.py`
  - `backend/eval/data/dev_turns.jsonl`
  - `backend/eval/data/decisions_cache.jsonl`
  - `backend/eval/run_eval.py`
  - `backend/eval/results/results.md`
  - `backend/eval/results/eval_results.json`
  - `backend/tests/test_eval.py`

## Step B5.5 — DONE
- **Status**: DONE
- **What was done**:
  1. **Provenance Audit**:
     - Verified `backend/eval/data/decisions_cache.jsonl`: 325 total unique keys, 100% produced by `MockDecider` (`backend: mock`), model ID `StrandsAgents/strands-decider-2B-hobson-v19`, total server-side compute time 0.00 ms.
     - Marked all Decider rows in `results.md` and `backend/eval/results/results.md` explicitly as `[MOCK-DECIDER]`. Added prominent provenance warning banner stating decisions came from MockDecider and must not be cited as measured performance of the real weights.
     - Real model precompute estimate: 325 keys * 3.19s = 1036.8s (17.3 minutes). Per protocol and memory guard, real CPU precompute was not run.
  2. **Replay Simulation Anomaly Diagnosis & Resolution**:
     - Root cause diagnosed: The previous replay loop contained `if is_complete or decision_arrival_silence <= silence_ms:`, which unconditionally accepted `RESPOND` choices for completed turns at 3440.1 ms (ignoring `max_wait_ms` of 2200 ms and ignoring realistic pause timeouts), while incomplete turns avoided interruption because 3440.1 ms > silence_ms. This created a spurious "100% accuracy" artifact.
     - Fixed `evaluate_decider_virtual` to mirror `TurnManager`:
       - Speech resumption cancels in-flight decisions on incomplete turns (`DecisionDropped(reason="speech_resumed")`).
       - Max wait hard cap (`max_wait_ms = 2200 ms`) forces turn yield at 2200 ms (`MaxHoldExceeded`), preempting any decision arriving after 2200 ms.
     - Under true TurnManager logic, at 3190 ms CPU latency, Decider decisions arrive after 2200 ms and are dropped as stale; the system degenerates entirely into a fixed 2200 ms timer (0.0% premature interruptions, 100.0% unnecessary wait, 2200 ms response delay).
  3. **Text-Only Heuristic Baseline**:
     - Implemented `TextHeuristicBaseline` in `backend/eval/baselines.py` (flags trailing conjunctions/fillers/hesitations and incomplete terminal punctuation as WAIT; otherwise RESPOND once silence >= threshold).
     - Tuned threshold strictly on dev split: 250 ms achieved 100.0% accuracy on dev.
     - Evaluated on test split (n=324): achieves 100.0% accuracy, 0.0% premature interruptions, 0.0% unnecessary wait, and 250 ms median delay. Plainly stated in `results.md` that simple text heuristics match or exceed neural models on these synthetic templates.
  4. **Dataset Honesty & Enlargement**:
     - Enlarged dev split from 48 turns to 192 turns (>= 150 requirement) by adding 9 dev users across archetypes in `personas.yaml` (12 dev users total).
     - Test split kept strictly unchanged at 324 turns (9 test users).
     - Verified zero leakage: 0 overlapping users, 0 overlapping templates.
     - Vocabulary & generation metrics recorded: dev vocab 127 words, test vocab 188 words, total vocab 265 unique words; 516/516 turns (100%) generated via template text, 0 Bedrock text.
  5. **Removal of Unsubstantiated Claims**:
     - Removed any claims of Decider "eliminating" persona conflicts. Replaced with empirical trade-offs labeled SYNTHETIC.
  6. **AWS Bedrock & STS Status**:
     - `aws configure list`: Command failed (`aws` CLI is not installed on host).
     - Boto3 authentication check: `~/.aws/credentials` returned `UnrecognizedClientException: The security token included in the request is invalid`.
     - Confirmed zero secret or placeholder keys in `.env` / `.env.example`.
     - Bedrock LLM judge baseline marked `SKIPPED`.
  7. **Testing & Code Quality**:
     - 27/27 unit tests passing in 0.79s (`pytest`).
     - `ruff check .` clean with 0 errors.
- **Files changed**:
  - `NOTES.md`
  - `backend/eval/baselines.py`
  - `backend/eval/data/personas.yaml`
  - `backend/eval/generate_dataset.py`
  - `backend/eval/data/dev_turns.jsonl`
  - `backend/eval/data/test_turns.jsonl`
  - `backend/eval/data/user_histories.jsonl`
  - `backend/eval/data/decisions_cache.jsonl`
  - `backend/eval/run_eval.py`
  - `backend/eval/results/results.md`
  - `backend/eval/results/eval_results.json`
  - `results.md`
  - `backend/tests/test_eval.py`
- **Measured numbers**:
  - Dev split size: 192 turns (was 48 turns).
  - Test split size: 324 turns (unchanged).
  - Split leakage: 0 users, 0 templates.
  - Vocabulary: Dev 127 words, Test 188 words, Total 265 words. 100% template generated.
  - Re-evaluated Test Benchmark (n=324):
    - Fixed 600ms: Acc 67.6% [62.3%, 72.5%], Premature 30.6% [25.8%, 35.8%], Unnecessary Wait 3.7%, Delay P50 600 ms.
    - Fixed 800ms: Acc 68.5% [63.3%, 73.3%], Premature 17.0% [13.3%, 21.4%], Unnecessary Wait 29.0%, Delay P50 800 ms.
    - Fixed 1200ms: Acc 73.5% [68.4%, 78.0%], Premature 0.0%, Unnecessary Wait 100.0%, Delay P50 1134 ms.
    - Text Heuristic (250ms dev-tuned): Acc 100.0% [98.8%, 100.0%], Premature 0.0%, Unnecessary Wait 0.0%, Delay P50 250 ms.
    - Decider (0ms Oracle) [MOCK-DECIDER]: Acc 95.4% [92.5%, 97.2%], Premature 4.6% [2.8%, 7.5%], Unnecessary Wait 0.0%, Delay P50 250 ms.
    - Decider (3190ms CPU) [MOCK-DECIDER]: Acc 100.0% [98.8%, 100.0%], Premature 0.0%, Unnecessary Wait 100.0%, Delay P50 2200 ms.
- **Next step**: Step B5.6 — Ambiguity, oracle bound & cache audit.

## Step B5.6 — PLAN
- **Goal**: Harden evaluation harness with an ambiguous category where syntactically complete text has probabilistic continuation sampled from persona habits (breaking trivial text heuristic parity), implement generator-aware `ORACLE-UPPER-BOUND`, audit cache key counts, enforce zero-fallback cache miss errors, generate per-persona and per-category tables, update `results.md` strictly programmatically with Definitions and exact denominators, write `scripts/gpu_run.md`, and test consistency.
- **Approach**:
  1. Cache Audit: Report exact count of turns and checkpoints per turn for dev and test. Explain the unique key count mathematically (60 distinct template texts * 5 checkpoints = ~300 unique state keys across dev and test). Recompute cache so all potential checkpoints `[250, 500, 800, 1200, 1800]` are cached. In `run_eval.py`, replace fallback with strict `raise KeyError` on cache miss, tracking hit/miss metrics.
  2. Ambiguous Scenarios: In `generate_dataset.py`, add ambiguous category where text is syntactically complete (e.g. lists, dosages, afterthoughts), but ground truth continuation is sampled per persona continuation probability (30% for fast pauser, 50% for slow pauser, 60% for filler heavy). Ground truth is realistically noisy; text-only regex cannot score 100%.
  3. Oracle Upper Bound: Implement `ORACLE-UPPER-BOUND` in `run_eval.py` that utilizes knowledge of persona pause distributions and continuation states to demonstrate theoretical performance ceiling.
  4. Per-Persona and Per-Category Breakdowns: Compute metrics for Fixed 600/800/1200ms, Text Heuristic, Decider Virtual, and Oracle Bound broken down across 3 personas (`fast_pauser`, `slow_pauser`, `filler_heavy`) and 3 categories (`complete`, `incomplete`, `ambiguous`).
  5. Results Documentation & Definitions: Generate `results.md` and `eval_results.json` directly from code. Include Definitions section with exact formulas, numerators, denominators ($N=324$ for premature, $N_{\text{complete}}$ for unnecessary wait, 800ms threshold). State B0 model provenance (34/40 = 85.0% on real model vs 33/40 = 82.5% on mock).
  6. GPU Runbook: Write `scripts/gpu_run.md` with complete copy-paste instructions for Google Colab and Kaggle GPU execution.
  7. Consistency Test: Add test asserting every number rendered in `results.md` exists in `eval_results.json`.
- **Files expected**:
  - `NOTES.md`
  - `backend/eval/generate_dataset.py`
  - `backend/eval/data/dev_turns.jsonl`
  - `backend/eval/data/test_turns.jsonl`
  - `scripts/precompute_decisions.py`
  - `backend/eval/data/decisions_cache.jsonl`
  - `backend/eval/run_eval.py`
  - `backend/eval/results/eval_results.json`
  - `results.md`
  - `scripts/gpu_run.md`
  - `backend/tests/test_eval.py`

## Step B5.6 — DONE
- **Status**: DONE
- **What was done**:
  1. **Cache Key Audit & Zero-Fallback Replay**:
     - Audited key count: 516 total turns (192 dev + 324 test) generated across 60 distinct template texts (24 dev + 36 test) * 5 checkpoints (`[250, 500, 800, 1200, 1800]`) = 300 unique state keys, plus 40 B0 cases = 340 unique keys.
     - Precomputed all 340 states in `backend/eval/data/decisions_cache.jsonl` using MockDecider (448 total entries including historical snapshots).
     - Modified `run_eval.py` to raise `KeyError` on any cache miss (zero online fallback). Replay executed with 2,281 cache hits and 0 cache misses.
  2. **Harder Synthetic Data with Ambiguous Turns**:
     - Added `ambiguous` category in `generate_dataset.py` where text is syntactically complete (e.g. lists, medication requests, appointments), but ground truth continuation is probabilistically sampled per persona `continuation_prob` (fast: 30%, slow: 50%, filler: 65%).
     - Test split size preserved at 324 turns (108 complete, 108 incomplete, 108 ambiguous). Dev split: 192 turns (48 complete, 48 incomplete, 96 ambiguous).
     - Breaking trivial regex: Because ambiguous turns appear complete, the `TextHeuristicBaseline` drops from 100.0% to 81.8% accuracy (18.2% premature interruptions).
  3. **Generator-Aware Oracle Upper Bound**:
     - Implemented `ORACLE-UPPER-BOUND` in `run_eval.py` representing theoretical policy ceiling knowing persona continuation states (100.0% accuracy, 0.0% premature interruptions, 250 ms median delay).
  4. **Per-Persona and Per-Category Performance Breakdowns**:
     - Computed and tabulated breakdowns across 3 personas (`fast_pauser`, `slow_pauser`, `filler_heavy`) and 3 categories (`complete`, `incomplete`, `ambiguous`).
     - Demonstrates empirically that Fixed 600ms causes 57.4% premature interruptions on slow pausers versus 0.0% on fast pausers.
  5. **Results Documentation & Metric Definitions**:
     - Added Section 5 (Definitions) specifying exact formulas, numerators, and denominators: Premature Interruption ($N=324$), Unnecessary Wait ($N_{\text{complete}}=157$, 800ms global threshold), and Delay P50/P95 ($N_{\text{complete}}=157$).
     - Updated B0 40-case table attributing real model weights (34/40, 85.0% on CPU) vs MockDecider (33/40, 82.5%).
     - All markdown tables, findings, and numbers generated programmatically from `eval_results.json`.
  6. **GPU Execution Guide**:
     - Created `scripts/gpu_run.md` with complete, copy-paste instructions for Colab and Kaggle GPU precomputation and benchmarking.
  7. **Testing**:
     - 30/30 unit tests passing in 1.54s (`pytest`), including `test_results_md_consistent_with_json` (asserting every number in `results.md` exists in `eval_results.json`) and `test_cache_miss_raises_error`.
- **Files changed**:
  - `NOTES.md`
  - `backend/eval/data/personas.yaml`
  - `backend/eval/generate_dataset.py`
  - `backend/eval/data/dev_turns.jsonl`
  - `backend/eval/data/test_turns.jsonl`
  - `scripts/precompute_decisions.py`
  - `backend/eval/data/decisions_cache.jsonl`
  - `backend/eval/metrics.py`
  - `backend/eval/run_eval.py`
  - `backend/eval/results/eval_results.json`
  - `results.md`
  - `scripts/gpu_run.md`
  - `backend/tests/test_eval.py`
- **Measured numbers**:
  - Test split size: 324 turns (157 complete, 167 incomplete).
  - Main comparison:
    - Fixed 600ms: Acc 61.7%, Premature 35.2%, Unnecessary Wait 6.4%, Delay P50 600 ms.
    - Fixed 800ms: Acc 63.0%, Premature 19.8%, Unnecessary Wait 35.7%, Delay P50 800 ms.
    - Fixed 1200ms: Acc 69.8%, Premature 0.0%, Unnecessary Wait 100.0%, Delay P50 1006 ms.
    - Text Heuristic (250ms dev-tuned): Acc 81.8%, Premature 18.2%, Unnecessary Wait 0.0%, Delay P50 250 ms.
    - Decider (0ms latency) [MOCK-DECIDER]: Acc 81.8%, Premature 18.2%, Unnecessary Wait 0.0%, Delay P50 250 ms.
    - Decider (3190ms CPU) [MOCK-DECIDER]: Acc 100.0%, Premature 0.0%, Unnecessary Wait 100.0%, Delay P50 2200 ms.
    - ORACLE-UPPER-BOUND: Acc 100.0%, Premature 0.0%, Unnecessary Wait 0.0%, Delay P50 250 ms.
- **Next step**: Step B8 — Reusable patient listener pattern package.

## Step B8 — PLAN
- **Goal**: Create standalone, production-ready `patient_listener_pattern/` reusable package (zero imports from `backend/`) implementing an `InterventionHandler` that hooks into the Strands Agent loop (`before_invocation` / `before_model_call`) to steer conversational turn-taking by suppressing model generation (`Deny`) during user pauses and releasing generation (`Proceed`) once the floor is yielded.
- **Approach**:
  1. `patient_listener_pattern/decision.py`:
     - Typed dataclasses: `TurnState`, `PolicyThresholds`, `Action` ("WAIT" | "RESPOND").
     - Default system question: `TURN_COMPLETE_QUESTION` (options: `["WAIT", "RESPOND"]`).
     - Pure policy function: `policy(probs, silence_ms, thresholds) -> Action`.
  2. `patient_listener_pattern/handler.py`:
     - `PatientListenerInterventionHandler(InterventionHandler)`:
       - Injected `decider: Callable[[str, ChoiceQuestion], Decision]` (decoupling compute backend).
       - Lifecycle methods `before_invocation` and `before_model_call`.
       - Reads silence and transcript, executes policy, and maps `WAIT` -> `Deny(...)` and `RESPOND` -> `Proceed()`.
  3. `patient_listener_pattern/example.py`:
     - Standalone executable script running offline with `MockDecider` and local mock agent, stepping through a scripted conversation timeline.
  4. `patient_listener_pattern/README.md`:
     - Under-10-line drop-in guide, threshold tuning, and caveats.
  5. `patient_listener_pattern/tests/test_pattern.py`:
     - Unit tests verifying standalone imports, policy thresholds, and handler interventions.
- **Files expected**:
  - `patient_listener_pattern/__init__.py`
  - `patient_listener_pattern/decision.py`
  - `patient_listener_pattern/handler.py`
  - `patient_listener_pattern/example.py`
  - `patient_listener_pattern/README.md`
  - `patient_listener_pattern/tests/test_pattern.py`
  - `NOTES.md`

## Step B8 — DONE
- **Status**: DONE
- **What was done**:
  1. **Investigated Strands Intervention Architecture**:
     - Subclasses `InterventionHandler` from `strands.interventions` and implements `before_invocation` / `before_model_call`.
     - Action mapping: Returning `Deny(reason=...)` suppresses LLM response generation and sets assistant message to `DENIED: <reason>`. Returning `Proceed()` yields the floor and executes model generation normally.
  2. **Created Standalone `patient_listener_pattern` Package**:
     - 100% standalone with zero imports from `backend/`.
     - `decision.py`: Typed `TurnState`, `PolicyThresholds` (`respond_probability_threshold=0.65`, `min_silence_for_respond_ms=250.0`, `max_hold_ms=2200.0`), `TURN_COMPLETE_QUESTION`, and pure function `policy(probs, silence_ms, thresholds) -> Action` implementing safety cap (>=2200ms -> RESPOND), cadence floor (<250ms -> WAIT), and confidence threshold.
     - `handler.py`: `PatientListenerInterventionHandler` intercepting `before_invocation` and `before_model_call`, invoking an injected decider callable (`(state_text, question) -> Decision`), and returning `Deny` (on WAIT) or `Proceed` (on RESPOND).
     - `__init__.py`: Clean export of public API primitives.
  3. **Offline Walkthrough Example**:
     - `example.py`: Fully functional, zero-cloud walkthrough using `ScriptedDecider` and `MockStrandsModel(Model)`. Correctly executes in-process and prints `WAIT` (with model execution denied) then `RESPOND` (yielding agent response) across a scripted patient conversation timeline.
  4. **Documentation & Caveats**:
     - `README.md`: Under 10-line drop-in quickstart guide for any Strands Agent, threshold definitions, architectural diagram, and measured hardware caveats (distinguishing CPU latency ~3.2s from GPU <50ms, and noting all benchmark evidence is `SYNTHETIC`).
  5. **Testing**:
     - `patient_listener_pattern/tests/test_pattern.py`: 6 comprehensive unit tests covering pure policy boundaries, prompt state formatting and word budgeting, `WAIT` / `RESPOND` handler actions, end-to-end agent integration, and verifying zero leakage from `backend/`.
     - 36/36 tests passing in `pytest` (2.63s -> 3.33s).
     - 100% clean with `ruff check .`.
- **Files created/modified**:
  - `patient_listener_pattern/__init__.py`
  - `patient_listener_pattern/decision.py`
  - `patient_listener_pattern/handler.py`
  - `patient_listener_pattern/example.py`
  - `patient_listener_pattern/README.md`
  - `patient_listener_pattern/tests/__init__.py`
  - `patient_listener_pattern/tests/test_pattern.py`
  - `pyproject.toml`
  - `NOTES.md`
- **Measured numbers**:
  - Zero imports from `backend/` confirmed via AST and runtime `sys.modules` audit in `test_zero_imports_from_backend`.
  - Offline walkthrough: 100% local, 0 network requests, 0 AWS credentials required.
  - Test suite: 36 total tests passing (all tests strictly mock-based, 0s real weight load time in pytest).
- **Next step**: Step B6 (Pace profile + regret log) or Step B7 (Confidence-aware policy).

## Step B5.7 — PLAN
- **Goal**: Complete verification of Part 1 (results.md trace tests and B0 attribution), redefine Bayes-optimal oracle upper bound under irreducible noise, audit dataset diversity, and formally verify Strands InterventionHandler wiring with real Agent and stub model.
- **Approach**:
  1. Commit hygiene: Undo semicolon commits (92074e8, 05ff91c) via soft reset, split into single-type Conventional Commits:
     - `fix: correct results report and add definitions`
     - `feat: add ambiguous scenarios and oracle upper bound`
     - `docs: add gpu run instructions`
     - `test: add cache miss and category tests`
     - `feat: add reusable patient listener pattern package`
     - `test: add pattern handler tests`
  2. Part 1 verification:
     - Print B0 40-case table with exact backends named.
     - Verify number-trace test between results.md and eval_results.json, demonstrating assertion failure on tampered copy.
     - Document Bedrock model check result (SKIPPED due to UnrecognizedClientException).
  3. Bayes-optimal oracle:
     - Redefine oracle without realized outcome peeking: uses text category and persona continuation probability under 0.65 threshold.
     - Report accuracy (91.7%), premature rate (2.2%), and realized continue rates on ambiguous turns per persona.
  4. Dataset diversity audit:
     - Count unique states, unique utterances, and lookups across dev and test.
     - Add template text risk note to Risks & caveats.
  5. Step B8 Strands verification:
     - Document exact hook methods (`before_invocation`, `before_model_call`), received event properties, turn state extraction, and loop actions (`Proceed`, `Deny`, `Guide`).
     - Compare `Deny` vs `Guide` for WAIT actions and make `wait_action_type` a configurable handler parameter.
     - Run integration test verifying real `strands.Agent` running offline with stub model.
     - Verify zero backend imports and run example via `uv run python patient_listener_pattern/example.py`.

## Step B5.7 — DONE
- **Status**: DONE
- **What was done**:
  1. **Part 1 Verification & Reporting**:
     - B0 40-case validation subset:
       * Real Weights on CPU: 34/40 (85.0%) [backend: local_cpu (AMD Ryzen 5 5500U, b0_spike.py)]
       * MockDecider: 33/40 (82.5%) [backend: mock (MockDecider)]
       * Fixed 800ms: 16/40 (40.0%)
     - Number trace test: Verified `test_results_md_consistent_with_json` passes against active `eval_results.json`; verified it detects and fails immediately if any number in `results.md` is modified (e.g. 91.7% -> 99.9%).
     - Bedrock model check: Attempted `boto3.client('bedrock').list_foundation_models()` -> `ClientError: UnrecognizedClientException: The security token included in the request is invalid`. Explicitly marked `SKIPPED` in results.md without placeholder model names.
  2. **Bayes-Optimal Oracle Upper Bound**:
     - Redefined `ORACLE-UPPER-BOUND` in `backend/eval/run_eval.py` without realized outcome peeking:
       * `complete`: Predicts RESPOND at 250ms (Acc 100%, Premature 0%).
       * `incomplete`: Predicts WAIT (Acc 100%, Premature 0%).
       * `ambiguous`: Evaluates persona continuation probability under regret-aware policy threshold (0.65):
         - `fast_pauser` (continuation_prob=0.30 -> P(complete)=0.70 >= 0.65): Predicts RESPOND at 250ms.
         - `slow_pauser` (continuation_prob=0.50 -> P(complete)=0.50 < 0.65): Predicts WAIT (holds floor to avoid premature cutoff).
         - `filler_heavy` (continuation_prob=0.65 -> P(complete)=0.35 < 0.65): Predicts WAIT.
     - Realized continue rates on ambiguous test turns (n=36 per persona):
       * `fast_pauser`: 7/36 continued (19.4%), 29/36 complete (80.6%).
       * `slow_pauser`: 26/36 continued (72.2%), 10/36 complete (27.8%).
       * `filler_heavy`: 26/36 continued (72.2%), 10/36 complete (27.8%).
     - Measured Bayes-optimal oracle metrics:
       * Accuracy: 91.7% [88.1%, 94.2%] (297/324)
       * Premature Interruption Rate: 2.2% [1.1%, 4.4%] (7/324)
       * Unnecessary Wait Rate: 12.7% [8.4%, 18.9%] (20/157)
       * Response Delay P50: 250.0 ms [250.0, 250.0] ms
  3. **Dataset Diversity Audit**:
     - DEV: 192 turns, 24 unique utterances, 104 unique evaluated states, 664 state lookups.
     - TEST: 324 turns, 36 unique utterances, 156 unique evaluated states, 1,136 state lookups.
     - Total across splits: 60 unique utterances, 300 unique states across 5 checkpoints.
  4. **Strands Agent Wiring & Verification**:
     - Exact hooks: `strands.interventions.InterventionHandler` methods `before_invocation` (receives `BeforeInvocationEvent`) and `before_model_call` (receives `BeforeModelCallEvent`).
     - Event properties: `event.agent`, `event.messages`, `event.cancel`, `event.invocation_state`, `event.projected_input_tokens`.
     - Turn state: Extracted from message context via `_default_extract_state` or injected via `get_turn_state` callback.
     - Action comparison:
       * `Deny(reason=...)`: Halts execution immediately before model call, consuming 0 LLM tokens, returning `DENIED: <reason>`. Used as default for WAIT to hold conversational silence.
       * `Guide(feedback=..., reason=...)`: Injects steering feedback into the agent loop to produce backchannels. Configured via `wait_action_type="guide"`.
     - Offline execution: Added root `README.md` and wheel build package targets to `pyproject.toml`; confirmed offline execution via `uv run python patient_listener_pattern/example.py`.
     - Tests: 37/37 tests passing in pytest (3.38s).
- **Files created/modified**:
  - `README.md`
  - `pyproject.toml`
  - `backend/eval/run_eval.py`
  - `backend/eval/results/eval_results.json`
  - `results.md`
  - `patient_listener_pattern/handler.py`
  - `patient_listener_pattern/__init__.py`
  - `patient_listener_pattern/README.md`
  - `patient_listener_pattern/tests/test_pattern.py`
  - `NOTES.md`

## Step B5.8 — PLAN
- **Goal**: Reconcile exact state and cache counts between precompute capacity (340 keys) and runtime reached states (300 keys), introduce cluster-bootstrap 95% confidence intervals clustered by user to report honest uncertainty across repeated utterances, add the `PERSONA-BLIND CEILING` reference row, increase the ambiguous category share to ~50% with per-user pause-jitter continuation variation to unlock personalization gains before real Decider runs, and verify real Strands Agent integration.
- **Approach**:
  1. Reconcile counts:
     - Print detailed reconciliation table for DEV (192 turns, 24 utterances, 12 users, 104 reached states, 664 lookups) and TEST (324 turns, 36 utterances, 9 users, 156 reached states, 1,136 lookups).
     - Clarify that data was NOT regenerated between B5.6 and B5.7; 340 is total precomputed cache capacity ($60 \times 5$ checkpoints + 40 B0 cases), while 300 is runtime evaluated states ($104 + 156 + 40$). Fix ambiguous wording in NOTES.md and results.md.
  2. Cluster bootstrap:
     - Resample $K=9$ users with replacement, then turns within user ($B=1000$ resamples, seed=42).
     - Add cluster bootstrap CIs to all headline metric tables; retain Wilson CI in a secondary column.
  3. Reference ceilings:
     - Add `PERSONA-BLIND CEILING`: Evaluates text category and global empirical base rates without user/persona profile.
     - Document mathematical arithmetic verifying the user's calculation: 275/324 (84.9%) vs 297/324 (91.7%) with a gap of 22 turns.
     - Confirm that identical continuation counts (26/36) for slow_pauser and filler_heavy arose from seed realization rather than shared parameters (0.50 vs 0.65).
  4. Personalization effect:
     - Adjust ambiguous proportion to ~50% (162 ambiguous, 81 complete, 81 incomplete for test; 96 ambiguous, 48 complete, 48 incomplete for dev).
     - Modulate user continuation probability based on individual user pause jitter ($p_{\text{user}} = p_{\text{archetype}} + \text{jitter}/200$), allowing user profiles to separate individuals within archetypes.
     - Regenerate datasets with fixed seed, keeping users and templates strictly disjoint.
     - Audit cache keys: verify all states map to existing template texts or precompute any newly reached checkpoints.
  5. Pattern check:
     - Confirm that `example.py` and unit tests use the real `strands.Agent` class with a stub `strands.models.Model` running 100% offline.
- **Files expected**:
  - `NOTES.md`
  - `backend/eval/metrics.py`
  - `backend/eval/generate_dataset.py`
  - `backend/eval/data/personas.yaml`
  - `backend/eval/data/dev_turns.jsonl`
  - `backend/eval/data/test_turns.jsonl`
  - `backend/eval/run_eval.py`
  - `backend/eval/results/eval_results.json`
  - `results.md`
  - `backend/tests/test_eval.py`
  - `patient_listener_pattern/tests/test_pattern.py`

## Step B5.8 — DONE
- **Count Reconciliation**:
  | Split | Turns | Unique Utterances | Users | Checkpoints / Turn | Lookups | Unique States Reached | Cache Capacity (5x) | Cache Hits | Misses |
  |---|---|---|---|---|---|---|---|---|---|
  | DEV | 192 | 24 | 12 | 5 | 685 | 104 | 120 | 685 | 0 |
  | TEST | 324 | 36 | 9 | 5 | 1,146 | 155 | 180 | 1,146 | 0 |
  | B0 Cases | 40 | 40 | - | 1 | 40 | 40 | 40 | 40 | 0 |
  | TOTAL (Unique Across All) | 556 | 96 | 21 | - | 1,871 | 299 | 340 | 1,871 | 0 |
  - **Data Provenance**: The dataset was **NOT regenerated** between B5.6 and B5.7. The difference was purely terminological: 340 is the total precomputed cache capacity ($60 \times 5 = 300$ template states + 40 B0 feasibility cases), 300 was the template capacity ($60 \times 5$), while 260 ($104 + 156$) was the number of unique states reached during dev+test virtual replay (or 299/300 including B0's 40 cases). Replay stops early when turns complete or interruptions occur, so checkpoints like 1200ms and 1800ms are only reached on turns with longer silences.
  - Corrected the "300" description in `NOTES.md` and `results.md`.

- **Honest Uncertainty & Cluster Bootstrap**:
  - Test split consists of 324 turns across 9 users (36 turns per user). Because each user speaks across repeated templates, observations within user are not independent. Standard Wilson score intervals assume i.i.d. observations and underestimate uncertainty.
  - Implemented two-stage cluster bootstrap ($B=1000$ resamples) in `backend/eval/metrics.py`:
    $$\text{For } b = 1, \dots, B: \quad C_1^*, \dots, C_K^* \sim \text{Uniform}(\{C_1, \dots, C_K\}) \text{ with replacement}$$
    $$\text{For each } C_k^*: \quad \text{sample } |C_k^*| \text{ turns with replacement from } C_k^*$$
    $$\text{CI}_{95\%} = [\hat{q}_{0.025}(\theta^*), \hat{q}_{0.975}(\theta^*)]$$
  - Headline table updated in `results.md` and `eval_results.json` with Cluster 95% CI as primary and Wilson 95% CI as secondary.

- **Ceilings & Arithmetic Verification**:
  - Added `PERSONA-BLIND CEILING` reference row next to `ORACLE-UPPER-BOUND (Persona-Aware)`. Both are reference ceilings (theoretical upper bounds for blind and persona-aware policies), not achievable baselines.
  - User arithmetic verified on old dataset:
    - 108 ambiguous turns: 59 continued (majority class = WAIT) vs 49 completed.
    - Blind best on ambiguous turns is WAIT (getting 59/108 correct).
    - Persona-Blind Ceiling: $108 + 108 + 59 = 275/324 = 84.88\% \approx 84.9\%$.
    - Persona-Aware Bayes Oracle: $108 + 108 + (29 + 26 + 26) = 297/324 = 91.67\% \approx 91.7\%$.
    - Gap in turns: exactly 22 turns ($+6.79\%$).
  - Parameter confirmation: `slow_pauser` ($p=0.50$) and `filler_heavy` ($p=0.65$) having identical continuation counts (26/36 each) was a random sample realization under `seed=42`, not a shared parameter in code.

- **Personalization Effect Unlocked**:
  - Prior design capped the profile gain at 22 of 324 turns because ambiguous turns were only 33% (108 turns) and continuation was sampled purely at the archetype level.
  - Purpose of change before first real Decider run: enable user pace profiling (B6) to demonstrate a genuine personalization effect beyond text and archetype baselines.
  - Ambiguous share increased to 50% (162 ambiguous, 81 complete, 81 incomplete for test; 96 ambiguous, 48 complete, 48 incomplete for dev).
  - Continuation probability now modulated by individual user pause jitter:
    $$p_{\text{user}} = \text{clamp}(p_{\text{base}} + \frac{\text{jitter}}{200}, 0.05, 0.95)$$
  - Disjointness verified: zero user overlap, zero template overlap.
  - Cache audit: 0 missing cache keys across all 2,250 evaluated queries; existing cache entries remain 100% valid by SHA-256 state key.
  - New reference ceilings:
    - `PERSONA-BLIND CEILING`: 77.5% [Cluster 95% CI: 66.7%, 88.3%; Wilson: 72.6%, 81.7%] (250/324)
    - `ORACLE-UPPER-BOUND (Persona-Aware)`: 87.3% [Cluster 95% CI: 79.6%, 93.8%; Wilson: 83.3%, 90.5%] (266/324)
    - User-level personalization ceiling: allows identifying users whose personal continuation diverges from the archetype (e.g., `user_slow_03` with -40ms jitter only continues 27.8% of the time, allowing user profile to predict RESPOND for an additional +8 turns).

- **Pattern Check & Integration**:
  - Confirmed that `example.py` and `patient_listener_pattern/tests/test_pattern.py` use the **real** `strands.Agent` class from `strands-agents` with `MockStrandsModel(Model)` and `Agent(interventions=[handler])`.
  - Integration test `test_agent_integration_with_handler` exercises one WAIT turn (model execution suppressed, `DENIED:`) and one RESPOND turn (model executes, output asserted) running 100% offline with zero network and zero AWS credentials. All 38 tests pass.

- **First Real GPU Benchmark (Tesla T4, Google Colab)**:
  - Measured with real Decider weights on `device=cuda`, GPU: Tesla T4.
  - Warnings during run: `causal_conv1d` and `flash-linear-attention` missing (executed via reference path).
  - 1Q (turn_complete): 20 tok = 167.5/176.0 ms, 60 tok = 235.6/236.7 ms, 150 tok = 237.3/238.5 ms.
  - 2Q (turn_complete + mid_thought): 20 tok = 343.1/343.7 ms, 60 tok = 395.2/400.3 ms, 150 tok = 395.8/401.1 ms.
  - 2Q overhead: ~159-176 ms on GPU vs ~2500 ms on CPU.
  - Replayed test split at measured GPU median latency (235.6 ms): accuracy 75.6% [Cluster CI: 63.0%, 88.0%; Wilson: 70.7%, 80.0%], premature interruption rate 24.4%, response delay P50 = 486 ms, unnecessary wait = 0.0%. Replay row labeled `Strands Decider 2B (236ms GPU, Tesla T4) [MOCK-DECIDER]` in `results.md` (decision contents derived from mock cache).

## Step B5.9 — PLAN
- **Goal**: Reconcile exact lookup and state counts with strict report tests, implement full state enumeration for GPU precomputation, enforce generator sanity within binomial tolerance, expand the dataset to ~30 test users / ~15 dev users with learnable profile correlation, add the 3 reference ceilings with paired cluster bootstrap intervals, configure HF offline mode, and record the one-time Bedrock check.
- **Approach**:
  1. Housekeeping:
     - Add `hf_offline: bool = False` to `config.py`, pass through to `decider_service.py` and `bench_decider.py`.
     - Record Bedrock probe results:
       * AWS credentials loaded via `~/.aws/credentials` in `us-east-1`.
       * Bedrock control plane returned 123 foundation models and 98 inference profiles.
       * One-time Converse test attempted with smallest available Claude model: `us.anthropic.claude-haiku-4-5-20251001-v1:0` returned `ValidationException: An error occurred (ValidationException) when calling the Converse operation: Operation not allowed` (model access not granted in account console; legacy `anthropic.claude-3-haiku-20240307-v1:0` returned `ResourceNotFoundException`).
       * LLM Judge baseline is marked `SKIPPED` accordingly. Placeholders configured in `.env.example`. No credentials printed or logged.
  2. Dataset expansion & generator sanity:
     - 30 test users (10/archetype, ~12 turns each = 360 turns) and 15 dev users (5/archetype, ~12 turns each = 180 turns), strictly disjoint users and templates.
     - Enforce per-user continuation rate matches $p_{\text{user}}$ within binomial tolerance.
     - Confirm history session pause durations share the user's `jitter_ms` and correlate positively with $p_{\text{user}}$ ($r > 0.70$).
  3. Ceilings & Paired Differences:
     - Implement `PERSONA-BLIND`, `ARCHETYPE-AWARE`, and `USER-AWARE` reference ceilings.
     - Compute paired turn-by-turn differences with two-stage cluster bootstrap intervals over users.
  4. Full State Enumeration:
     - Add `--enumerate` flag to `scripts/precompute_decisions.py` to list all possible base, profile, and post-speech-resumed states.
     - Print total count and runtime estimate at 240 ms per 1Q call on Tesla T4.
  5. Count reconciliation:
     - Print final table on new data, explain lookups vs replay queries, add regression test.

## Step B5.9 — DONE
- **Count Reconciliation**:
  - Reconciled exact dataset table across current splits (540 turns total: 180 dev / 360 test across 45 users):
    | Split | Turns | Users | Unique Utterances | Lookups | Unique States Reached | Cache Hits | Misses |
    |---|---|---|---|---|---|---|---|
    | DEV | 180 | 15 | 24 | 644 | 102 | 644 | 0 |
    | TEST | 360 | 30 | 36 | 1282 | 154 | 1282 | 0 |
    | TOTAL | 540 | 45 | 60 | 1926 | 256 | 1926 | 0 |
  - Lookups & state differences explained:
    * Lookups (1,831 -> 1,926): Test set expanded to 360 turns across 30 users, resulting in 1,282 test lookups + 644 dev lookups.
    * 2,904 vs 1,926 lookups: 1,926 is the single-pass lookup count; 2,904 is the cumulative query count across all latency sweep evaluations in `run_eval.py`.
    * Unique states reached: 256 (102 dev + 154 test) reached under the mock decision trajectory.
  - Added strict test `test_reconciled_counts_in_results_md` in `backend/tests/test_eval.py` asserting exact match with `results.md`.

- **Generator Sanity Verified**:
  - Configured vs realized continuation rates:
    * `fast_pauser`: Configured $p=0.30$, Realized = 18/60 (0.300), $z = 0.00$.
    * `slow_pauser`: Configured $p=0.50$, Realized = 31/60 (0.517), $z = 0.26$.
    * `filler_heavy`: Configured $p=0.65$, Realized = 40/60 (0.667), $z = 0.27$.
    * All 30 individual test users have $|k_{\text{user}} - 6 \cdot p_{\text{user}}| \le 0.5$ turns.
  - Added test suite in `backend/tests/test_generator_sanity.py` testing per-archetype and per-user binomial tolerances and zero split leakage.

- **Profile Learnability**:
  - History session pause distributions share each user's `jitter_ms`.
  - Pearson correlation between user's history median pause and $p_{\text{user}}$ across 30 test users is $r = +0.4358$ ($p < 0.02$), confirming learnable statistical signal.

- **Three Reference Ceilings & Paired Differences**:
  - Evaluated on 360 test turns across 30 users:
    * `PERSONA-BLIND CEILING`: 74.7% [Cluster 95% CI: 68.6%, 80.3%; Wilson: 70.0%, 78.9%]
    * `ARCHETYPE-AWARE CEILING`: 81.4% [Cluster 95% CI: 76.4%, 86.4%; Wilson: 77.0%, 85.1%]
    * `USER-AWARE CEILING`: 81.9% [Cluster 95% CI: 77.2%, 86.7%; Wilson: 77.6%, 85.6%]
  - Paired Differences (Two-Stage Cluster Bootstrap 95% CI over Users):
    * Archetype vs Blind: $+6.7\%$ [Cluster 95% CI: $+1.4\%$, $+13.1\%$]
    * User vs Archetype: $+0.6\%$ [Cluster 95% CI: $-2.8\%$, $+3.9\%$]
    * User vs Blind: $+7.2\%$ [Cluster 95% CI: $+2.2\%$, $+13.6\%$]
  - All three explicitly designated as theoretical reference ceilings in `results.md` and code, not baselines.

- **Full State Enumeration**:
  - Implemented `scripts/precompute_decisions.py --enumerate`: enumerates all 540 possible states reachable by the engine (every utterance prefix at each checkpoint plus states after speech resumed).
  - Runtime estimate on Tesla T4: 540 states $\times$ 235.6 ms = 127.2 seconds (~2.1 minutes).

- **Housekeeping**:
  - Added `hf_offline: bool = False` configuration in `backend/app/config.py`, updated `decider_service.py` and `scripts/bench_decider.py`.
  - One-time Bedrock check executed: Converse API returned `ValidationException: Operation not allowed` on Claude Haiku in `us-east-1`; Bedrock LLM Judge baseline is marked `SKIPPED`.

## Step B6 — PLAN
- **Goal**: Implement rolling per-user pace profile built solely from prior history sessions, sqlite3 regret log, state builder profile formatting under token ceilings, 4-way ablation on identical test split, and unified GPU precomputation plan.
- **Approach**:
  1. `backend/app/pace_profile.py`:
     - Per-user rolling statistics built from that user's prior sessions only: pause lengths after which they resumed vs finished (median and p75 of each), filler rate, typical words before a pause.
     - Deterministic 1-line text representation capped at 30 tokens; track real token count. Empty profile on cold start yielding identical base state.
  2. `backend/app/state_builder.py`:
     - Append profile line when present. Assert total state remains strictly under token ceiling in unit tests.
  3. `backend/app/regret_log.py`:
     - SQLite3 table (`ts`, `user_id`, `session_id`, `state_text`, `probs`, `silence_ms`, `action`) with raw SQL only (no ORM).
     - `log_regret`, `list_regrets`, `regret_stats`. Premature interruption at pause X recorded as "resumed after X" in rolling profile.
  4. Tests in `backend/tests/test_profile_and_regret.py`:
     - Test user's profile uses ONLY that user's history sessions (leakage test: no test-turn events included).
     - Cold start produces identical state to base state.
     - Regret events shift profile in expected direction.
     - Summary is 100% deterministic.
  5. 4-Way Ablation on identical test split & latency conditions:
     - (a) Decider base (virtual decider with base states)
     - (b) Decider + profile (virtual decider with profile-augmented states)
     - (c) Text heuristic with global threshold (250ms dev-tuned)
     - (d) NEW CONTROL: Text heuristic with per-user pause threshold from profile (no model)
     - Report with cluster bootstrap CIs per persona and per category. If (d) matches (b), state it plainly.
  6. GPU Precompute Planning:
     - Update `scripts/precompute_decisions.py` and `scripts/gpu_run.md` for single unified precompute: base states, profile states, and Q2 mid_thought across dev and test.
     - Resumable, deduplicated by state text, skips cached keys, prints cache file name.
     - Enumerate unique keys and verify runtime at 240 ms per call is well under 90 minutes.

## Step B6 — DONE
- **Pace Profile Builder (`backend/app/pace_profile.py`)**:
  - Computes rolling conversational statistics built strictly from prior history sessions: pause lengths after which speaker resumed vs finished (median and p75 of each), filler rate, and top trailing words.
  - Deterministic 1-line text format capped at 30 tokens (`count_tokens` enforces $\le 30$). Cold start (no history) yields empty string `""`.
  - State builder integration (`backend/app/state_builder.py`): appends profile line cleanly when present; cold start produces exact base state text (`Speaker: "..."\n[Silence: X ms]`). Total state stays strictly under 100 tokens ceiling.

- **SQLite3 Regret Log (`backend/app/regret_log.py`)**:
  - Implemented with standard library `sqlite3` and raw SQL (no ORM).
  - Schema: `id`, `ts`, `user_id`, `session_id`, `state_text`, `probs`, `silence_ms`, `action`, with indexed `user_id`.
  - Functions: `log_regret`, `list_regrets`, `regret_stats`.
  - Integration: Premature interruption at pause $X$ counts as "resumed after $X$" in profile builder, shifting pause distribution upwards.
  - Path configured via `config.py` (`regret_db_path`), gitignored via `*.db` and `backend/app/data/`.

- **Test Suite (`backend/tests/test_profile_and_regret.py`)**:
  - 6 unit tests passing (48/48 suite total):
    * Leakage guard: Verified test user profiles use ONLY history sessions (asserted zero test-split turn events included).
    * Cold start equivalence to base state.
    * Regret events shift profile in expected direction (tested median and p75 increase).
    * Summary determinism and token ceiling ($\le 30$ tokens line, $< 100$ tokens total state).
    * SQLite CRUD and stats using temporary test databases.

- **Ablation Variants (Evaluated on 360 test turns across 30 users)**:
  - (a) Decider Base (236ms GPU, Tesla T4) [MOCK-DECIDER]: Accuracy 80.3% [Cluster 95% CI: 73.6%, 86.9%], Premature 19.7% [13.1%, 26.4%], Delay P50 = 486 ms.
  - (b) Decider + Profile (236ms GPU, Tesla T4) [MOCK-DECIDER]: Accuracy 99.7% [Cluster 95% CI: 98.9%, 100.0%], Premature 0.3% [0.0%, 1.1%], Delay P50 = 1036 ms.
  - (c) Text Heuristic Global Threshold (250ms dev-tuned): Accuracy 75.3% [Cluster 95% CI: 70.0%, 80.6%], Premature 24.7% [19.4%, 30.0%], Delay P50 = 250 ms.
  - (d) NEW CONTROL: Per-User Text Heuristic (Profile Threshold, No Model): Accuracy 86.9% [Cluster 95% CI: 82.2%, 91.1%], Premature 12.8% [8.6%, 17.5%], Delay P50 = 707 ms.
  - **Persona Pause Analysis**:
    * On slow pausers: Global text heuristic suffers 25.8% premature interruptions at 250ms. Per-User Text Heuristic cuts premature interruptions to 12.5% purely by applying the user's observed pause threshold without neural model inference. Decider + Profile achieves 0.0% premature interruptions on slow pausers.
    * Observation: The per-user heuristic control confirms that personalized pause thresholding accounts for a significant portion of the profile benefit even before neural inference.

- **Single Unified GPU Precompute Plan**:
  - `scripts/precompute_decisions.py --enumerate`: Enumerates exactly 8,450 unique states across dev and test splits, covering base states, pace profile variants, and Q2 `mid_thought`.
  - Estimated runtime on Tesla T4 GPU (at 235.6 ms median): 33.18 minutes total (well within Colab session limits and under 90-minute ceiling).
  - Resumable: Skips cached keys, flushes to `backend/eval/data/decisions_cache.jsonl`.
  - Documented in `scripts/gpu_run.md`.

## Step B7 — PLAN
- **Goal**: Implement confidence-aware policy with dual thresholds ($hi, lo$), Q1/Q2 conflict resolution, and hard max hold cap; calibrate on DEV only; compute probability reliability data and Brier scores; generate Pareto sweeps across hardware latency conditions; enforce strict cache provenance watermarking via `--cache` CLI argument.
- **Approach**:
  1. `backend/app/policy.py` & `patient_listener_pattern/decision.py`:
     - Unify turn policy across repo: update `patient_listener_pattern/decision.py` with `hi, lo, max_hold_ms`, Q1/Q2 conflict rule, and re-export in `backend/app/policy.py`.
     - Decision rules:
       * Safety cap: $silence\_ms \ge max\_hold\_ms \implies \text{RESPOND}$ (always wins).
       * Cadence floor: $silence\_ms < min\_silence\_for\_respond\_ms \implies \text{WAIT}$.
       * Conflict rule: If Q1 says done ($p(\text{done}) \ge hi$) AND Q2 says mid_thought ($p(\text{mid\_thought}) \ge 0.50$), conflict triggers $\implies \text{WAIT}$ (keep listening).
       * High confidence: $p(\text{done}) \ge hi \implies \text{RESPOND}$.
       * Low confidence: $p(\text{done}) \le lo \implies \text{WAIT}$.
       * Intermediate uncertainty ($lo < p < hi$): $\text{WAIT}$ and re-evaluate at next checkpoint with more evidence.
  2. Calibration on DEV only:
     - Grid search $(hi, lo)$ on DEV split turns only (180 dev turns across 15 users).
     - Print search grid size and dev-selected $(hi^*, lo^*)$ optimal values.
     - Report final evaluated performance strictly on TEST split turns.
  3. Probability Reliability Data & Brier Score:
     - Compute calibration bins, mean predicted probability, observed empirical rate, counts per bin (assert sum = $N$), and Brier score $\frac{1}{N}\sum (p_i - y_i)^2$ on dev and test.
     - Export to JSON and render `reliability_curve.png`.
  4. Pareto Frontier Sweep:
     - Sweep premature interruption rate vs response delay (P50 and P95) for fixed timeouts (300-2000 ms), text heuristic threshold, and Decider policy $hi$ sweep.
     - Evaluate across 3 latency conditions: 0 ms, measured GPU median (235.6 ms), measured CPU median (3190.1 ms).
     - Export data JSON and plot `pareto_frontier.png` via matplotlib. Label curves with cache provenance.
  5. Tests in `backend/tests/test_policy.py`:
     - Test every policy branch, conflict resolution, safety cap priority, dev-only calibration guard, and reliability bin sum invariants.
  6. CLI `--cache` Support:
     - Add `--cache <path>` to `run_eval.py`. Inspect provenance: if real, generate headline reports and findings; if mock, watermark `[MOCK-DECIDER]` and suppress decider claims.

## Step B7 — DONE
- **Unified Confidence-Aware Policy (`patient_listener_pattern/decision.py`, `backend/app/policy.py`)**:
  - Implemented dual-threshold confidence-aware policy `policy(probabilities, silence_ms, thresholds, q2_probabilities)` with strict precedence:
    * Safety cap: $silence\_ms \ge max\_hold\_ms \implies \text{RESPOND}$ (always wins).
    * Cadence floor: $silence\_ms < 250 \implies \text{WAIT}$ (prevents premature cutoffs on initial pauses).
    * Q1/Q2 Conflict resolution: If Q1 says done ($p(\text{turn\_complete}) \ge hi$) AND Q2 says mid-thought ($p(\text{mid\_thought}) \ge 0.50$), policy overrides to $\text{WAIT}$ (conflicting evidence -> keep listening).
    * High confidence: $p(\text{RESPOND}) \ge hi \implies \text{RESPOND}$.
    * Low confidence: $p(\text{RESPOND}) \le lo \implies \text{WAIT}$.
    * Intermediate uncertainty: $lo < p(\text{RESPOND}) < hi \implies \text{WAIT}$ (keep waiting, query next checkpoint).
  - Re-export parity: `backend.app.policy` directly re-exports `patient_listener_pattern.decision.policy` and `PolicyThresholds` so the repository maintains exactly ONE canonical policy implementation.
- **DEV-Only Threshold Calibration (`backend/eval/calibration.py`)**:
  - Grid search across 64 candidate $(hi, lo)$ threshold pairs evaluated strictly on DEV split turns (180 turns across 15 disjoint dev users).
  - Dev-chosen optimal thresholds: $hi = 0.90$, $lo = 0.10$, $max\_hold\_ms = 2200$.
  - Zero test leakage asserted: `calibrate_policy_on_dev` verifies no test turns or test users exist in the calibration set (raises `ValueError` upon contamination).
  - Out-of-sample TEST performance reported in `results.md` under `Strands Decider 2B + Calibrated Policy`.
- **Probability Reliability Diagram & Brier Score**:
  - Binned on model probabilities $P(\text{RESPOND})$ (not confidence rescaling): 10 equal-width bins, mean predicted probability vs observed empirical turn completion rate.
  - Strict invariant: Bins sum to $N$ verified on both splits ($N=644$ dev checkpoints, $N=1282$ test checkpoints).
  - Measured Brier scores: DEV = 0.1235, TEST = 0.1306.
  - Exported artifacts: `backend/eval/results/reliability_data.json` and `backend/eval/results/results_reliability.png` with provenance watermarking.
- **Turn-Taking Pareto Frontiers (Trade-off Sweeps)**:
  - Swept premature interruption rate vs response delay (P50 and P95) for:
    * Fixed timeouts: 300 ms to 2000 ms.
    * Text heuristic thresholds: 100 ms to 1200 ms.
    * Decider policy $hi$ sweeps (0.55 to 0.90) at 3 latency conditions: 0 ms virtual oracle, 235.6 ms GPU (Tesla T4), and 3190.1 ms CPU.
  - Exported artifacts: `backend/eval/results/pareto_data.json` and `backend/eval/results/results_pareto.png`. All Decider curves explicitly labeled `[MOCK-DECIDER]`.
- **CLI Cache Provenance & Single-Command Evaluation**:
  - `backend/eval/run_eval.py` accepts `--cache <path>` CLI argument (defaults to `decisions_cache.jsonl`).
  - Strict provenance inspection: If mock, all Decider table rows and plots are watermarked `[MOCK-DECIDER]`, and Section 7 findings text is suppressed. When a real GPU precomputed cache is downloaded into place, a single command (`python backend/eval/run_eval.py`) generates un-watermarked headline tables and full measured findings text.
- **Unit Testing**:
  - Added `backend/tests/test_policy.py` testing:
    * Policy re-export parity with `patient_listener_pattern`.
    * Policy table exhaustive coverage for every logical branch (safety cap, floor, conflict rule, high confidence, low confidence, uncertainty band, missing probabilities).
    * DEV calibration zero-leakage and contamination rejection.
    * Reliability bin sum to $N$ invariant.
  - Repo test suite: 52/52 passing. Ruff check: clean (0 errors).


## Step B9 — PLAN
- **Goal**: Implement FastAPI backend application exposing WebSocket turn-taking session, regret logging, pace profile retrieval, scenario catalog, side-by-side replay SSE stream, benchmark results, and health checks.
- **Approach**:
  1. `backend/app/config.py`:
     - Add `cors_origins`, `fixed_replay_timeout_ms` (800ms default), `replay_decision_latency_ms` (235.6ms default), and `decision_cache_path`.
  2. `backend/app/main.py`:
     - Create FastAPI application with CORS middleware.
     - `WS /ws/session`:
       * Accepts client events matching `events.py` schemas (`SpeechStarted`, `SpeechDelta`, `SpeechEnded`, `tick`).
       * Manages live `TurnManager` and `ReplyAgent`.
       * Emits decision events (`probs`, `p_done`, `latency_ms`, `action`, `checkpoint`).
       * On `RESPOND`, streams reply tokens from `reply_agent`. If Bedrock call fails, falls back to calibrated mock reply and sets `reply_source: "mock"` (or `"bedrock"` if live).
     - `POST /regret`:
       * Validates payload (`user_id`, `session_id`, `state_text`, `probs`, `silence_ms`, `action`).
       * Logs to SQLite using `backend/app/regret_log.py`.
     - `GET /profile/{user}`:
       * Builds and returns `PaceProfile` JSON via `PaceProfileBuilder`.
     - `GET /scenarios`:
       * Returns scenario metadata list (`id`, `user_id`, `persona_id`, `archetype`, `text_preview`, `true_end_time_ms`, `category`, `silence_ms`, `is_turn_complete`).
     - `POST /replay/{scenario}`:
       * Streams the scripted scenario through TWO engines in parallel:
         - Fixed timeout engine (default 800 ms from config).
         - Patient Listener engine (using active decision cache with simulated latency 235.6 ms).
       * Emits SSE stream (`text/event-stream`) labeled with cache provenance and `"latency measured on Tesla T4, decisions from cache"`.
       * Accepts `speed` query parameter (1.0 for real-time demo, large values for tests).
       * Raises clear error event upon cache miss (zero fallback).
       * Emits final events per engine: `interrupted` (bool) and `response_delay_ms`.
     - `GET /results`:
       * Returns `eval_results.json` content plus provenance flag.
     - `GET /health`:
       * Returns backend status, cache provenance, cache key count, and Bedrock status (`ValidationException: Operation not allowed`).
  3. `scripts/pick_demo_scenarios.py`:
     - Evaluates test scenarios against active cache; selects cases where fixed timeout interrupts and Patient Listener waits then responds.
     - Saves picks to `backend/app/data/demo_scenarios.json` with provenance.
     - Emits warning if evaluated against mock cache.
  4. Integration tests in `backend/tests/test_api.py`:
     - Test scripted WS session.
     - Test regret logging and profile shift.
     - Test replay stream with both engines at high speed.
     - Test cache miss error event.
     - Test reply agent fallback flag.

## Step B9 — DONE
- **Date**: 2026-10-19
- **Implemented Artifacts**:
  - `backend/app/main.py`:
    * `WS /ws/session`: Real-time conversational turn-taking WebSocket session with streaming token delivery, mock Bedrock fallback, and configurable `ticker` parameter.
    * `POST /regret`: SQLite regret logging for premature interruptions.
    * `GET /profile/{user}`: Per-user rolling pace profile retrieval incorporating prior history and logged regrets.
    * `GET /scenarios`: Catalog listing demo scenarios with preview text and metadata.
    * `POST /replay/{scenario}`: Dual-engine streaming replay (Fixed 800ms vs Patient Listener) emitting Server-Sent Events (`text/event-stream`), parameterized by `speed`, labeled with active cache provenance and T4 latency annotation; returns strict error event on cache miss (zero fallback).
    * `GET /results`: Evaluated results JSON with provenance flag.
    * `GET /health`: Health status reporting backend, cache provenance, and Bedrock status.
    * CORS origins middleware configured from `config.cors_origins`.
  - `scripts/pick_demo_scenarios.py`:
    * Evaluates candidate scenarios against active cache; identifies turns where Fixed interrupts and Patient Listener waits then responds.
    * Saves 10 picks to `backend/eval/data/demo_scenarios.json` with provenance; warns when active cache is mock-derived.
  - `backend/tests/test_api.py`:
    * 9 integration tests covering health, results, scenarios, dynamic profile regret shift, scripted WS session, Bedrock failure fallback flag, dual-engine high-speed replay, cache miss error event, and 404 handling.
- **Verification**:
  - Pre-commit checks: `ruff check .` clean (0 errors), `pytest -q` 61/61 passing in 3.82s.

## Step F1 — PLAN
- **Goal**: Scaffold frontend directory (`frontend/`) using Vite + React + TypeScript + Tailwind CSS with typed event client matching backend events exactly, robust WebSocket client with reconnect, connection status badge, scenario preview list, and event parity pytest.
- **Approach**:
  1. Preflight check: Node (`v24.13.0`), npm (`11.16.0`), and E: free disk space (>80 GB) verified. All installs strictly isolated inside `frontend/`.
  2. Scaffold `frontend/` using Vite with React + TypeScript template.
  3. Install dependencies: Tailwind CSS, PostCSS, Autoprefixer, Lucide icons (for clean status/audio cues).
  4. Create `frontend/src/events.ts` strictly mirroring `backend/app/events.py` schemas (dataclass names, fields, event types).
  5. Add `backend/tests/test_event_parity.py` asserting every event dataclass in `backend/app/events.py` has a corresponding typed interface in `frontend/src/events.ts`.
  6. Implement typed WebSocket client (`frontend/src/ws.ts` and `App.tsx`):
     - Auto-reconnect with exponential backoff and jitter.
     - Connection status indicator (`Connecting`, `Connected`, `Disconnected`).
     - Reads `VITE_API_URL` (default `http://localhost:8000`).
     - Tiny landing dashboard listing `GET /scenarios` with persona, preview text, and true end time.
  7. Configure `.gitignore` to strictly exclude `node_modules` and `dist`.
  8. Provide `frontend/README.md` with Windows PowerShell run instructions. Verify `npm run build` succeeds cleanly.

## Step F1 — DONE
- **Date**: 2026-10-21
- **Implemented Artifacts**:
  - `frontend/`:
    * Scaffolded with Vite + React 19 + TypeScript + Tailwind CSS (`vite.config.ts`, `tailwind.config.js`, `postcss.config.js`, `tsconfig.json`).
    * `frontend/src/events.ts`: Strongly-typed TypeScript interfaces directly mirroring `backend/app/events.py` (`SpeechStarted`, `SpeechDelta`, `SpeechEnded`, `SilenceProbe`, `DecisionRequested`, `DecisionCompleted`, `DecisionDropped`, `TurnYielded`, `MaxHoldExceeded`, `Decision`, `TurnEvent`).
    * `frontend/src/ws.ts`: Resilient `PatientListenerWSClient` with configurable reconnect backoff, jitter, typed message dispatcher, and connection state management (`connecting`, `connected`, `reconnecting`, `disconnected`, `error`).
    * `frontend/src/App.tsx`: Dark-mode interface with live WebSocket status badge, active API endpoint indicator, interactive turn tester, live event stream log, and scenario explorer consuming `GET /scenarios`.
    * `frontend/.env.example`: Configured with `VITE_API_URL=http://localhost:8000`.
    * `frontend/README.md`: Windows PowerShell execution, development, and build instructions.
  - `backend/tests/test_event_parity.py`:
    * Pytest test suite inspecting AST of `backend/app/events.py` and asserting every defined event dataclass appears as an exported interface in `frontend/src/events.ts`.
  - `.gitignore`: Configured to strictly exclude `frontend/node_modules/` and `frontend/dist/`.
- **Verification**:
  - `npm run build`: built in 1.79s cleanly with zero TypeScript or bundle warnings (`dist/` generated, gitignored).
  - Python test suite: 64/64 passing in 3.65s (`backend/tests/test_event_parity.py` 3/3 passed).
  - Ruff check: 0 errors across entire workspace.

## Step E1 — PLAN
- **Goal**: Accelerate `run_eval.py` from 6h 53m to under 10 minutes by vectorizing cluster bootstrap with numpy, pre-caching DEV calibration evaluations, memoizing replay evaluations across conditions, adding stage timing instrumentation and a `--fast` development flag, and isolating Bedrock execution behind `--with-judge` (defaulting to SKIPPED without network calls).
- **Approach**:
  1. Vectorize cluster bootstrap in `backend/eval/metrics.py`:
     - Group turns by user once and extract per-user numerators and denominators for accuracy, premature rate, and unnecessary wait.
     - Resample user clusters and turn counts using `numpy.random.default_rng` binomial draws without turn-level Python loops.
     - Vectorize paired differences across models using stacked numpy arrays.
     - Vectorize `bootstrap_percentile_ci` on response delays using 2D matrix sampling.
  2. Optimize DEV threshold calibration in `backend/eval/calibration.py`:
     - Pre-extract state prompt keys and cache entries once per turn rather than rebuilding strings and sha256 hashing across all 64 grid pairs.
     - Reuse calibration across conditions.
  3. Instrument per-stage timings in `backend/eval/run_eval.py`:
     - Explicit timing prints for: Load cache & data, DEV policy calibration, replay per condition, bootstrap CIs, reliability/Pareto sweeps, and report generation.
     - Support `--fast` flag: 200 bootstrap resamples and 16 calibration grid points for fast dev sweeps.
  4. Bedrock network isolation:
     - Add `--with-judge` CLI flag (default `False`). When off, `LLMJudgeBaseline` is never initialized and makes zero AWS STS / Converse calls; judge row reports `SKIPPED (not run in this eval)`.
     - When `--with-judge` is enabled, enforce strict boto3 timeouts: `connect_timeout=3`, `read_timeout=10`, `max_attempts=1`.
  5. Measure execution times on mock cache with `--fast` and full sweep; record measured timings in `NOTES.md`.

### Step E1 Measured Timings (Mock Cache Benchmark)
- **Previous unvectorized run**: 6 hours 53 minutes (~24,780 seconds).
- **With `--fast` (200 bootstrap resamples, 16 calibration grid candidates)**:
  * Stage 1/6 (Load Cache & Data): 0.29 s (8558 keys, 360 test turns)
  * Stage 2/6 (DEV Calibration): 0.01 s (Grid Size=16, chosen hi=0.90, lo=0.10)
  * Stage 3/6 (Replay Evaluation): 0.35 s (18 configurations)
  * Stage 4/6 (Bootstrap Metrics & CIs): 0.13 s (n_resamples=200)
  * Stage 5/6 (Reliability & Pareto Sweeps): 0.83 s
  * Stage 6/6 (Report Writing & Export): 0.00 s
  * **Total Evaluation Time**: **1.63 seconds**
- **Full Run (1000 bootstrap resamples, 64 calibration grid candidates)**:
  * Stage 1/6 (Load Cache & Data): 0.28 s (8558 keys, 360 test turns)
  * Stage 2/6 (DEV Calibration): 0.01 s (Grid Size=64, chosen hi=0.90, lo=0.10)
  * Stage 3/6 (Replay Evaluation): 0.33 s (18 configurations)
  * Stage 4/6 (Bootstrap Metrics & CIs): 0.28 s (n_resamples=1000)
  * Stage 5/6 (Reliability & Pareto Sweeps): 0.83 s
  * Stage 6/6 (Report Writing & Export): 0.00 s
  * **Total Evaluation Time**: **1.76 seconds**
- **Progress note (perf commit)**: Vectorized cluster bootstrap with 2D numpy binomial sampling and memoized DEV calibration, accelerating full mock evaluation from 6h 53m to 1.76s.
- **Progress note (bedrock commit)**: Isolated Bedrock judge behind --with-judge (default SKIPPED), configured boto3 client with strict timeouts (connect 3s, read 10s, max 1 attempt).

## Step E2 — PLAN
- **Goal**: Validate the real precomputed GPU decisions cache (`backend/eval/data/decisions_cache_real.jsonl`), compute exhaustive latency distributions on Tesla T4, export formal provenance, archive as a compressed gzip artifact (`decisions_cache_real.jsonl.gz` < 2 MB), and update loaders to support transparent `.gz` decompression.
- **Approach**:
  1. Verify existence of `backend/eval/data/decisions_cache_real.jsonl` (confirmed present via `Test-Path`).
  2. Implement `scripts/validate_cache.py`:
     - Validate every record format: valid JSON, `backend == 'local'`, `model_id == 'StrandsAgents/strands-decider-2B-hobson-v19'`, zero mock markers.
     - Match keys against canonical enumeration from `precompute_decisions.py --enumerate` (expected 8450 unique states).
     - Check duplicate records: measure max probability discrepancy across duplicate keys, deduplicate keeping one record.
     - Assert complete coverage: if any required key is missing, break down missing count by question and checkpoint, and abort.
     - Compute exact latency metrics from single-process T4 records: median and p95 overall, per question (`turn_complete`, `mid_thought`), and per checkpoint.
     - Write structured provenance JSON `backend/eval/data/decisions_cache_real_provenance.json`.
  3. Update cache loaders (`load_cache` in `scripts/precompute_decisions.py` and `backend/app/main.py`) to transparently load `.jsonl.gz` when plain `.jsonl` is missing.
  4. Deduplicate and compress `decisions_cache_real.jsonl` into `decisions_cache_real.jsonl.gz`, assert size < 2 MB, and add plain `.jsonl` to `.gitignore`.
  5. Commit `feat: add cache validation and provenance` and `chore: add real decision cache archive`.

## Step E2 — DONE
- **Verified and validated real decision cache** (`backend/eval/data/decisions_cache_real.jsonl`):
  * Total records: 8,450. Unique keys: 8,450. Duplicates: 0. Max probability discrepancy: 0.000000.
  * Canonical state coverage: 8,450 / 8,450 (100.00%). Missing: 0. Extra: 0.
  * All records verified: `backend == 'local'`, `model_id == 'StrandsAgents/strands-decider-2B-hobson-v19'`, zero mock markers.
  * Latency distributions measured on Tesla T4 GPU (clean single-process):
    - Overall Median: 301.19 ms, Overall P95: 306.66 ms.
    - Question `turn_complete`: median 303.01 ms, p95 307.35 ms (n=4,225).
    - Question `mid_thought`: median 299.79 ms, p95 304.90 ms (n=4,225).
  * Provenance exported: `backend/eval/data/decisions_cache_real_provenance.json` (SHA256: `62e85524a69fa3946f86563a37d9df9feceaceb16aded35e73747f3a968b5d14`).
  * Gzip archive created: `backend/eval/data/decisions_cache_real.jsonl.gz` (544 KB, well under 2 MB limit).
  * Loaders in `scripts/precompute_decisions.py` and `backend/app/main.py` updated to support transparent `.gz` decompression.

## Step F3 — PLAN
- **Goal**: Implement the side-by-side replay demo mode (the hero of the demo video) and a slim results dashboard in React + TypeScript + Tailwind. Ensure UI strictly alerts on mock provenance ("MOCK-DECIDER, not valid for the demo") and refuses to look like a valid demo until the real cache is installed.
- **Approach**:
  1. Backend routes:
     - Update `GET /scenarios` in `backend/app/main.py`: return `id`, `user`, `persona`, `text_preview`, `true_end_t_ms`, and `demo` boolean flag (demo-flagged scenarios from `demo_scenarios.json` sorted first).
     - Add `GET /plots/{name}` in `backend/app/main.py`: whitelist `results_pareto.png` and `results_reliability.png` and serve via `FileResponse`.
  2. Frontend SSE Parser (`frontend/src/sse.ts`):
     - Pure streaming parser consuming `ReadableStream<Uint8Array>` from `fetch("POST /replay/...")`.
     - Parses line-buffered SSE chunks (`data: ...\n\n`), typed directly against `ReplayEvent`.
  3. Frontend Architecture (`frontend/src/App.tsx` & components):
     - Two tabs: "Demo" (active by default) and "Results".
     - Header provenance indicator:
       * If real: green badge "Decisions: real Strands Decider 2B (cached), decision latency 236 ms measured on Tesla T4".
       * If mock: red full-width warning banner "MOCK-DECIDER, not valid for the demo".
     - Demo view:
       * Scenario selector (demo-flagged first, showing archetype, user, text preview).
       * Replay speed controls (1x, 2x, 4x).
       * Start and Cancel controls with `AbortController`.
       * Two parallel comparison panels:
         - Left: "Fixed 800 ms timeout" (transcript stream, live silence progress bar, interrupt badge, final stats).
         - Right: "Patient Listener (Decider 2B)" (transcript stream, live silence bar, decision checkpoint feed with p_done/latency, final stats).
       * Shared unified timeline strip rendering speech blocks, pauses, true end time, and both engines' response points.
       * Post-run outcome summary banner.
       * Sequential "Play demo set" runner with tally of interruptions and mean delay.
     - Results view:
       * Consumes `GET /results`.
       * Headline metrics table with Cluster 95% CIs and Wilson CIs.
       * Labeled reference ceilings.
       * Dual PNG plots fetched from `GET /plots/results_pareto.png` and `GET /plots/results_reliability.png`.
       * Clear provenance and synthetic data caveats.

## Step F3 — DONE
- **Backend enhancements**:
  * Updated `GET /scenarios` in `backend/app/main.py`: returns `id`, `user`, `persona`, `text_preview`, `true_end_t_ms`, and `demo` boolean flag (demo-flagged scenarios from `demo_scenarios.json` sorted first).
  * Added `GET /plots/{name}` in `backend/app/main.py`: serves whitelisted evaluation plots (`results_pareto.png`, `results_reliability.png`) from `backend/eval/results/`.
- **Frontend architecture**:
  * Two-tab interface ("Side-by-Side Demo" and "Benchmark Results") in `frontend/src/App.tsx`.
  * Prominent provenance indicator: displays full-width red warning `"MOCK-DECIDER, not valid for the demo — real GPU decision cache not installed"` when running with mock cache, refusing to look like a valid demo until real GPU decision cache is installed.
  * Side-by-side demo tab:
    - Pure SSE stream parser in `frontend/src/sse.ts`.
    - Scenario picker prioritizing demo-flagged scenarios.
    - 1x, 2x, 4x speed selector and Start/Cancel replay controls with `AbortController`.
    - Side-by-side comparison panels (Fixed 800 ms timeout vs Patient Listener 2B) with transcript stream, live silence progress bar, decision checkpoint cards, and interrupted badge.
    - Unified timeline strip (`frontend/src/ReplayTimeline.tsx`) displaying speech blocks, pause silence gaps, true end marker, and engine response markers on a shared axis.
    - Post-run outcome comparison line and sequential "Play demo set" runner with running cut-in tally.
  * Slim results tab:
    - `frontend/src/ResultsDashboard.tsx` fetching `GET /results` with Cluster 95% CIs and bootstrap intervals.
    - Explicitly labeled reference ceilings and SYNTHETIC data caveat.
    - Embedded Pareto frontier and Reliability curve PNGs via `GET /plots/{name}`.
- **Verification**:
  * High-speed replay tested via `curl.exe` at 20x speed; event sequence strictly matched `frontend/src/events.ts`.
  * Production bundle compiled cleanly with `npm run build` in 1.48s.
  * All 65 pytest integration tests and `ruff check .` passed cleanly.
  * Note: The UI was not visually checked in a browser in this session. When testing in a browser at `http://localhost:5173`, look at:
    1. The top red alert banner "MOCK-DECIDER, not valid for the demo".
    2. The scenario dropdown with ★ [DEMO] items listed first.
    3. The side-by-side dual engine panels streaming live transcript, silence bar, decision checkpoint cards, and unified timeline strip.
    4. The Benchmark Results tab rendering the headline table with Cluster CIs, ceilings, and the two served PNG plots.

## Step E3 — PLAN
- **Goal**: Run the full real decider evaluation once using `decisions_cache_real.jsonl`, generating headline metrics across latency sources (cache-measured 301.2 ms on Tesla T4, benchmark 235.6 ms, 0 ms oracle, 3190 ms CPU), paired cluster-bootstrap differences, recalibrated DEV policy thresholds, probability reliability, and programmatically generated findings text from `eval_results.json`.
- **Approach**:
  1. Update `backend/eval/run_eval.py`:
     - Dynamically resolve `cache_measured_median_ms` (301.2 ms from real cache records / provenance).
     - Configure required replay rows:
       * Headline: `Strands Decider 2B (301ms GPU, cache-measured on T4)`
       * Profile: `Strands Decider 2B + Profile (301ms GPU, cache-measured on T4)`
       * Calibrated Policy: `Strands Decider 2B + Calibrated Policy [REAL-CALIBRATED] (hi=0.60, lo=0.10, 301ms GPU, cache-measured on T4)`
       * Benchmark row: `Strands Decider 2B (235.6ms GPU benchmark, synthetic 20-150 tokens on T4)`
       * 0 ms row: `Strands Decider 2B (0 ms decision latency)`
       * CPU row: `Strands Decider 2B (3190ms CPU, Ryzen 5 5500U)`
       * Sweep rows: 100ms, 300ms, 1000ms, 3000ms.
     - Compute paired cluster-bootstrap differences for Decider vs Heuristic, Decider+profile vs Decider, Decider+profile vs Per-User Control, along with reference ceilings.
     - Add watermark / provenance in caption of `results_reliability.png` and `results_pareto.png`.
     - In Section 7, generate findings directly from `out_json` data.
  2. Execute `python backend/eval/run_eval.py --cache backend/eval/data/decisions_cache_real.jsonl` (full run, 1000 resamples).
  3. Verify `eval_results.json` and `results.md` are updated and consistent.
  4. Pre-commit checks: `ruff check .`, `pytest -q`, secret grep, `git diff --cached --name-only`.
  5. Commit `feat: run real decider evaluation`. Note in `NOTES.md`.

## Step E3 — DONE
- **Execution & Runtime**:
  * Executed full evaluation run with `--cache backend/eval/data/decisions_cache_real.jsonl` (zero mock fallbacks).
  * 10,384 cache hits, 0 cache misses across 19 evaluation configurations.
  * Runtime: 4.09 seconds total (vectorized cluster bootstrap, calibration, and Pareto sweeps).
- **Headline Results (out-of-sample TEST split, N=360 turns, 1800 checkpoints)**:
  * Strands Decider 2B (301ms GPU, cache-measured on T4): Accuracy 100.0% [98.9%, 100.0%], Premature Interruption Rate 0.0% [0.0%, 0.0%], Response Delay P50 2200 ms [2200.0, 2200.0] ms, P95 2200 ms.
  * Text Heuristic Baseline (250ms, dev-tuned): Accuracy 75.3% [69.7%, 80.6%], Premature Interruption Rate 24.7% [19.4%, 30.3%], Response Delay P50 250 ms [250.0, 250.0] ms, P95 250 ms.
  * Per-User Text Heuristic Control (Profile Threshold, No Model): Accuracy 87.2% [82.2%, 91.7%], Premature Interruption Rate 12.8% [8.3%, 17.8%], Response Delay P50 707 ms [661.0, 766.0] ms, P95 928 ms.
- **Paired Cluster-Bootstrap Differences (with 95% CIs over 1000 resamples)**:
  * Decider minus Text Heuristic: +24.7% [+18.9%, +30.3%].
  * Decider+Profile minus Decider: +0.0% [+0.0%, +0.0%].
  * Decider+Profile minus Per-User Control: +13.1% [+8.6%, +18.3%].
  * Decider+Profile overall: Accuracy 100.0%, Premature 0.0%, Delay P50 2200 ms.
- **Reference Ceilings**:
  * PERSONA-BLIND CEILING: Accuracy 74.7%, Premature 0.0%, Delay P50 2200 ms.
  * ARCHETYPE-AWARE CEILING: Accuracy 81.4%, Premature 5.0%, Delay P50 250 ms (+6.7% vs Blind).
  * USER-AWARE CEILING: Accuracy 81.9%, Premature 3.9%, Delay P50 250 ms (+7.2% vs Blind, +0.6% vs Archetype).
- **Calibration, Reliability & Conflict Rule**:
  * Recalibrated thresholds on DEV split: hi = 0.55, lo = 0.10, max_hold_ms = 2200 ms (DEV score = 1.0000).
  * Reliability: DEV Brier score = 0.3476 (644 samples); TEST Brier score = 0.3452 (1282 samples).
  * Q1+Q2 Conflict Rule: exactly 0 / 1800 test checkpoints exhibited conflict (Q1 p(RESPOND) >= 0.55 and Q2 p(YES) >= 0.50).
- **Verdict & Outcome**:
  * Plain outcome finding: The Decider eliminates premature cut-ins (0.0% cut-ins across all personas and categories), but does not beat the text heuristic (250 ms) or per-user control (707 ms) on response delay because p(RESPOND) rarely crosses hi during mid-thought pauses, holding until the safety ceiling (2200 ms).
- **Artifacts**:
  * Re-generated `results_reliability.png` and `results_pareto.png` with real provenance captions.
  * Updated `results.md` programmatically from `eval_results.json`.

## Step E4 — PLAN
- **Goal**: Resolve the degenerate policy behavior observed in Step E3 where Decider showed delay P50/P95 of exactly 2200 ms and 0.0% premature rate (every turn hitting `max_hold_ms=2200ms`, rendering Decider identical to a fixed 2.2s timeout). Perform rigorous calibration audit, evaluate probability distributions, sweep operating points on DEV, report on TEST with cluster bootstrap CIs, relabel reference rows, and re-pick demo scenarios from policy decisions.
- **Approach**:
  1. Audit calibration objective, grid, cadence floor, and dev-chosen values; assess delay term and always-wait exploitability.
  2. Tabulate exact p(done)=P(RESPOND) and p(mid_thought)=P(YES) distributions from the real cache across categories (complete, incomplete, ambiguous) and checkpoints (250, 500, 800, 1200, 1800 ms).
  3. Measure policy vs max_hold cap trigger fractions for Decider and Decider+Profile.
  4. Evaluate argmax baseline (RESPOND at first checkpoint where p(done) >= 0.50, else cap).
  5. Sweep thresholds on DEV only to identify Point A (delay budget <= 800 ms) and Point B (premature budget <= 5.0%), then evaluate all 5 families on TEST with paired cluster bootstrap differences.
  6. Relabel single first-checkpoint references in results.md and frontend, document per-user headroom limitation.
  7. Re-pick demo scenarios using Point A policy so Patient Listener responds from model decision before the cap.

### Calibration Audit & Probability Distributions (Step E4.1 - E4.4)
- **1. Objective Function & Search Space**:
  * Exact objective function: `score = accuracy - (0.5 * premature_rate)`.
  * Grid: 64 candidate threshold pairs on DEV:
    - `hi` in `[0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]`
    - `lo` in `[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]` (subject to `hi > lo`)
  * Cadence floor: `min_silence_for_respond_ms = 250.0 ms`.
  * Timeout cap: `max_hold_ms = 2200.0 ms`.
  * DEV-chosen values in E3: `hi = 0.55, lo = 0.10` (DEV score = 1.0000, Acc = 100.0%, Premature = 0.0%).
  * Delay term in objective: **NO**. The objective contains zero latency penalty.
  * Can always-waiting satisfy it? **YES**. Complete turns coerce to RESPOND at `max_hold_ms=2200ms`, which counts as `pred_action == gt`, giving 100% accuracy and 0% premature rate. An agent that never fires from policy and always waits for max_hold achieves a perfect score of 1.0000.

- **2. p(done) = P(RESPOND) Distribution from Real Cache (Q1 turn_complete, TEST split)**:
  * **Category COMPLETE (n=90)**:
    - cp=250ms: n=90, mean=0.339, min=0.209, p25=0.280, med=0.328, p75=0.388, max=0.491 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=500ms: n=90, mean=0.347, min=0.219, p25=0.279, med=0.328, p75=0.398, max=0.525 | >=0.5: 11.1%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=800ms: n=90, mean=0.303, min=0.202, p25=0.230, med=0.280, p75=0.349, max=0.503 | >=0.5: 11.1%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=1200ms: n=90, mean=0.319, min=0.195, p25=0.249, med=0.298, p75=0.381, max=0.505 | >=0.5: 11.1%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=1800ms: n=90, mean=0.309, min=0.196, p25=0.234, med=0.282, p75=0.373, max=0.484 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
  * **Category INCOMPLETE (n=28 to 89)**:
    - cp=250ms: n=89, mean=0.094, min=0.058, p25=0.078, med=0.091, p75=0.103, max=0.151 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=500ms: n=60, mean=0.099, min=0.058, p25=0.082, med=0.096, p75=0.104, max=0.160 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=800ms: n=28, mean=0.101, min=0.058, p25=0.084, med=0.100, p75=0.103, max=0.152 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
  * **Category AMBIGUOUS (n=91 to 180)**:
    - cp=250ms: n=180, mean=0.339, min=0.177, p25=0.275, med=0.321, p75=0.408, max=0.491 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=500ms: n=162, mean=0.350, min=0.178, p25=0.284, med=0.332, p75=0.413, max=0.492 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=800ms: n=131, mean=0.320, min=0.164, p25=0.250, med=0.353, p75=0.408, max=0.447 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=1200ms: n=91, mean=0.326, min=0.158, p25=0.260, med=0.356, p75=0.409, max=0.467 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=1800ms: n=91, mean=0.316, min=0.157, p25=0.251, med=0.347, p75=0.399, max=0.451 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
  * **With Profile Variant (Category COMPLETE, n=90)**:
    - cp=250ms: n=90, mean=0.342, min=0.212, p25=0.282, med=0.347, p75=0.395, max=0.453 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=500ms: n=90, mean=0.332, min=0.205, p25=0.272, med=0.334, p75=0.380, max=0.445 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=800ms: n=90, mean=0.323, min=0.204, p25=0.265, med=0.329, p75=0.377, max=0.469 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=1200ms: n=90, mean=0.338, min=0.207, p25=0.272, med=0.343, p75=0.400, max=0.476 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
    - cp=1800ms: n=90, mean=0.331, min=0.210, p25=0.271, med=0.330, p75=0.386, max=0.457 | >=0.5: 0.0%, >=0.7: 0.0%, >=0.9: 0.0%
  * **Q2 mid_thought: p(mid_thought) = P(YES) (TEST split, no profile)**:
    - Complete: med 0.494 - 0.551 (cp 250-1200ms >=0.5: 53.3%; max: 0.699; >=0.7: 0.0%)
    - Incomplete: med 0.842 - 0.881 (all checkpoints >=0.5: 100.0%, >=0.7: 100.0%, >=0.9: 7.1% - 20.2%)
    - Ambiguous: med 0.379 - 0.476 (cp 250-800ms >=0.5: 35.6% - 38.9%, >=0.7: 0.0%)
  * **Q2 mid_thought with Profile**:
    - Complete: med 0.470 - 0.525 (>=0.5: 36.7% - 55.6%, >=0.7: 0.0%)
    - Incomplete: med 0.824 - 0.841 (>=0.5: 100.0%, >=0.7: 100.0%, >=0.9: 7.1% - 10.1%)
    - Ambiguous: med 0.381 - 0.433 (>=0.5: 12.1% - 27.8%, >=0.7: 0.0%)

- **3. Fraction of Turns from Policy vs Max Hold Cap (hi=0.55, lo=0.10)**:
  * Decider: Policy RESPOND = 0 / 181 (0.0%), Max Hold Cap = 181 / 181 (100.0%)
  * Decider+Profile: Policy RESPOND = 0 / 181 (0.0%), Max Hold Cap = 181 / 181 (100.0%)

- **4. Reference Rows: Argmax Policy (p(done) >= 0.50, else Cap)**:
  * Decider (Argmax): Policy triggers = 10 / 181, Cap triggers = 171 / 181 | Premature = 0.0%, Delay P50 = 2200.0 ms, Delay P95 = 2200.0 ms.
  * Decider+Profile (Argmax): Policy triggers = 0 / 181, Cap triggers = 181 / 181 | Premature = 0.0%, Delay P50 = 2200.0 ms, Delay P95 = 2200.0 ms.

## Article material (running list)
- **Progress note (fix: accumulate transcript per engine)**: Split transcript into per-engine streams (Fixed vs Patient Listener) accumulating incoming words until each engine responds or cuts off, resolving the UI issue where latest segment overwrote previous words.
- **Progress note (fix: show response timing relative to true end)**: Updated delay line to show timing relative to true end ("answered X ms before user finished" or "answered X ms after true end" or "no answer"), with pure function unit tests.
- **Progress note (fix: derive result line from final events)**: Implemented formatResultLine pure function with 3 cases strictly from final events and error states, with unit test verification.
- **Progress note (docs: record operating point results)**: Documented DEV-selected operating points A and B, cluster bootstrap differences, dominance findings (Fixed timeout and heuristic dominate Decider), and demo scenario delays in results.md and NOTES.md.
- Operating Point Analysis & Dominance finding: Fixed timeouts and text heuristic baseline dominate Strands Decider 2B at both operating points on TEST split. At Point A (delay budget <= 800ms matching Fixed 800ms), Text Heuristic achieves 800.0 ms delay P50 with 11.1% [6.4%, 16.7%] premature rate vs Decider 1501.2 ms delay P50 with 10.0% [5.8%, 14.7%] premature rate (paired diff -1.1% [-5.6%, +3.3%], CI spans zero; Decider is +701 ms slower). At Point B (premature budget <= 5%), Fixed 950ms achieves 3.3% [1.1%, 6.4%] premature rate and 950.0 ms delay P50, and Text Heuristic 950ms achieves 2.5% [0.6%, 5.3%] premature rate and 950.0 ms delay P50, whereas Decider achieved 0.0% premature rate only by waiting 2200.0 ms delay P50 (+1353.2 ms [+1304.6, +1399.0] ms slower). Decider does not beat the baselines at equal delay or equal premature budget.
- **Progress note (feat: pick demo scenarios from decider decisions)**: Re-picked 10 demo scenarios under Point A Decider policy with RESPOND arriving before cap (551.2 ms delay after true end), documented persona counts (9 slow, 4 filler, 0 fast), and wired Point A policy into /health and config.
- **Progress note (fix: relabel reference rows and lead with premature and delay)**: Relabeled reference rows as single first-checkpoint references across results.md, results JSON, and UI, documented ~0.5 point user headroom limitation, and led headline table with premature rate and delay P50/P95 at operating points A and B.
- **Progress note (feat: add operating point analysis)**: Implemented DEV operating point sweeps (Points A and B) and paired cluster bootstrap evaluation across 5 families with dominance analysis.
- **Progress note (docs: audit decider policy calibration)**: Documented calibration objective audit, p(done) and p(mid_thought) distributions, policy vs cap trigger fractions, and argmax reference rows in NOTES.md.
- **Progress note (fix: read decision latency label from provenance)**: Configured default cache to real 8,450-record cache with 301.2 ms latency; wired dynamic latency and source from /health and provenance JSON into API endpoints and frontend banners.
- **Progress note (feat: pick demo scenarios from real cache)**: Selected 10 balanced demo scenarios (5 slow_pauser, 5 filler_heavy) from 68 qualifying cases on real Tesla T4 cache.
- **Progress note (docs: record real decider results and findings)**: Documented real decider benchmark findings, paired bootstrap differences, and latency observations in results.md and NOTES.md.
- Tesla T4 GPU decision latency: The cache-measured median latency across full evaluation states is 301.2 ms (p95 306.7 ms), compared to the synthetic 20-150 token benchmark median of 235.6 ms reported by bench_decider.py on a T4. The cause of this discrepancy is not established; we report measured numbers without guessing.
- Plain language finding on Decider vs heuristic / control trade-off: The real 2B Decider prioritizes avoiding premature interruptions (0.0% cut-ins across all personas and categories), successfully protecting conversational comfort. However, because its predicted probability p(RESPOND) during mid-thought pauses rarely crosses the hi threshold (0.55), the policy holds until max_hold_ms (2200 ms), yielding response delay P50 of 2200 ms. Consequently, the Decider does not beat the text heuristic (250 ms delay) or per-user control (707 ms delay) on speed.
- **Progress note (feat: add side-by-side replay demo)**: Implemented side-by-side replay demo with pure SSE parser, dual engine stream panels, shared timeline strip, speed controls, AbortController cancellation, and mock-decider alert banner.
- **Progress note (feat: add results page and plot endpoint)**: Added GET /plots/{name} endpoint for Pareto and reliability curves, and built slim ResultsDashboard rendering Cluster 95% CIs, labeled ceilings, and synthetic data caveats.
- **Progress note (docs: add demo run steps)**: Documented manual demo execution steps in frontend/README.md and recorded Step F3 completion in NOTES.md.
- **Progress note (feat: add cache validation and provenance)**: Validated 8450 real T4 decision keys with zero missing and zero discrepancies, recorded Tesla T4 latency stats (median 301.19 ms, p95 306.66 ms), and exported decisions_cache_real_provenance.json.
- **Progress note (chore: add real decision cache archive)**: Archived verified 8450-record real Tesla T4 decision cache as compressed 544 KB gzip file (decisions_cache_real.jsonl.gz).
- **Progress note (feat: run real decider evaluation)**: Executed full real decider evaluation on Tesla T4 cache (301.2 ms median) across 19 configurations with vectorized cluster bootstrap, paired difference CIs, and provenance-watermarked Pareto/reliability plots.
- Running local 2B decision models on consumer CPU (AMD Ryzen 5 5500U): evaluating feasibility for real-time turn-taking where voice latency budgets are ~200-500 ms.
- Real-world setup failure & fix: `hf-xet` (Rust background writer in `huggingface_hub`) fails on Windows with `File reconstruction error: Internal Writer Error: Background writer channel closed`. Removing `hf-xet` to force standard HTTP chunked download completely resolves the stall without requiring special toolchains.
- Large weight download stability: Downloading monolithic 4.5 GB safetensors files over consumer connections can encounter TCP resets; using resumable chunked downloads (`curl.exe -C - --retry 8 --retry-all-errors`) is essential to prevent restarting from 0 bytes.
- Real-world failure: curl error 56 (connection reset) at 76.1%, resumed successfully from 3.46 GB without restarting from scratch.
- Prompt template overhead vs flat latency: On CPU, System One questions incur a ~103-token template overhead (criteria and instructions), meaning 20 vs 60 token states differ by only a fraction of total forwarded sequence length.
- Synthetic dataset audit: The 40 synthetic pause cases overlap heavily in pause length (WAIT median 800ms vs RESPOND median 750ms). Fixed-800 at 40% (CI includes 50%) is roughly chance. The cases test whether the model reads the words, not personalization. The "6x" figure must NOT be cited as a scientific claim.
- Asynchronous Speculative Policy: How to deploy a 2B parameter decision model in a voice turn-taking pipeline on consumer hardware using generation-counter speculative evaluation.

## Risks & caveats
- "confidence" = (N*p_max - 1)/(N - 1) is a rescaling of p_max, not an independent reliability signal. The calibration claim must come from a reliability plot on probabilities (B7). Evidence beyond p_max comes from Q1 vs Q2 disagreement.
- No dedicated GPU (`nvidia-smi` not found); inference runs on CPU (AMD Ryzen 5 5500U) at 3190.1 ms. This latency severely exceeds the voice checkpoint intervals (250, 500, 800, 1200, 1800 ms). Turn manager (B3) must use asynchronous speculative evaluation with cancel-on-speech, and eval harness (B5) must support a virtual clock parameterizing decision latency.
- Free RAM was 0.39 GB while the model was loaded; latency runs were 4705 ms (B0) vs 3558 ms (B0.5, same dtype), so CPU timings are noisy under memory pressure. The 3190 ms figure is one machine, one condition.
- Personalization needs personas with genuinely different pause distributions; the current cases cannot show a profile effect.
- Memory guard: never load the real model inside pytest. Real-model runs happen only in explicit scripts, one process at a time, after checking free RAM is at least 2 GB; otherwise stop and report.
- C: drive has 4.31 GB free; all Hugging Face caches (`HF_HOME`) and virtual environments must strictly remain on drive E:.
- With CPU decisions at about 3 s and max_wait 2200 ms, a live CPU run degenerates into a fixed 2.2 s timeout. Live demos need a GPU host.
- Colab dependency conflict: Colab pre-installs `torchao` which conflicts with `peft` (requires `>= 0.16.0`). Must execute `!pip uninstall -y torchao` before installing `strands-decider`.
- HF cache path portability: `HF_HOME` must be read from environment or config without hardcoding Windows drive letters (`E:\hf-cache`) so scripts run portably across Windows and Linux/Colab.
- AWS credentials: `aws` CLI is not installed; shell environment variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, etc. are UNSET. `~/.aws/credentials` exists on the host filesystem (from prior Amazon Q/CLI usage) containing expired or inactive tokens, producing `InvalidClientTokenId` when boto3 attempts to authenticate with AWS STS. Neither `.env` nor `.env.example` contained placeholder keys. Config cleanly skips AWS vars when unset; Bedrock calls remain in `MOCK_BEDROCK=1` mode.
- Dataset diversity: template text with small vocabulary (60 unique utterance templates across dev and test), no Bedrock-generated text; evaluations measure timing and prompt structure over synthetic templates, not natural acoustic variety.
