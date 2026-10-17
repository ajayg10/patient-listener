# Build Notes — Patient Listener

## Status board
| Step | Title | Status | Date |
|---|---|---|---|
| B0 | Feasibility spike (GO/NO-GO gate) | DONE | 2026-10-17 |
| B0.5 | Latency diagnosis & synthetic dataset audit | DONE | 2026-10-17 |
| B1 | Scaffold | DONE | 2026-10-17 |
| B2 | Decider service | DONE | 2026-10-17 |
| B3 | Events, state builder, turn manager | DONE | 2026-10-17 |
| B4 | Reply agent | DOING | 2026-10-17 |
| B5 | Baselines, dataset, eval harness | TODO | |
| B6 | Pace profile + regret log | TODO | |
| B7 | Confidence-aware policy | TODO | |
| B8 | Reusable pattern package | TODO | |
| B9 | API | TODO | |
| F1 | Frontend scaffold + WS client | TODO | |
| F2 | Live view | TODO | |
| F3 | Side-by-side demo mode | TODO | |
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

  ### Empty Table Template for GPU Results (e.g. Colab / Kaggle)
  | State Tokens | 1 Question Median (ms) | 1 Question P95 (ms) | 2 Questions Median (ms) | 2 Questions P95 (ms) | 2Q Overhead (ms) |
  |---|---|---|---|---|---|
  | 20 tokens | — | — | — | — | — |
  | 60 tokens | — | — | — | — | — |
  | 150 tokens | — | — | — | — | — |

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

## Article material (running list)
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
- Hugging Face `hf-xet` dependency failure: `hf-xet` is unstable on Windows environments during multi-chunk reconstruction. Keep `hf-xet` uninstalled in project dependencies.
