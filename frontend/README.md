# Patient Listener — Frontend Application

A responsive, real-time React + TypeScript + Tailwind CSS web interface for Patient Listener: the personalized, regret-aware voice conversational turn-taking engine.

---

## 1. Quickstart (Windows PowerShell)

### Prerequisites
- Node.js >= 20.x (`node --version`)
- npm >= 10.x (`npm --version`)
- FastAPI Backend running locally on port 8000

### Step 1: Start the Backend (Terminal 1)
From the repository root `E:\patient-listener`:
```powershell
# Activate Python virtual environment and run FastAPI
.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Verify backend health:
```powershell
curl http://127.0.0.1:8000/health
```

### Step 2: Configure Environment (Terminal 2)
In `frontend/`:
```powershell
cd frontend
Copy-Item .env.example .env
```
Ensure `VITE_API_URL` points to your backend:
```env
VITE_API_URL=http://localhost:8000
```

### Step 3: Run Frontend Development Server
```powershell
cd frontend
npm install
npm run dev
```
Open your browser at `http://localhost:5173`.

---

---

## 2. Manual Demo Run Steps (Step F3 Hero Demo)

### Step 1: Start Backend Server (Terminal 1)
From repository root `E:\patient-listener`:
```powershell
.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Verify the server starts with the mock cache:
```powershell
curl.exe -s http://127.0.0.1:8000/health
```
Expected output:
```json
{"status":"healthy","backend":"mock","cache_provenance":{"backend":"mock","total_keys":448,"is_mock":true},...}
```

### Step 2: Verify High-Speed Replay via SSE Stream (PowerShell / curl.exe)
In another terminal, test a replay run at 20x speed:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/replay/turn_user_fast_test_01_tmpl_test_04_004?speed=20"
```
You will see the line-delimited SSE event stream matching `src/events.ts`:
1. `metadata` (backend provenance, cache key count, latency note)
2. `speechstarted` / `speechdelta` / `speechended` (audio word tokens)
3. `decision_requested` / `decision_completed` (speculative System One checkpoints)
4. `respond` and `final` summary events per engine (`fixed` and `patient_listener`)

### Step 3: Run Frontend UI (Terminal 2)
In `frontend/`:
```powershell
npm run dev
```
Navigate to `http://localhost:5173`.

### Visual Verification Checklist (What to Look At in Browser)
1. **Provenance Banner (Top)**:
   - With mock cache: Prominent red full-width banner:
     `MOCK-DECIDER, not valid for the demo — real GPU decision cache not installed`
   - With real T4 cache: Emerald badge:
     `Decisions: real Strands Decider 2B (cached), decision latency 236 ms measured on Tesla T4`
2. **Side-by-Side Demo Tab**:
   - **Scenario Picker**: Catalog dropdown showing demo-flagged scenarios (`★ [DEMO]`) first with persona, speaker, and text preview.
   - **Speed Controls**: Toggle 1x, 2x, 4x.
   - **Start & Cancel Replay**: Click **Start Replay**. Button flips to **Cancel Replay** (`AbortController`).
   - **Dual Panels**:
     - *Left Panel*: **Fixed 800 ms timeout** shows real-time transcript, live silence meter (turns red at 800 ms), premature cut-in indicator if speech continues.
     - *Right Panel*: **Patient Listener (Decider 2B)** shows transcript, live silence meter, and decision checkpoint cards (`checkpoint_ms`, `WAIT` vs `RESPOND`, `p(done)`, and `latency_ms`).
   - **Unified Timeline Strip**: Shared horizontal time axis rendering speech blocks, pause silence gaps, true end marker, and response markers for both engines.
   - **Outcome Line**: Below panels, displays the comparative summary (e.g. Fixed cut in vs Patient Listener waited and answered correctly).
   - **Play Demo Set**: Click **Play Demo Set** to sequentially evaluate all demo scenarios with a live tally banner showing total cut-ins and mean response delays.
3. **Benchmark Results Tab**:
   - Click **Benchmark Results** tab at the top.
   - **Headline Table**: Approaches (Fixed 800, Text Heuristic, Per-User Control, Strands Decider 2B, Decider+profile, and Reference Ceilings). Displays Cluster 95% CIs and Bootstrap CIs directly from `GET /results`.
   - **Dual Evaluation Plots**: Serves `results_pareto.png` and `results_reliability.png` directly from `GET /plots/{name}`.
   - **Caveats**: Prominently highlights `SYNTHETIC Data Caveat`, non-parametric cluster bootstrap methodology, and Tesla T4 latency baseline.

---

## 3. Production Build

To test and compile the production bundle:
```powershell
cd frontend
npm run build
```
Build outputs are written to `frontend/dist/`.

To preview the production bundle locally:
```powershell
npm run preview
```

---

## 3. Architecture & Features

1. **Typed Event Protocol (`src/events.ts`)**:
   - Strictly mirrors backend event dataclasses (`backend/app/events.py`).
   - Includes `SpeechStarted`, `SpeechDelta`, `SpeechEnded`, `SilenceProbe`, `DecisionRequested`, `DecisionCompleted`, `DecisionDropped`, `TurnYielded`, `MaxHoldExceeded`.
   - Python-TypeScript parity is continuously enforced via `backend/tests/test_event_parity.py`.

2. **Resilient WebSocket Client (`src/ws.ts`)**:
   - Connects to `WS /ws/session?user={user}&ticker=true`.
   - Automatic reconnect with exponential backoff and randomized jitter.
   - Dispatches structured events to the server and receives streamed decisions, agent reply chunks, and final metrics.

3. **Scenario Catalog**:
   - Fetches dynamic test scenarios from `GET /scenarios`.
   - Inspects archetype pause distributions (Fast, Slow, Filler-Heavy) and true turn completion times.

---

## 4. Git Hygiene
The following paths are strictly gitignored:
- `node_modules/`
- `dist/`
- `.env`
