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

## 2. Production Build

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
