# Patient Listener

Turn-taking for voice agents that waits when you pause to think and answers when you're done. Built for the AWS Agent Experience Hackathon with **Strands Decider 2B** in the `#policy` pattern.

`#strands-decider-2b` `#policy`

A voice agent has to decide, every time the user goes quiet, whether the turn is over. Most agents use a fixed silence timeout. That cuts off slow speakers mid-sentence and makes fast speakers wait. Patient Listener asks Decider a typed question at a few silence checkpoints ("is this turn complete?"), turns the answer into WAIT or RESPOND with a small policy, and can add a per-user pace profile to the state.

## What I measured

All evaluation numbers are on a **SYNTHETIC** benchmark: 360 test turns from 30 users (3 pause-habit archetypes), 180 dev turns from 15 users. Decisions come from the **real** Decider 2B, precomputed once on a Tesla T4 (8,450 states, 0 missing) and replayed on a virtual clock. Full tables: [results.md](results.md).

Headline at **Point A** (operating points chosen on dev, reported on test; dev delay budget equal to fixed 800 ms):

| Approach | Premature cut-ins (cluster 95% CI) | Delay p50 | Delay p95 |
|---|---|---|---|
| Fixed 800 ms timeout | 18.9% [11.9%, 26.4%] | 800 ms | 800 ms |
| Text heuristic (dev-tuned) | 11.1% [6.4%, 16.7%] | 800 ms | 800 ms |
| Per-user pause threshold (no model) | 4.2% [1.4%, 7.8%] | 808 ms | 1067 ms |
| Decider 2B | 10.0% [5.8%, 14.7%] | 1501 ms | 2200 ms |
| Decider 2B + pace profile | 8.1% [4.2%, 12.2%] | 551 ms | 2200 ms |

Three findings, stated plainly:

1. **Decider beats a fixed timeout.** Paired difference vs fixed 800 at Point A: -8.9 points of cut-ins [-15.6, -3.0]. Decider + profile beats fixed 800 and the text heuristic on both cut-ins and median delay (point estimates).
2. **Personalization helps even without a model.** A per-user pause threshold built from each user's history gets the lowest cut-in rate. Against it and against the tuned text heuristic, the Decider's paired differences span zero. The Decider's added value over tuned model-free baselines is **not established** on this data.
3. **Latency decides whether it works live.** Decider runs at 3,190 ms per decision on my laptop CPU (Ryzen 5 5500U), which is slower than the 2,200 ms max-hold cap, so a CPU run collapses into a fixed timeout. On a T4 it is 235.6 ms median (benchmark) and 301.2 ms (median over the real evaluation states). Asking two questions in one call cost 1.7x to 2.0x one question, so there was no batching gain.

Also measured: the policy needs a delay term. My first real run calibrated on cut-ins only and every turn waited for the 2,200 ms cap. The Q1/Q2 conflict rule never fired (0 of 1,800 checkpoints).

## Architecture

```mermaid
flowchart LR
  A[Speech events<br/>words + silence] --> B[State builder<br/>last ~40 words, silence ms,<br/>optional pace profile]
  B --> C[Turn manager<br/>checkpoints 250..1800 ms,<br/>generation counter, cancel on speech]
  C --> D[DeciderClient<br/>mock / local / remote]
  D -->|p(done)| E[Policy<br/>hi / lo / max hold]
  E -->|RESPOND| F[Reply agent<br/>Strands Agent + Bedrock]
  E -->|WAIT| C
  D -.precompute.-> G[(Decision cache<br/>8,450 real T4 decisions)]
  G --> H[Eval harness<br/>virtual clock replay]
  G --> I[FastAPI<br/>/replay /results /health]
  I --> J[React demo<br/>fixed 800 vs Patient Listener]
  K[Regret log + pace profile<br/>sqlite] --> B
```

## Run the demo (Windows PowerShell)

Prerequisites: Python 3.11, Node 20+, and [uv](https://docs.astral.sh/uv/) (`pip install uv`).

Or just open the live site: https://main.d179rqv15h3x1f.amplifyapp.com

```powershell
git clone https://github.com/ajayg10/patient-listener.git
cd patient-listener
uv sync --extra dev
$env:MOCK_BEDROCK="1"   # set to 0 to generate replies with Amazon Bedrock
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second window:

```powershell
cd patient-listener\frontend
Copy-Item .env.example .env
npm ci
npm run dev
```

Open http://localhost:5173. The green banner confirms the real decision cache is loaded. Pick a scenario and press **Start Replay**, or **Play Demo Set**.

The demo set is **selected**: 10 scenarios where fixed 800 cuts in and the Decider answers at its first checkpoint. It shows the mechanism. The aggregate numbers above and on the Results tab cover all 360 test turns.

No GPU is needed: the replay reads the committed cache `backend/eval/data/decisions_cache_real.jsonl.gz`.

## Reproduce the evaluation

```powershell
uv run python -m backend.eval.run_eval --cache backend/eval/data/decisions_cache_real.jsonl
uv run pytest -q
```

The eval takes a few seconds and writes `results.md` and the plots in `backend/eval/results/`. To regenerate the decision cache with the real model, follow [scripts/gpu_run.md](scripts/gpu_run.md) on a Colab or Kaggle T4 (about 35 minutes).

## The reusable pattern

[`patient_listener_pattern/`](patient_listener_pattern/) is standalone (no imports from `backend/`). It is a Strands `InterventionHandler` that hooks `before_invocation` and `before_model_call`, asks an injected decider, and returns `Proceed` on RESPOND and `Deny` (or `Guide`) on WAIT.

```python
from strands import Agent
from patient_listener_pattern import PatientListenerInterventionHandler, PolicyThresholds

handler = PatientListenerInterventionHandler(
    decider=my_decider.decide,  # real model, remote server, or mock
    thresholds=PolicyThresholds(respond_probability_threshold=0.30, max_hold_ms=2200.0),
)
agent = Agent(interventions=[handler])
```

Calibrate the thresholds on your own data. The values here come from this synthetic benchmark. Offline example: `uv run python patient_listener_pattern/example.py`.

## Bedrock

The reply agent (`backend/app/reply_agent.py`) is a Strands Agent on `BedrockModel` (Claude Haiku 4.5, `us.anthropic.claude-haiku-4-5-20251001-v1:0`) with streaming. In the replay, the moment Patient Listener decides RESPOND, the user's words go to Bedrock and the reply streams into the right-hand panel with the model name and time to first token. The fixed-timeout engine never calls Bedrock. If a Bedrock call fails, the replay continues with a mock reply that is labeled "Mock reply" in the UI. Set `MOCK_BEDROCK=0` to use Bedrock locally (credentials come from your normal AWS profile); tests always use the mock.

## Live deployment

- Frontend (Amplify Hosting): https://main.d179rqv15h3x1f.amplifyapp.com
- API (AWS Lambda Function URL, response streaming): https://segabhplz5dsghnz7nzvt65hcq0bjihq.lambda-url.us-east-1.on.aws/health

The API runs FastAPI on Lambda through the [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) layer, so the replay's event stream works over a plain Function URL. The Lambda serves cached Decider decisions (it never loads the 2B model) and calls Bedrock for the reply. Its role can only write logs and invoke Bedrock models. Regrets logged on the live site are stored in the function's `/tmp` and reset when the instance is recycled. Redeploy with `.\scripts\deploy_aws.ps1` (needs the AWS CLI with your credentials).

## Limits

- **Synthetic data**: 60 utterance templates, 3 archetypes, 30 test and 15 dev users. No real speech or audio.
- Per-user headroom in this dataset is small (archetype-aware vs user-aware reference: 81.4% vs 81.9%).
- Operating points chosen on dev did not transfer exactly to test (Decider at Point A: test median delay above the 800 ms budget; per-user control at Point B: test cut-ins above 5%). Dev has only 15 users.
- T4 numbers use reference kernels (`causal_conv1d` and `flash-linear-attention` not installed). Latency on other hardware will differ.
- Replay uses cached decisions. Live mic input was cut for time.

## Repo map

| Path | What |
|---|---|
| `backend/app/` | events, state builder, turn manager, Decider client, policy, reply agent, FastAPI |
| `backend/eval/` | dataset generator, baselines, metrics, replay eval, real decision cache |
| `patient_listener_pattern/` | the reusable intervention handler |
| `frontend/` | React side-by-side demo and results page |
| `scripts/` | GPU precompute, benchmark, cache validation, demo picker |
| `NOTES.md` | build log: every step, failure and decision |
