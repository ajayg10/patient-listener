# Patient Listener Pattern: Regret-Aware Voice Agent Turn-Taking

A standalone, drop-in [Strands Agents](https://strandsagents.com) `InterventionHandler` that prevents voice agents from awkwardly cutting off users mid-thought.

By coupling fast probability estimation from [Strands Decider](https://github.com/strands-labs/strands-decider) with personalized cadence thresholds, the handler dynamically holds the floor when a speaker pauses to think, trailing off with connectors (`and...`, `so...`, `because...`) or hesitating (`um`, `uh`).

---

## Drop Into Any Agent in Under 10 Lines

```python
from strands import Agent
from patient_listener_pattern import PatientListenerInterventionHandler, PolicyThresholds

# 1. Initialize handler with an injected decider and policy thresholds
handler = PatientListenerInterventionHandler(
    decider=decider_client.decide,
    thresholds=PolicyThresholds(respond_probability_threshold=0.65, min_silence_for_respond_ms=250.0, max_hold_ms=2200.0),
    wait_action_type="deny",  # "deny" (default, suppresses generation) or "guide" (steers backchannel)
)

# 2. Attach intervention to any Strands Agent
agent = Agent(interventions=[handler], system_prompt="You are a helpful assistant.")
```

---

## Strands Intervention Hook Specification

The pattern implements Strands' native typed intervention interface from `strands.interventions.InterventionHandler`:

### 1. Exact Hooks Used
- **Class**: Subclasses `strands.interventions.InterventionHandler`.
- **Methods**:
  - `before_invocation(self, event: BeforeInvocationEvent) -> Proceed | Deny | Guide`
  - `before_model_call(self, event: BeforeModelCallEvent) -> Proceed | Deny | Guide`

### 2. What Each Hook Receives
- `BeforeInvocationEvent`: Carries the agent instance (`event.agent`), current input messages (`event.messages`), optional caller context (`event.invocation_state`), and cancellation flag (`event.cancel`).
- `BeforeModelCallEvent`: Carries `event.agent`, `event.invocation_state`, and `event.projected_input_tokens`.

### 3. Where Turn State Comes From
- **Default Extraction** (`_default_extract_state`): Reads recent transcript tokens from `event.messages[-1]` and pairs them with current silence duration (defaulting to the 250ms voice checkpoint cadence).
- **Custom Injection** (`get_turn_state`): Callers can inject a custom `Callable[[BeforeInvocationEvent | BeforeModelCallEvent], TurnState]` to pull real-time vad/silence metrics directly from websocket buffers, audio streaming frames, or session metadata.

### 4. What Each Action Does to the Agent Loop
- **`Proceed()`**: The agent loop continues execution unimpeded. The model call executes, tokens are generated, and the assistant responds.
- **`Deny(reason=...)`**: Blocks execution immediately. In `_apply_before_invocation`, Strands sets `event.cancel = f"DENIED: {action.reason}"` and stops the pipeline. The agent terminates the invocation cycle cleanly without invoking LLM inference or consuming output tokens.
- **`Guide(feedback=..., reason=...)`**: Injects steering feedback into the agent loop. On `before_invocation`, sets `event.cancel = f"GUIDANCE: {action.feedback}"`. On `before_model_call`, appends the guidance as a user message to steer model generation.

### 5. `Deny` vs `Guide` for `WAIT` Actions
- **Why `Deny` is Default**: When a speaker pauses mid-thought, the primary goal is conversational silence (floor retention). Calling `Deny` halts the model before a single token is consumed, preventing disruptive voice overlaps and wasteful LLM charges.
- **When to Use `Guide`**: If a voice application requires active listening backchannels (e.g. nodding, saying "mm-hmm" or "take your time"), configuring `wait_action_type="guide"` instructs the model to hold the main answer while emitting an empathetic backchannel.

---

## Configurable Thresholds

`PolicyThresholds` parameters can be configured per persona or globally:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `respond_probability_threshold` | `float` | `0.65` | Minimum $P(\text{RESPOND})$ required from decider to yield turn to assistant. |
| `min_silence_for_respond_ms` | `float` | `250.0` ms | Conversational breath buffer; forces `WAIT` on sub-250ms pauses. |
| `max_hold_ms` | `float` | `2200.0` ms | Deadlock timeout; unconditionally triggers `RESPOND` if silence exceeds this limit. |
| `wait_action_type` | `str` | `"deny"` | Action returned on WAIT: `"deny"` (suppresses model generation) or `"guide"` (steers backchannel). |

---

## Measured Benchmarks & Caveats

All reported performance metrics originate strictly from the measured benchmark suite recorded in `results.md`:

1. **Synthetic Evaluation Dataset**: All performance numbers are measured on the synthetic turn-taking dataset (`test_turns.jsonl`, $N = 324$ turns across fast, slow, and filler personas).
2. **Hardware & Latency Constraints**:
   - **Local CPU (AMD/Intel 6-core)**: Local inference on CPU takes ~3,190 ms median latency per decision. Synchronous evaluation on CPU causes turn-taking to hit the 2,200 ms timeout window.
   - **GPU Inference**: Synchronous real-time conversational turn-taking requires deploying Strands Decider on a GPU host (e.g. AWS g5.xlarge or Colab T4) or running speculative pre-computation. See `scripts/gpu_run.md` for GPU server deployment.
3. **Standalone Integrity**: The `patient_listener_pattern` package has **zero imports** from `backend/` or any web framework. It depends solely on standard Python, `strands`, and `strands_decider.schema`.

---

## Offline Walkthrough

To run the offline demonstration (using `MockDecider` and `MockStrandsModel` without cloud dependencies):

```bash
uv run python patient_listener_pattern/example.py
```
