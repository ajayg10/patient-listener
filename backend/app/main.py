"""FastAPI application for Patient Listener backend.

Exposes:
- WS /ws/session: real-time turn-taking session with streaming LLM replies
- POST /regret: premature interruption regret logging
- GET /profile/{user}: user conversational pacing profile
- GET /scenarios: demo scenarios catalog
- POST /replay/{scenario}: side-by-side SSE engine replay
- GET /results: benchmark evaluation results
- GET /health: backend and Bedrock health status
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.app.clock import SystemClock
from backend.app.config import AppConfig, get_config
from backend.app.decider_service import MockDecider
from backend.app.events import (
    DecisionCompleted,
    DecisionDropped,
    DecisionRequested,
    MaxHoldExceeded,
    TurnEvent,
    TurnYielded,
)
from backend.app.pace_profile import PaceProfileBuilder
from backend.app.policy import PolicyThresholds, policy
from backend.app.reply_agent import ReplyAgent
from backend.app.state_builder import StateBuilder
from backend.app.turn_manager import TurnManager
from scripts.precompute_decisions import compute_cache_key, load_cache

logger = logging.getLogger(__name__)

cfg = get_config()
app = FastAPI(title="Patient Listener API", version="0.1.0")

# CORS middleware from configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pace_profile_builder = PaceProfileBuilder()


def load_scenario_by_id(scenario_id: str) -> dict[str, Any] | None:
    """Find scenario by id from test or dev datasets."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    test_file = repo_root / "backend" / "eval" / "data" / "test_turns.jsonl"
    dev_file = repo_root / "backend" / "eval" / "data" / "dev_turns.jsonl"

    for fpath in [test_file, dev_file]:
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        t = json.loads(line)
                        if t.get("id") == scenario_id:
                            return t
    return None


def build_dual_engine_timeline(
    t: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    app_cfg: AppConfig,
    model_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Simulate side-by-side execution of Fixed Timeout engine vs Patient Listener engine."""
    events: list[dict[str, Any]] = []
    cache_miss_error: dict[str, Any] | None = None

    fixed_timeout = float(app_cfg.fixed_replay_timeout_ms)
    decision_latency = float(app_cfg.replay_decision_latency_ms)
    policy_thresh = PolicyThresholds(
        hi=app_cfg.policy.confidence_hi,
        lo=app_cfg.policy.confidence_lo,
    )
    checkpoints = sorted(app_cfg.turn_manager.checkpoints_ms)
    max_wait_ms = float(app_cfg.turn_manager.max_wait_ms)

    speech_end_t = float(t.get("speech_end_t", 0.0))
    turn_end_t = float(t.get("turn_end_t", speech_end_t))
    silence_ms = float(t.get("silence_ms", 0.0))
    is_complete = bool(t.get("is_turn_complete", True))
    raw_events = t.get("events", [])

    # 1. Forward raw audio events
    for rev in raw_events:
        events.append({
            "event": rev.get("type", "speech_event").lower(),
            "engine": "audio",
            "text": rev.get("text"),
            "t_ms": float(rev.get("timestamp_ms", 0.0)),
        })

    # 2. Fixed Timeout Engine
    if not is_complete and silence_ms > fixed_timeout:
        # Cuts off speaker during mid-thought pause
        act_t = speech_end_t + fixed_timeout
        events.append({
            "event": "respond",
            "engine": "fixed",
            "reason": "fixed_timeout",
            "silence_ms": fixed_timeout,
            "t_ms": act_t,
        })
        events.append({
            "event": "final",
            "engine": "fixed",
            "interrupted": True,
            "response_delay_ms": 0.0,
            "t_ms": act_t,
        })
    else:
        # Responds after true turn completion
        act_t = turn_end_t + fixed_timeout
        events.append({
            "event": "respond",
            "engine": "fixed",
            "reason": "fixed_timeout",
            "silence_ms": fixed_timeout,
            "t_ms": act_t,
        })
        events.append({
            "event": "final",
            "engine": "fixed",
            "interrupted": False,
            "response_delay_ms": fixed_timeout,
            "t_ms": act_t,
        })

    # 3. Patient Listener Engine (evaluated strictly against active cache)
    sb = StateBuilder()
    first_text = " ".join(t.get("text", "").split()[-40:])
    sb.append_transcript(first_text)

    pl_responded = False
    for cp in checkpoints:
        if not is_complete and cp > silence_ms:
            break
        if cp >= max_wait_ms:
            break

        dispatch_t = speech_end_t + cp
        events.append({
            "event": "decision_requested",
            "engine": "patient_listener",
            "checkpoint_ms": cp,
            "t_ms": dispatch_t,
        })

        st = sb.build_state_text(silence_ms=cp)
        k = compute_cache_key(model_id, "turn_complete", st)
        if k not in cache:
            cache_miss_error = {
                "event": "error",
                "engine": "patient_listener",
                "error": "cache_miss",
                "message": f"Cache miss for state key {k} at checkpoint {cp}ms. State text: '{st}'",
                "t_ms": dispatch_t,
            }
            break

        cached = cache[k]
        act = policy(cached["probabilities"], cp, thresholds=policy_thresh)
        arrival_t = dispatch_t + decision_latency

        # Check if speech resumed before decision arrived
        if not is_complete and arrival_t >= (speech_end_t + silence_ms):
            events.append({
                "event": "decision_dropped",
                "engine": "patient_listener",
                "reason": "speech_resumed",
                "checkpoint_ms": cp,
                "t_ms": speech_end_t + silence_ms,
            })
            break

        events.append({
            "event": "decision_completed",
            "engine": "patient_listener",
            "checkpoint_ms": cp,
            "choice": cached.get("choice", act),
            "p_done": cached.get("probabilities", {}).get("RESPOND", 0.0),
            "action": act,
            "latency_ms": decision_latency,
            "t_ms": arrival_t,
        })

        if act == "RESPOND":
            interrupted = (not is_complete)
            events.append({
                "event": "respond",
                "engine": "patient_listener",
                "reason": "decider_respond",
                "t_ms": arrival_t,
            })
            events.append({
                "event": "final",
                "engine": "patient_listener",
                "interrupted": interrupted,
                "response_delay_ms": max(0.0, arrival_t - turn_end_t) if not interrupted else 0.0,
                "t_ms": arrival_t,
            })
            pl_responded = True
            break

    # If Patient Listener did not respond during initial pause, evaluate checkpoints after true end
    if not pl_responded and not cache_miss_error:
        full_text = " ".join((t.get("full_text") or t.get("text", "")).split()[-40:])
        sb_full = StateBuilder()
        sb_full.append_transcript(full_text)

        pl_final_responded = False
        for cp in checkpoints:
            dispatch_t = turn_end_t + cp
            events.append({
                "event": "decision_requested",
                "engine": "patient_listener",
                "checkpoint_ms": cp,
                "t_ms": dispatch_t,
            })

            st = sb_full.build_state_text(silence_ms=cp)
            k = compute_cache_key(model_id, "turn_complete", st)
            if k not in cache:
                cache_miss_error = {
                    "event": "error",
                    "engine": "patient_listener",
                    "error": "cache_miss",
                    "message": f"Cache miss for state key {k} at checkpoint {cp}ms after true end. State text: '{st}'",
                    "t_ms": dispatch_t,
                }
                break

            cached = cache[k]
            act = policy(cached["probabilities"], cp, thresholds=policy_thresh)
            arrival_t = dispatch_t + decision_latency

            events.append({
                "event": "decision_completed",
                "engine": "patient_listener",
                "checkpoint_ms": cp,
                "choice": cached.get("choice", act),
                "p_done": cached.get("probabilities", {}).get("RESPOND", 0.0),
                "action": act,
                "latency_ms": decision_latency,
                "t_ms": arrival_t,
            })

            if act == "RESPOND":
                delay_after_end = arrival_t - turn_end_t
                events.append({
                    "event": "respond",
                    "engine": "patient_listener",
                    "reason": "decider_respond",
                    "t_ms": arrival_t,
                })
                events.append({
                    "event": "final",
                    "engine": "patient_listener",
                    "interrupted": False,
                    "response_delay_ms": delay_after_end,
                    "t_ms": arrival_t,
                })
                pl_final_responded = True
                break

        if not pl_final_responded and not cache_miss_error:
            # Fallback to max_wait if no checkpoint yielded RESPOND
            max_hold_t = turn_end_t + max_wait_ms
            events.append({
                "event": "respond",
                "engine": "patient_listener",
                "reason": "max_hold_exceeded",
                "t_ms": max_hold_t,
            })
            events.append({
                "event": "final",
                "engine": "patient_listener",
                "interrupted": False,
                "response_delay_ms": max_wait_ms,
                "t_ms": max_hold_t,
            })

    events.sort(key=lambda x: (x["t_ms"], 0 if x.get("engine") == "audio" else 1))
    return events, cache_miss_error


@app.get("/scenarios")
def list_scenarios() -> list[dict[str, Any]]:
    """Lists demo scenarios (id, user, persona, text preview, true end time)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    demo_file = Path(cfg.demo_scenarios_path)
    if not demo_file.is_absolute():
        demo_file = repo_root / demo_file

    if demo_file.exists():
        try:
            with open(demo_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                scenarios = data.get("scenarios", [])
                if scenarios:
                    return scenarios
        except Exception as e:
            logger.warning(f"Failed to read demo_scenarios file: {e}")

    # Fallback to scanning test_turns.jsonl
    test_file = repo_root / "backend" / "eval" / "data" / "test_turns.jsonl"
    if not test_file.exists():
        return []

    results = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                results.append({
                    "id": t["id"],
                    "user": t.get("user_id", ""),
                    "persona": t.get("archetype") or t.get("persona_id", ""),
                    "text_preview": t.get("full_text") or t.get("text", ""),
                    "true_end_time": t.get("turn_end_t", 0.0),
                    "silence_ms": t.get("silence_ms", 0.0),
                    "category": t.get("category", ""),
                })
                if len(results) >= 20:
                    break
    return results


@app.post("/replay/{scenario}")
async def replay_scenario(scenario: str, request: Request, speed: float = 1.0) -> StreamingResponse:
    """Streams scripted scenario through Fixed Timeout and Patient Listener engines in parallel.

    Emits SSE (text/event-stream) labeled with cache provenance.
    Simulates Tesla T4 median latency from config.
    Returns clear error event on cache miss (never fallback).
    """
    t = load_scenario_by_id(scenario)
    if not t:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario}' not found")

    repo_root = Path(__file__).resolve().parent.parent.parent
    cache_file = Path(cfg.decision_cache_path)
    if not cache_file.is_absolute():
        cache_file = repo_root / cache_file

    cache = load_cache(cache_file) if cache_file.exists() else {}

    sample_entry = next(iter(cache.values())) if cache else {}
    backend = sample_entry.get("backend", "mock")
    model_id = sample_entry.get("model_id", cfg.decider_model_id)

    cache_provenance = {
        "backend": backend,
        "model_id": model_id,
        "keys_count": len(cache),
        "note": "latency measured on Tesla T4, decisions from cache",
        "fixed_timeout_ms": cfg.fixed_replay_timeout_ms,
        "decision_latency_ms": cfg.replay_decision_latency_ms,
    }

    effective_speed = max(0.01, speed)

    async def sse_stream() -> AsyncIterator[str]:
        # Initial provenance metadata header event
        yield f"data: {json.dumps({'event': 'metadata', 'engine': 'system', 'provenance': cache_provenance, 't_ms': 0.0})}\n\n"

        timeline, cache_miss_error = build_dual_engine_timeline(t, cache, cfg, model_id)

        last_t = 0.0
        for ev in timeline:
            dt = ev["t_ms"] - last_t
            if dt > 0 and effective_speed < 100.0:
                await asyncio.sleep(dt / (1000.0 * effective_speed))
            last_t = ev["t_ms"]
            yield f"data: {json.dumps(ev)}\n\n"

        if cache_miss_error:
            yield f"data: {json.dumps(cache_miss_error)}\n\n"

    return StreamingResponse(
        sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Cache-Provenance": backend,
        },
    )


@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket) -> None:
    """Real-time conversational turn-taking WebSocket session.

    Client sends events: SpeechStarted, SpeechDelta, SpeechEnded, tick / SilenceProbe.
    Server emits decisions (probs, p_done, latency_ms, action, checkpoint) and on RESPOND,
    streams reply tokens from ReplyAgent. If live Bedrock fails, falls back to calibrated
    mock reply with reply_source: 'mock'.
    """
    await websocket.accept()

    user_id = websocket.query_params.get("user", "default_user")
    clock = SystemClock()
    profile = pace_profile_builder.build_profile(user_id)
    state_builder = StateBuilder()

    policy_thresholds = PolicyThresholds(
        hi=cfg.policy.confidence_hi,
        lo=cfg.policy.confidence_lo,
    )
    reply_agent = ReplyAgent(cfg=cfg)

    # Outgoing event queue for async dispatch over websocket
    outgoing_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def handle_turn_event(event: TurnEvent) -> None:
        ts = event.timestamp_ms
        if isinstance(event, DecisionCompleted):
            dec = event.decision
            act = policy(dec.probabilities, event.checkpoint_ms, thresholds=policy_thresholds)
            outgoing_queue.put_nowait({
                "type": "decision",
                "probs": dec.probabilities,
                "p_done": dec.probabilities.get("RESPOND", 0.0),
                "latency_ms": dec.latency_ms,
                "action": act,
                "checkpoint": event.checkpoint_ms,
                "timestamp_ms": ts,
            })
        elif isinstance(event, TurnYielded):
            outgoing_queue.put_nowait({
                "type": "turn_yielded",
                "action": event.action,
                "reason": event.reason,
                "timestamp_ms": ts,
            })
        elif isinstance(event, DecisionRequested):
            outgoing_queue.put_nowait({
                "type": "decision_requested",
                "checkpoint_ms": event.checkpoint_ms,
                "generation": event.generation,
                "timestamp_ms": ts,
            })
        elif isinstance(event, DecisionDropped):
            outgoing_queue.put_nowait({
                "type": "decision_dropped",
                "reason": event.reason,
                "generation": event.generation,
                "timestamp_ms": ts,
            })
        elif isinstance(event, MaxHoldExceeded):
            outgoing_queue.put_nowait({
                "type": "max_hold_exceeded",
                "max_hold_ms": event.max_hold_ms,
                "timestamp_ms": ts,
            })

    turn_mgr = TurnManager(
        decider=MockDecider(),
        clock=clock,
        state_builder=state_builder,
        policy_thresholds=policy_thresholds,
        pace_summary=profile.render(),
        on_event=handle_turn_event,
    )

    async def stream_agent_reply(transcript: str) -> None:
        """Stream reply tokens with graceful fallback to mock on Bedrock errors."""
        source = "mock" if reply_agent.mock else "bedrock"
        tokens: list[str] = []
        try:
            async for chunk in reply_agent.stream_reply(transcript):
                tokens.append(chunk)
                await websocket.send_json({
                    "type": "reply_chunk",
                    "chunk": chunk,
                    "reply_source": source,
                    "timestamp_ms": clock.now_ms(),
                })
        except Exception as exc:
            logger.warning(f"Error streaming reply from {source}: {exc}. Falling back to mock.")
            source = "mock"
            mock_text = ReplyAgent._generate_mock_reply(transcript)
            for word in mock_text.split():
                chunk = word + " "
                tokens.append(chunk)
                await websocket.send_json({
                    "type": "reply_chunk",
                    "chunk": chunk,
                    "reply_source": "mock",
                    "timestamp_ms": clock.now_ms(),
                })

        await websocket.send_json({
            "type": "reply_done",
            "full_reply": "".join(tokens).strip(),
            "reply_source": source,
            "timestamp_ms": clock.now_ms(),
        })

    async def sender_task() -> None:
        """Forward queued turn events and trigger replies on RESPOND."""
        while True:
            msg = await outgoing_queue.get()
            await websocket.send_json(msg)
            if msg.get("type") == "turn_yielded" and msg.get("action") == "RESPOND":
                transcript = state_builder.get_full_transcript()
                await stream_agent_reply(transcript)
            outgoing_queue.task_done()

    async def ticker_task() -> None:
        """Periodic background ticker for real-time live sessions."""
        while True:
            await asyncio.sleep(0.05)
            if turn_mgr.state == "PAUSED":
                turn_mgr.tick()

    sender_future = asyncio.create_task(sender_task())
    ticker_future = asyncio.create_task(ticker_task())

    try:
        while True:
            msg = await websocket.receive_json()
            event_type = msg.get("type")
            ts = msg.get("timestamp_ms")

            if event_type == "SpeechStarted":
                turn_mgr.on_speech_started(timestamp_ms=ts)
            elif event_type == "SpeechDelta":
                text = msg.get("text", "")
                turn_mgr.on_speech_delta(text=text, timestamp_ms=ts)
            elif event_type == "SpeechEnded":
                turn_mgr.on_speech_ended(timestamp_ms=ts)
            elif event_type in ("tick", "SilenceProbe"):
                turn_mgr.tick(timestamp_ms=ts)
            elif event_type == "reset":
                turn_mgr.reset()

            # Ensure any synchronous tick consequences are processed
            if turn_mgr.state == "PAUSED":
                turn_mgr.tick(timestamp_ms=ts)
    except WebSocketDisconnect:
        logger.info(f"WebSocket session disconnected for user {user_id}")
    finally:
        sender_future.cancel()
        ticker_future.cancel()
