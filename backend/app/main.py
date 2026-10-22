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
import gzip
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

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
from backend.app.regret_log import list_regrets, log_regret
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


class RegretPayload(BaseModel):
    user_id: str
    session_id: str
    state_text: str
    probs: dict[str, float]
    silence_ms: float
    action: str = "RESPOND"


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
                "checkpoint_ms": cp,
                "state_key": k,
                "state_text": st,
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
                    "checkpoint_ms": cp,
                    "state_key": k,
                    "state_text": st,
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
    """Lists demo scenarios (id, user, persona, text preview, true end t_ms, demo flag)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    demo_file = Path(cfg.demo_scenarios_path)
    if not demo_file.is_absolute():
        demo_file = repo_root / demo_file

    demo_ids: set[str] = set()
    demo_scenarios: list[dict[str, Any]] = []
    if demo_file.exists():
        try:
            with open(demo_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                scenarios = data.get("scenarios", [])
                for s in scenarios:
                    sid = s["id"]
                    demo_ids.add(sid)
                    t_end = float(s.get("true_end_time", s.get("true_end_t_ms", 0.0)))
                    demo_scenarios.append({
                        "id": sid,
                        "user": s.get("user") or s.get("user_id", ""),
                        "persona": s.get("persona") or s.get("archetype", ""),
                        "text_preview": s.get("text_preview") or s.get("full_text") or s.get("text", ""),
                        "true_end_t_ms": t_end,
                        "true_end_time": t_end,
                        "silence_ms": float(s.get("silence_ms", 0.0)),
                        "category": s.get("category", ""),
                        "demo": True,
                    })
        except Exception as e:
            logger.warning(f"Failed to read demo_scenarios file: {e}")

    # Also load from test_turns.jsonl for complete catalog
    test_file = repo_root / "backend" / "eval" / "data" / "test_turns.jsonl"
    other_scenarios: list[dict[str, Any]] = []
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    sid = t["id"]
                    if sid in demo_ids:
                        continue
                    t_end = float(t.get("turn_end_t", 0.0))
                    other_scenarios.append({
                        "id": sid,
                        "user": t.get("user_id", ""),
                        "persona": t.get("archetype") or t.get("persona_id", ""),
                        "text_preview": t.get("full_text") or t.get("text", ""),
                        "true_end_t_ms": t_end,
                        "true_end_time": t_end,
                        "silence_ms": float(t.get("silence_ms", 0.0)),
                        "category": t.get("category", ""),
                        "demo": False,
                    })

    return demo_scenarios + other_scenarios


@app.get("/plots/{name}")
def get_plot(name: str) -> FileResponse:
    """Serves whitelisted evaluation plots (results_pareto.png, results_reliability.png)."""
    whitelist = {"results_pareto.png", "results_reliability.png"}
    if name not in whitelist:
        raise HTTPException(status_code=404, detail="Plot not found")
    repo_root = Path(__file__).resolve().parent.parent.parent
    plot_path = repo_root / "backend" / "eval" / "results" / name
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot file does not exist")
    return FileResponse(plot_path, media_type="image/png")


@app.post("/replay/{scenario}")
async def replay_scenario(scenario: str, request: Request, speed: float = 1.0) -> StreamingResponse:
    """Streams scripted scenario through Fixed Timeout and Patient Listener engines in parallel."""
    t = load_scenario_by_id(scenario)
    if not t:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario}' not found")

    repo_root = Path(__file__).resolve().parent.parent.parent
    cache_file = Path(cfg.decision_cache_path)
    if not cache_file.is_absolute():
        cache_file = repo_root / cache_file

    cache = load_cache(cache_file)

    sample_entry = next(iter(cache.values())) if cache else {}
    backend = sample_entry.get("backend", "mock")
    model_id = sample_entry.get("model_id", cfg.decider_model_id)

    prov_file = repo_root / "backend" / "eval" / "data" / "decisions_cache_real_provenance.json"
    latency_source = "Tesla T4 (cache-measured)" if backend != "mock" else "mock"
    latency_ms = float(cfg.replay_decision_latency_ms)
    if prov_file.exists() and backend != "mock":
        try:
            with open(prov_file, "r", encoding="utf-8") as pf:
                prov_data = json.load(pf)
                latency_stats = prov_data.get("latency_statistics", {})
                latency_ms = float(latency_stats.get("overall_median_ms", latency_ms))
                hardware = prov_data.get("hardware", "Tesla T4 (Google Colab free tier)")
                latency_source = f"{hardware} (cache-measured)"
        except Exception:
            pass

    cache_provenance = {
        "backend": backend,
        "model_id": model_id,
        "keys_count": len(cache),
        "note": "latency measured on Tesla T4, decisions from cache",
        "fixed_timeout_ms": cfg.fixed_replay_timeout_ms,
        "decision_latency_ms": latency_ms,
        "latency_source": latency_source,
    }

    effective_speed = max(0.01, speed)

    async def sse_stream() -> AsyncIterator[str]:
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


@app.post("/regret")
def create_regret(payload: RegretPayload) -> dict[str, Any]:
    """Logs a turn decision regret event (e.g. premature interruption) to SQLite database."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    db_path = Path(cfg.regret_db_path)
    if not db_path.is_absolute():
        db_path = repo_root / db_path

    row_id = log_regret(
        user_id=payload.user_id,
        session_id=payload.session_id,
        state_text=payload.state_text,
        probs=payload.probs,
        silence_ms=payload.silence_ms,
        action=payload.action,
        db_path=db_path,
    )
    return {
        "status": "ok",
        "regret_id": row_id,
        "user_id": payload.user_id,
        "silence_ms": payload.silence_ms,
    }


@app.get("/profile/{user}")
def get_user_profile(user: str) -> dict[str, Any]:
    """Retrieves personalized pace profile incorporating user prior history and recorded regrets."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    db_path = Path(cfg.regret_db_path)
    if not db_path.is_absolute():
        db_path = repo_root / db_path

    user_regrets = list_regrets(user_id=user, db_path=db_path)
    profile = pace_profile_builder.build_profile(user_id=user, regrets=user_regrets)

    return {
        "user_id": profile.user_id,
        "resumed_median_ms": profile.resumed_median_ms,
        "resumed_p75_ms": profile.resumed_p75_ms,
        "finished_median_ms": profile.finished_median_ms,
        "finished_p75_ms": profile.finished_p75_ms,
        "filler_rate": profile.filler_rate,
        "typical_words": profile.typical_words,
        "has_history": profile.has_history,
        "rendered": profile.render(),
        "token_count": profile.token_count,
        "regrets_count": len(user_regrets),
    }


@app.get("/results")
def get_eval_results() -> dict[str, Any]:
    """Returns benchmark results JSON plus provenance flag."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_file = repo_root / "backend" / "eval" / "results" / "eval_results.json"

    if not results_file.exists():
        return {
            "status": "not_found",
            "provenance_flag": "missing",
            "results": {},
        }

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    prov = data.get("provenance", {})
    is_mock = prov.get("is_mock", True) if isinstance(prov, dict) else True
    prov_flag = "mock" if is_mock else "real"

    return {
        "provenance_flag": prov_flag,
        "is_mock": is_mock,
        **data,
    }


@app.get("/health")
def health_status() -> dict[str, Any]:
    """Returns backend, cache provenance, and Bedrock status."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    cache_file = Path(cfg.decision_cache_path)
    if not cache_file.is_absolute():
        cache_file = repo_root / cache_file

    cache_backend = "unknown"
    cache_keys = 0
    model_id = cfg.decider_model_id
    if not cache_file.exists():
        gz_cand = Path(str(cache_file) + ".gz")
        if gz_cand.exists():
            cache_file = gz_cand
        elif cache_file.suffix == ".jsonl":
            alt_gz = cache_file.with_suffix(".jsonl.gz")
            if alt_gz.exists():
                cache_file = alt_gz

    if cache_file.exists():
        try:
            is_gz = cache_file.suffix == ".gz" or str(cache_file).endswith(".jsonl.gz")
            open_fn = gzip.open if is_gz else open
            with open_fn(cache_file, "rt", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line.strip():
                    item = json.loads(first_line)
                    cache_backend = item.get("backend", "mock")
                    model_id = item.get("model_id", model_id)
            with open_fn(cache_file, "rt", encoding="utf-8") as f:
                cache_keys = sum(1 for line in f if line.strip())
        except Exception:
            pass

    prov_file = repo_root / "backend" / "eval" / "data" / "decisions_cache_real_provenance.json"
    latency_ms = float(cfg.replay_decision_latency_ms)
    latency_source = "config"
    hardware = "unknown"
    if prov_file.exists() and cache_backend != "mock":
        try:
            with open(prov_file, "r", encoding="utf-8") as pf:
                prov_data = json.load(pf)
                lat_stats = prov_data.get("latency_statistics", {})
                latency_ms = float(lat_stats.get("overall_median_ms", latency_ms))
                hardware = prov_data.get("hardware", "Tesla T4 (Google Colab free tier)")
                latency_source = f"{hardware} (cache-measured)"
        except Exception:
            pass
    elif cache_backend == "mock":
        latency_source = "mock"

    bedrock_status = {
        "status": "SKIPPED",
        "model_id": cfg.env.bedrock_model_id,
        "region": cfg.env.aws_region,
        "mock_mode": cfg.env.mock_bedrock,
        "error": "ValidationException: Operation not allowed / ResourceNotFoundException",
    }

    return {
        "status": "healthy",
        "backend": cfg.decider_backend,
        "policy": {
            "hi": cfg.policy.confidence_hi,
            "lo": cfg.policy.confidence_lo,
            "operating_point": cfg.policy.operating_point,
            "provenance": cfg.policy.operating_point_provenance,
        },
        "cache_provenance": {
            "backend": cache_backend,
            "total_keys": cache_keys,
            "keys_count": cache_keys,
            "model_id": model_id,
            "is_mock": (cache_backend == "mock"),
            "decision_latency_ms": latency_ms,
            "latency_source": latency_source,
            "hardware": hardware,
        },
        "bedrock_status": bedrock_status,
    }


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
        while True:
            msg = await outgoing_queue.get()
            await websocket.send_json(msg)
            if msg.get("type") == "turn_yielded" and msg.get("action") == "RESPOND":
                transcript = state_builder.get_full_transcript()
                await stream_agent_reply(transcript)
            outgoing_queue.task_done()

    async def ticker_task() -> None:
        while True:
            await asyncio.sleep(0.05)
            if turn_mgr.state == "PAUSED":
                turn_mgr.tick()

    enable_ticker = websocket.query_params.get("ticker", "true").lower() in ("true", "1")
    sender_future = asyncio.create_task(sender_task())
    ticker_future = asyncio.create_task(ticker_task()) if enable_ticker else None

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
    except WebSocketDisconnect:
        logger.info(f"WebSocket session disconnected for user {user_id}")
    finally:
        sender_future.cancel()
        if ticker_future:
            ticker_future.cancel()
