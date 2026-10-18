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
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.app.clock import SystemClock
from backend.app.config import get_config
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
        confidence_hi=cfg.policy.confidence_hi,
        confidence_lo=cfg.policy.confidence_lo,
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
