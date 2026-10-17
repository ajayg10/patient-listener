"""Unit tests for ReplyAgent and end-to-end turn-taking to reply pipeline."""

from __future__ import annotations

import pytest

from backend.app.clock import VirtualClock
from backend.app.decider_service import MockDecider
from backend.app.events import TurnYielded
from backend.app.reply_agent import ReplyAgent
from backend.app.turn_manager import TurnManager


@pytest.mark.asyncio
async def test_reply_agent_mock_streaming():
    """ReplyAgent with MOCK_BEDROCK=1 must stream contextual tokens."""
    agent = ReplyAgent(mock=True)

    # 1. Test streaming appointment response
    chunks: list[str] = []
    async for chunk in agent.stream_reply("I want to schedule an appointment"):
        chunks.append(chunk)

    full_reply = "".join(chunks).strip()
    assert len(chunks) > 1
    assert "appointment" in full_reply.lower()

    # 2. Test non-streaming generate_reply
    prescription_reply = await agent.generate_reply("Can you help refill my prescription medication?")
    assert "prescription" in prescription_reply.lower()


def test_reply_agent_should_respond_gating():
    """Reply agent must strictly gate generation on RESPOND action."""
    assert ReplyAgent.should_respond("RESPOND") is True
    assert ReplyAgent.should_respond("respond") is True
    assert ReplyAgent.should_respond("WAIT") is False
    assert ReplyAgent.should_respond("wait") is False
    assert ReplyAgent.should_respond("HOLD") is False


@pytest.mark.asyncio
async def test_end_to_end_scripted_turn_to_reply():
    """End-to-end integration: speech -> pause -> MockDecider -> RESPOND -> ReplyAgent stream."""
    clock = VirtualClock(start_ms=0.0)
    decider = MockDecider(fake_latency_ms=0.0)
    tm = TurnManager(decider=decider, clock=clock)
    reply_agent = ReplyAgent(mock=True)

    # 1. User says completed sentence
    tm.on_speech_started(clock.now_ms())
    clock.advance_ms(500)
    user_utterance = "Can I make an appointment for next Monday?"
    tm.on_speech_delta(user_utterance, clock.now_ms())
    clock.advance_ms(200)
    tm.on_speech_ended(clock.now_ms())  # Silence begins at t=700 ms

    # 2. Advance time past first checkpoint (250 ms -> t=960 ms)
    clock.advance_ms(260)
    events = tm.tick(clock.now_ms())

    # 3. Verify Decider yielded turn to agent
    yielded_events = [e for e in events if isinstance(e, TurnYielded)]
    assert len(yielded_events) == 1
    turn_yielded = yielded_events[0]
    assert turn_yielded.action == "RESPOND"

    # 4. Trigger reply agent only because action is RESPOND
    assert ReplyAgent.should_respond(turn_yielded.action) is True

    transcript = tm.state_builder.get_full_transcript()
    assert transcript == user_utterance

    # 5. Stream reply tokens
    reply_tokens = []
    async for token in reply_agent.stream_reply(transcript):
        reply_tokens.append(token)

    assert len(reply_tokens) > 0
    full_response = "".join(reply_tokens).strip()
    assert "appointment" in full_response.lower()


@pytest.mark.asyncio
async def test_end_to_end_hesitation_does_not_trigger_reply():
    """When Decider outputs WAIT on hesitation, reply agent must NOT be triggered."""
    clock = VirtualClock(start_ms=0.0)
    decider = MockDecider(fake_latency_ms=0.0)
    tm = TurnManager(decider=decider, clock=clock)
    reply_agent = ReplyAgent(mock=True)
    assert reply_agent.mock is True

    # User pauses mid-thought with trailing connector
    tm.on_speech_started(clock.now_ms())
    tm.on_speech_delta("I need to confirm my insurance number and...", clock.now_ms())
    tm.on_speech_ended(clock.now_ms())

    # Advance past 250 ms checkpoint
    clock.advance_ms(260)
    events = tm.tick(clock.now_ms())

    # No TurnYielded event should be emitted
    yielded_events = [e for e in events if isinstance(e, TurnYielded)]
    assert len(yielded_events) == 0
    assert tm.state == "PAUSED"

    # Reply agent should not be triggered while in PAUSED state
    assert not any(ReplyAgent.should_respond(getattr(e, "action", "")) for e in events)
