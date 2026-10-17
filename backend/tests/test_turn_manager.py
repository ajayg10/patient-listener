"""Unit tests for TurnManager and StateBuilder with scripted virtual timelines."""

from __future__ import annotations

from backend.app.clock import VirtualClock
from backend.app.config import StateBuilderConfig, TurnManagerConfig
from backend.app.decider_service import Decision, MockDecider
from backend.app.events import (
    DecisionCompleted,
    DecisionDropped,
    DecisionRequested,
    MaxHoldExceeded,
    SilenceProbe,
    TurnYielded,
)
from backend.app.state_builder import StateBuilder
from backend.app.turn_manager import TurnManager


def test_state_builder_word_and_token_cap():
    """StateBuilder must enforce max recent words and keep total state tokens < 100."""
    sb = StateBuilder(StateBuilderConfig(max_recent_words=40))

    # Add 80 words
    long_speech = (
        "hello clinic I have a very long story about my health insurance policy "
        "and last week when I called the doctor they told me my copay would be "
        "thirty dollars but then the pharmacy charged me fifty dollars and I do not "
        "understand why there is a discrepancy between the doctor and pharmacy "
        "so I am calling today to ask if someone can clarify this for me please"
    )
    sb.append_transcript(long_speech)

    state_text = sb.build_state_text(silence_ms=650, pace_summary="p50=650ms, p90=1100ms")

    # Word count check: speech part should have at most 40 words
    assert "Speaker:" in state_text
    assert "[Silence: 650 ms]" in state_text
    assert "Pace profile:" in state_text

    # Approximate token count (< 100 tokens)
    # Using conservative heuristic: 1 word ~ 1.3 tokens
    words = state_text.split()
    assert len(words) <= 55
    # Token count should easily fit within 100 tokens
    est_tokens = len(words) * 1.3
    assert est_tokens < 100


def test_fast_decision_turn_completion():
    """Fast decision (0 ms) at checkpoint triggers RESPOND when speaker finishes turn."""
    clock = VirtualClock(start_ms=0.0)
    decider = MockDecider(fake_latency_ms=0.0)
    tm = TurnManager(decider=decider, clock=clock)

    # 1. User speaks a complete question
    clock.advance_ms(100)
    tm.on_speech_started(clock.now_ms())
    clock.advance_ms(500)
    tm.on_speech_delta("Can I make an appointment for tomorrow afternoon?", clock.now_ms())
    clock.advance_ms(200)
    tm.on_speech_ended(clock.now_ms())  # Silence begins at t=800 ms

    # 2. Advance time past first checkpoint (250 ms -> t=1050 ms)
    clock.advance_ms(260)
    events = tm.tick(clock.now_ms())

    # Checkpoint 250 triggered decision which completed and yielded turn
    event_types = [type(e) for e in events]
    assert SilenceProbe in event_types
    assert DecisionRequested in event_types
    assert DecisionCompleted in event_types
    assert TurnYielded in event_types

    turn_yielded = [e for e in events if isinstance(e, TurnYielded)][0]
    assert turn_yielded.action == "RESPOND"
    assert turn_yielded.reason == "decider_respond"
    assert tm.state == "RESPONDING"


def test_slow_decision_dropped_on_subsequent_speech():
    """A slow decision in-flight must be dropped if user resumes speech before it arrives."""
    clock = VirtualClock(start_ms=0.0, simulated_decision_latency_ms=3000.0)
    decider = MockDecider(fake_latency_ms=3000.0)
    tm = TurnManager(decider=decider, clock=clock)

    # 1. Speaker pauses mid-thought
    tm.on_speech_started(clock.now_ms())
    clock.advance_ms(200)
    tm.on_speech_delta("I need to speak with Dr. Evans and...", clock.now_ms())
    clock.advance_ms(100)
    tm.on_speech_ended(clock.now_ms())  # Pause starts at t=300 ms, generation=2

    # 2. Reach checkpoint 250 ms (t=550 ms). Decision is requested with generation=2
    clock.advance_ms(250)
    tm.tick(clock.now_ms())
    assert tm.in_flight is True
    assert tm.in_flight_generation == 2

    # 3. Before 3000 ms decision finishes, speaker resumes at t=800 ms
    clock.advance_ms(250)
    tm.on_speech_started(clock.now_ms())  # Should cancel in-flight decision immediately

    # Verify decision was dropped due to speech resumption
    dropped_events = [e for e in tm.events if isinstance(e, DecisionDropped)]
    assert len(dropped_events) == 1
    assert dropped_events[0].generation == 2
    assert dropped_events[0].reason == "speech_resumed"
    assert tm.in_flight is False

    # 4. If a late response from generation 2 arrives later, it must be rejected as stale
    late_decision = Decision(
        choice="WAIT",
        probabilities={"WAIT": 0.88, "RESPOND": 0.12},
        confidence=0.76,
        latency_ms=3000.0,
        backend="mock",
    )
    clock.advance_ms(3000)
    tm.on_decision_arrived(late_decision, generation=2, checkpoint_ms=250, timestamp_ms=clock.now_ms())

    stale_drops = [e for e in tm.events if isinstance(e, DecisionDropped) and e.reason == "stale_generation"]
    assert len(stale_drops) == 1


def test_max_hold_cap_reached():
    """When silence reaches max_wait_ms, TurnManager forces RESPOND via MaxHoldExceeded."""
    clock = VirtualClock(start_ms=0.0)
    # MockDecider configured to always output WAIT
    decider = MockDecider()
    cfg = TurnManagerConfig(checkpoints_ms=[250, 500, 800], max_wait_ms=1000)
    tm = TurnManager(decider=decider, clock=clock, cfg=cfg)

    # User says hesitation filler
    tm.on_speech_started(clock.now_ms())
    tm.on_speech_delta("Um, let me think...", clock.now_ms())
    tm.on_speech_ended(clock.now_ms())  # Silence starts at t=0 ms

    # Advance through checkpoints: 250, 500, 800 ms (all return WAIT)
    clock.advance_ms(300)
    tm.tick(clock.now_ms())
    assert tm.state == "PAUSED"

    clock.advance_ms(300)  # t=600 ms
    tm.tick(clock.now_ms())
    assert tm.state == "PAUSED"

    clock.advance_ms(300)  # t=900 ms
    tm.tick(clock.now_ms())
    assert tm.state == "PAUSED"

    # Advance past max_wait_ms (1000 ms -> t=1050 ms)
    clock.advance_ms(150)  # t=1050 ms
    events = tm.tick(clock.now_ms())

    event_types = [type(e) for e in events]
    assert MaxHoldExceeded in event_types
    assert TurnYielded in event_types

    max_exceeded = [e for e in events if isinstance(e, MaxHoldExceeded)][0]
    assert max_exceeded.max_hold_ms == 1000

    yielded = [e for e in events if isinstance(e, TurnYielded)][0]
    assert yielded.action == "RESPOND"
    assert yielded.reason == "max_hold_exceeded"
    assert tm.state == "RESPONDING"


def test_speech_resumed_during_in_flight_call():
    """Speech resumption drops in-flight query and transitions back to LISTENING."""
    clock = VirtualClock(start_ms=100.0)
    decider = MockDecider()
    tm = TurnManager(decider=decider, clock=clock)

    tm.on_speech_started(clock.now_ms())
    tm.on_speech_delta("I wanted to check", clock.now_ms())
    tm.on_speech_ended(clock.now_ms())

    # Manually simulate in-flight request
    clock.advance_ms(260)
    tm.in_flight = True
    tm.in_flight_generation = tm.generation
    tm.in_flight_checkpoint_ms = 250

    # User resumes speech before decision completion
    clock.advance_ms(50)
    tm.on_speech_delta("my prescription status", clock.now_ms())

    assert tm.state == "LISTENING"
    assert tm.in_flight is False
    dropped = [e for e in tm.events if isinstance(e, DecisionDropped)]
    assert len(dropped) == 1
    assert dropped[0].reason == "speech_delta_interruption"
