"""Turn Manager controlling conversational turn-taking, pause checkpoints, and speculative inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from backend.app.clock import Clock, SystemClock, VirtualClock
from backend.app.config import AppConfig, TurnManagerConfig, get_config
from backend.app.decider_service import DeciderClient, Decision, MockDecider
from backend.app.events import (
    DecisionCompleted,
    DecisionDropped,
    DecisionRequested,
    MaxHoldExceeded,
    SilenceProbe,
    SpeechDelta,
    SpeechEnded,
    SpeechStarted,
    TurnEvent,
    TurnYielded,
)
from backend.app.questions import TURN_COMPLETE_QUESTION
from backend.app.state_builder import StateBuilder

if TYPE_CHECKING:
    from strands_decider.schema import ChoiceQuestion


class TurnManager:
    """Manages audio state machine, silence checkpoints, and speculative decider queries."""

    def __init__(
        self,
        decider: DeciderClient | None = None,
        clock: Clock | None = None,
        state_builder: StateBuilder | None = None,
        cfg: TurnManagerConfig | None = None,
        question: ChoiceQuestion | None = None,
        on_event: Callable[[TurnEvent], None] | None = None,
    ) -> None:
        app_cfg: AppConfig = get_config()
        self.cfg = cfg or app_cfg.turn_manager
        self.decider = decider or MockDecider()
        self.clock = clock or SystemClock()
        self.state_builder = state_builder or StateBuilder()
        self.question = question or TURN_COMPLETE_QUESTION
        self.on_event_cb = on_event

        # Checkpoints in ascending order
        self.checkpoints = sorted(self.cfg.checkpoints_ms)
        self.max_wait_ms = self.cfg.max_wait_ms

        # State machine
        self.state: str = "IDLE"  # "IDLE" | "LISTENING" | "PAUSED" | "RESPONDING"
        self.generation: int = 0
        self.silence_start_time_ms: float | None = None
        self.next_checkpoint_idx: int = 0

        # In-flight decision tracking
        self.in_flight: bool = False
        self.in_flight_generation: int | None = None
        self.in_flight_checkpoint_ms: int | None = None

        # Virtual clock pending callback simulation
        self._pending_virtual_decision: tuple[float, Decision, int, int] | None = None

        # Event log
        self.events: list[TurnEvent] = []

    def _emit(self, event: TurnEvent) -> None:
        self.events.append(event)
        if self.on_event_cb:
            self.on_event_cb(event)

    def on_speech_started(self, timestamp_ms: float | None = None) -> None:
        """Called when VAD signals speech onset."""
        now = timestamp_ms if timestamp_ms is not None else self.clock.now_ms()

        # If speech resumes while paused or decision in-flight, cancel immediately
        if self.in_flight:
            assert self.in_flight_generation is not None
            self._emit(DecisionDropped(
                generation=self.in_flight_generation,
                reason="speech_resumed",
                timestamp_ms=now,
            ))
            self.in_flight = False
            self.in_flight_generation = None
            self.in_flight_checkpoint_ms = None
            self._pending_virtual_decision = None

        self.generation += 1
        self.state = "LISTENING"
        self.silence_start_time_ms = None
        self.next_checkpoint_idx = 0

        self._emit(SpeechStarted(timestamp_ms=now))

    def on_speech_delta(self, text: str, timestamp_ms: float | None = None) -> None:
        """Called when ASR emits incremental words."""
        now = timestamp_ms if timestamp_ms is not None else self.clock.now_ms()

        # Speech delta while in pause state also indicates speech resumption
        if self.state == "PAUSED" or self.in_flight:
            if self.in_flight:
                assert self.in_flight_generation is not None
                self._emit(DecisionDropped(
                    generation=self.in_flight_generation,
                    reason="speech_delta_interruption",
                    timestamp_ms=now,
                ))
                self.in_flight = False
                self.in_flight_generation = None
                self.in_flight_checkpoint_ms = None
                self._pending_virtual_decision = None

            self.generation += 1
            self.state = "LISTENING"
            self.silence_start_time_ms = None
            self.next_checkpoint_idx = 0

        self.state_builder.append_transcript(text)
        self._emit(SpeechDelta(text=text, timestamp_ms=now))

    def on_speech_ended(self, timestamp_ms: float | None = None) -> None:
        """Called when VAD signals speech offset / silence onset."""
        now = timestamp_ms if timestamp_ms is not None else self.clock.now_ms()

        self.generation += 1
        self.state = "PAUSED"
        self.silence_start_time_ms = now
        self.next_checkpoint_idx = 0

        self._emit(SpeechEnded(timestamp_ms=now))

    def tick(self, timestamp_ms: float | None = None) -> list[TurnEvent]:
        """Periodic clock tick checking silence thresholds and resolving pending decisions."""
        now = timestamp_ms if timestamp_ms is not None else self.clock.now_ms()
        start_event_count = len(self.events)

        # 1. Resolve pending virtual decisions if due
        if self._pending_virtual_decision is not None:
            due_ms, decision, gen, cp = self._pending_virtual_decision
            if now >= due_ms:
                self._pending_virtual_decision = None
                self.on_decision_arrived(decision, generation=gen, checkpoint_ms=cp, timestamp_ms=now)

        # If not paused, nothing to evaluate
        if self.state != "PAUSED" or self.silence_start_time_ms is None:
            return self.events[start_event_count:]

        silence_ms = now - self.silence_start_time_ms

        # 2. Hard max-wait cap check
        if silence_ms >= self.max_wait_ms:
            if self.in_flight:
                assert self.in_flight_generation is not None
                self._emit(DecisionDropped(
                    generation=self.in_flight_generation,
                    reason="max_hold_exceeded",
                    timestamp_ms=now,
                ))
                self.in_flight = False
                self.in_flight_generation = None
                self.in_flight_checkpoint_ms = None
                self._pending_virtual_decision = None

            self._emit(MaxHoldExceeded(max_hold_ms=self.max_wait_ms, timestamp_ms=now))
            self._emit(TurnYielded(action="RESPOND", reason="max_hold_exceeded", timestamp_ms=now))
            self.state = "RESPONDING"
            return self.events[start_event_count:]

        # 3. Checkpoint probes
        if self.next_checkpoint_idx < len(self.checkpoints):
            cp_target = self.checkpoints[self.next_checkpoint_idx]
            if silence_ms >= cp_target:
                self.next_checkpoint_idx += 1
                self._emit(SilenceProbe(silence_ms=silence_ms, timestamp_ms=now))

                # Launch speculative query only if no decision is already in-flight
                if not self.in_flight:
                    self._dispatch_decision(checkpoint_ms=cp_target, silence_ms=int(silence_ms), now=now)

        return self.events[start_event_count:]

    def _dispatch_decision(self, checkpoint_ms: int, silence_ms: int, now: float) -> None:
        """Dispatch decision request to DeciderClient."""
        gen = self.generation
        self.in_flight = True
        self.in_flight_generation = gen
        self.in_flight_checkpoint_ms = checkpoint_ms

        self._emit(DecisionRequested(checkpoint_ms=checkpoint_ms, generation=gen, timestamp_ms=now))

        state_text = self.state_builder.build_state_text(silence_ms=silence_ms)

        # In VirtualClock mode with simulated latency, schedule future delivery
        if isinstance(self.clock, VirtualClock) and self.clock.simulated_decision_latency_ms > 0:
            decision = self.decider.decide(state_text, self.question)
            due_time = now + self.clock.simulated_decision_latency_ms
            self._pending_virtual_decision = (due_time, decision, gen, checkpoint_ms)
        else:
            # Synchronous delivery (oracle or immediate mock)
            decision = self.decider.decide(state_text, self.question)
            self.on_decision_arrived(decision, generation=gen, checkpoint_ms=checkpoint_ms, timestamp_ms=now)

    def on_decision_arrived(
        self,
        decision: Decision,
        generation: int,
        checkpoint_ms: int,
        timestamp_ms: float | None = None,
    ) -> None:
        """Called when an asynchronous Decider query finishes."""
        now = timestamp_ms if timestamp_ms is not None else self.clock.now_ms()

        # Stale result check: generation counter mismatch or state transition away from PAUSED
        if generation != self.generation or self.state != "PAUSED":
            self.in_flight = False
            self.in_flight_generation = None
            self.in_flight_checkpoint_ms = None
            self._emit(DecisionDropped(
                generation=generation,
                reason="stale_generation",
                timestamp_ms=now,
            ))
            return

        self.in_flight = False
        self.in_flight_generation = None
        self.in_flight_checkpoint_ms = None

        self._emit(DecisionCompleted(
            decision=decision,
            generation=generation,
            checkpoint_ms=checkpoint_ms,
            timestamp_ms=now,
        ))

        # Placeholder Policy Hook (B3): RESPOND if model choice is RESPOND
        # In B7, this will be replaced with calibrated confidence/regret gating.
        if decision.choice == "RESPOND":
            self._emit(TurnYielded(
                action="RESPOND",
                reason="decider_respond",
                timestamp_ms=now,
            ))
            self.state = "RESPONDING"
        else:
            # Model says WAIT; continue in PAUSED state awaiting next checkpoint or speech
            pass

    def reset(self) -> None:
        """Reset state machine for a new conversation turn."""
        self.state = "IDLE"
        self.generation = 0
        self.silence_start_time_ms = None
        self.next_checkpoint_idx = 0
        self.in_flight = False
        self.in_flight_generation = None
        self.in_flight_checkpoint_ms = None
        self._pending_virtual_decision = None
        self.state_builder.clear()
