"""Typed conversational turn and audio event dataclasses for Patient Listener."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from backend.app.decider_service import Decision


@dataclass(frozen=True)
class SpeechStarted:
    """Emitted when Voice Activity Detection (VAD) detects speech onset."""

    timestamp_ms: float


@dataclass(frozen=True)
class SpeechDelta:
    """Incremental transcript fragment from Automatic Speech Recognition (ASR)."""

    text: str
    timestamp_ms: float


@dataclass(frozen=True)
class SpeechEnded:
    """Emitted when VAD detects speech offset / silence onset."""

    timestamp_ms: float


@dataclass(frozen=True)
class SilenceProbe:
    """Periodic probe emitted at configured pause checkpoints (e.g. 250, 500, 800 ms)."""

    silence_ms: float
    timestamp_ms: float


@dataclass(frozen=True)
class DecisionRequested:
    """Emitted when an asynchronous Decider query is dispatched."""

    checkpoint_ms: int
    generation: int
    timestamp_ms: float


@dataclass(frozen=True)
class DecisionCompleted:
    """Emitted when an asynchronous Decider query completes successfully."""

    decision: Decision
    generation: int
    checkpoint_ms: int
    timestamp_ms: float


@dataclass(frozen=True)
class DecisionDropped:
    """Emitted when an in-flight decision is discarded (stale generation or interrupted by speech)."""

    generation: int
    reason: str
    timestamp_ms: float


@dataclass(frozen=True)
class TurnYielded:
    """Emitted when the policy determines the conversational turn is complete and yields to agent reply."""

    action: str  # "RESPOND" | "WAIT"
    reason: str  # e.g., "decider_respond", "max_hold_exceeded"
    timestamp_ms: float


@dataclass(frozen=True)
class MaxHoldExceeded:
    """Emitted when silence exceeds the hard safety maximum wait cap."""

    max_hold_ms: int
    timestamp_ms: float


TurnEvent = Union[
    SpeechStarted,
    SpeechDelta,
    SpeechEnded,
    SilenceProbe,
    DecisionRequested,
    DecisionCompleted,
    DecisionDropped,
    TurnYielded,
    MaxHoldExceeded,
]
