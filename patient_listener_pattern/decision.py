"""Pure turn-taking decision policy, typed question schema, and thresholds.

STANDALONE MODULE: Contains zero imports from backend/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from strands_decider.schema import ChoiceQuestion

Action = Literal["WAIT", "RESPOND"]

TURN_COMPLETE_QUESTION = ChoiceQuestion(
    instructions="Has the speaker finished their conversational turn and yielded the floor, or are they pausing mid-thought?",
    criteria={
        "WAIT": "The speaker is pausing mid-thought, hesitating, or trailing off, intending to continue speaking.",
        "RESPOND": "The speaker has finished their turn and yielded the conversational floor.",
    },
)


@dataclass(frozen=True)
class PolicyThresholds:
    """Configurable thresholds for the turn-taking policy."""

    respond_probability_threshold: float = 0.65
    min_silence_for_respond_ms: float = 250.0
    max_hold_ms: float = 2200.0


@dataclass
class TurnState:
    """Current conversational turn state passed to decider."""

    transcript: str
    silence_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_state_text(self) -> str:
        """Format state text into standard Decider prompt format."""
        # Budget recent tokens (approx 40 words)
        words = self.transcript.split()[-40:]
        recent_text = " ".join(words)
        return f'Speaker: "{recent_text}"\n[Silence: {int(self.silence_ms)} ms]'


class DecisionProtocol(Protocol):
    """Protocol for decider return values."""

    choice: str
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float


def policy(
    probabilities: dict[str, float],
    silence_ms: float,
    thresholds: PolicyThresholds | None = None,
) -> Action:
    """Pure turn-taking policy mapping probabilities and silence duration to WAIT or RESPOND.

    Rules:
    1. Safety Cap: If silence_ms >= max_hold_ms -> RESPOND (prevents indefinite silence deadlocks).
    2. Cadence Floor: If silence_ms < min_silence_for_respond_ms -> WAIT (natural conversational pause buffer).
    3. Model Confidence: If P(RESPOND) >= respond_probability_threshold -> RESPOND.
    4. Default: WAIT.
    """
    cfg = thresholds or PolicyThresholds()

    if silence_ms >= cfg.max_hold_ms:
        return "RESPOND"

    if silence_ms < cfg.min_silence_for_respond_ms:
        return "WAIT"

    p_respond = probabilities.get("RESPOND", 0.0)
    if p_respond >= cfg.respond_probability_threshold:
        return "RESPOND"

    return "WAIT"
