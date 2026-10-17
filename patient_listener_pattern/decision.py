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

    hi: float = 0.70
    lo: float = 0.30
    respond_probability_threshold: float = 0.65  # legacy fallback
    min_silence_for_respond_ms: float = 250.0
    max_hold_ms: float = 2200.0

    def get_hi(self) -> float:
        return self.hi if self.hi is not None else self.respond_probability_threshold


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
    q2_probabilities: dict[str, float] | None = None,
) -> Action:
    """Pure turn-taking policy mapping probabilities and silence duration to WAIT or RESPOND.

    Rules:
    1. Safety Cap: If silence_ms >= max_hold_ms -> RESPOND (always wins, prevents indefinite silence).
    2. Cadence Floor: If silence_ms < min_silence_for_respond_ms -> WAIT (natural conversational pause buffer).
    3. Conflict Rule: If Q1 says done (p(RESPOND) >= hi) AND Q2 says mid_thought (p(YES) >= 0.50),
       that is conflicting: keep listening -> WAIT.
    4. Confidence Thresholds:
       - If p(RESPOND) >= hi -> RESPOND.
       - If p(RESPOND) <= lo -> WAIT.
       - In between (lo < p < hi): keep waiting and re-ask at the next checkpoint with more evidence -> WAIT.
    """
    cfg = thresholds or PolicyThresholds()

    # 1. Hard safety cap always wins
    if silence_ms >= cfg.max_hold_ms:
        return "RESPOND"

    # 2. Cadence floor
    if silence_ms < cfg.min_silence_for_respond_ms:
        return "WAIT"

    p_respond = probabilities.get("RESPOND", probabilities.get("YES", 0.0))
    hi_threshold = cfg.get_hi()
    lo_threshold = cfg.lo

    # 3. Conflict Rule: Q1 done AND Q2 mid_thought
    if q2_probabilities is not None:
        p_mid_thought = q2_probabilities.get("YES", q2_probabilities.get("WAIT", 0.0))
        if p_respond >= hi_threshold and p_mid_thought >= 0.50:
            return "WAIT"

    # 4. Confidence thresholds
    if p_respond >= hi_threshold:
        return "RESPOND"

    if p_respond <= lo_threshold:
        return "WAIT"

    # Intermediate uncertainty: wait for next checkpoint
    return "WAIT"
