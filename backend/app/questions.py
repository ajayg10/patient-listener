"""Typed System One question definitions for Patient Listener.

Question template overhead token counts (measured against Qwen/Qwen3.5-2B-Base tokenizer):
- B0.5 baseline template: 95 suffix tokens (103 tokens with state prompt framing).
- Q1 turn_complete: 81 suffix tokens (saving 14 tokens vs original baseline).
- Q2 mid_thought: 73 suffix tokens (saving 22 tokens vs original baseline).
"""

from __future__ import annotations

from strands_decider.schema import ChoiceQuestion

# Q1: Binary turn completion choice (WAIT vs RESPOND)
# Used as the primary turn-taking decision.
TURN_COMPLETE_QUESTION = ChoiceQuestion(
    instructions="Has the speaker finished their turn, or are they pausing mid-thought?",
    criteria={
        "WAIT": "Speaker paused mid-thought or trailing off, intending to continue.",
        "RESPOND": "Speaker finished their thought or question, yielding the turn.",
    },
)

# Q2: Mid-thought inquiry (YES vs NO)
# Used for secondary confirmation or multi-question joint evaluation.
MID_THOUGHT_QUESTION = ChoiceQuestion(
    instructions="Is the speaker pausing mid-thought with intention to continue?",
    criteria={
        "YES": "Speaker paused mid-sentence, hesitating or searching for words.",
        "NO": "Speaker completed their statement or question.",
    },
)

# Registry for lookup by canonical name
QUESTIONS: dict[str, ChoiceQuestion] = {
    "turn_complete": TURN_COMPLETE_QUESTION,
    "mid_thought": MID_THOUGHT_QUESTION,
}
