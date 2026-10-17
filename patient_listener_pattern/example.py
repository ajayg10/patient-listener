"""Example walkthrough demonstrating the Patient Listener turn-taking intervention.

Runs completely offline with a mock decider and mock model.
Usage:
    python patient_listener_pattern/example.py
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from typing import Any

# Ensure patient_listener_pattern can be imported when run directly as a script
_root = str(pathlib.Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from strands import Agent  # noqa: E402
from strands.models.model import Model  # noqa: E402
from strands.types.content import Messages  # noqa: E402
from strands.types.streaming import StreamEvent  # noqa: E402
from strands_decider.schema import ChoiceQuestion  # noqa: E402

from patient_listener_pattern.decision import (  # noqa: E402
    PolicyThresholds,
    TurnState,
)
from patient_listener_pattern.handler import PatientListenerInterventionHandler  # noqa: E402


@dataclass
class MockDecision:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float = 0.0


class ScriptedDecider:
    """Mock decider returning realistic turn-completion probabilities."""

    def __call__(self, state_text: str, question: ChoiceQuestion) -> MockDecision:
        lower = state_text.lower()
        # Trailing hesitation or connector indicates mid-thought pause
        if "and..." in lower or "so..." in lower or "because..." in lower or "..." in lower:
            return MockDecision(
                choice="WAIT",
                probabilities={"WAIT": 0.88, "RESPOND": 0.12},
                confidence=0.76,
            )
        # Complete grammatical query
        return MockDecision(
            choice="RESPOND",
            probabilities={"WAIT": 0.10, "RESPOND": 0.90},
            confidence=0.80,
        )


class MockStrandsModel(Model):
    """Mock Strands model provider for offline verification without cloud credentials."""

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {}

    async def structured_output(
        self, output_model: Any, prompt: Messages, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        yield {}

    async def stream(
        self,
        messages: Messages,
        tool_specs: Any = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        reply = "Certainly! I have processed your refill request for your blood pressure medication."
        yield {"messageStart": {"role": "assistant"}}
        yield {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"text": ""},
            }
        }
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": reply},
            }
        }
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "end_turn"}}


async def main() -> None:
    print("=" * 70)
    print("Patient Listener Pattern: Offline Walkthrough")
    print("=" * 70)

    # 1. Initialize handler with injected decider and policy thresholds
    current_state = TurnState(transcript="", silence_ms=0.0)

    def state_extractor(_event: Any) -> TurnState:
        return current_state

    handler = PatientListenerInterventionHandler(
        decider=ScriptedDecider(),
        thresholds=PolicyThresholds(
            respond_probability_threshold=0.65,
            min_silence_for_respond_ms=250.0,
            max_hold_ms=2200.0,
        ),
        get_turn_state=state_extractor,
    )

    # 2. Attach handler to any Strands Agent in under 10 lines
    agent = Agent(
        model=MockStrandsModel(),
        interventions=[handler],
        system_prompt="You are a helpful healthcare assistant.",
    )

    # Step 1: User pauses mid-thought
    print("\n[Step 1] User begins speaking and pauses mid-thought:")
    user_chunk_1 = "I called the pharmacy yesterday about my prescription, and..."
    current_state = TurnState(transcript=user_chunk_1, silence_ms=250.0)
    print(f'  User text: "{user_chunk_1}"')
    print("  Silence: 250 ms")

    result_1 = await agent.invoke_async(user_chunk_1)
    print()
    print(f"  Decision : {handler.last_action}")
    print("  Action   : Denied by intervention (Model execution suppressed)")
    print(f"  Result   : {result_1.message['content'][0]['text']}")

    # Step 2: User completes their turn
    print("\n[Step 2] User continues and yields the floor:")
    user_chunk_2 = "Can I get a refill on my blood pressure medication please?"
    current_state = TurnState(transcript=user_chunk_2, silence_ms=300.0)
    print(f'  User text: "{user_chunk_2}"')
    print("  Silence: 300 ms")

    result_2 = await agent.invoke_async(user_chunk_2)
    print()
    print(f"  Decision : {handler.last_action}")
    print("  Action   : Proceeded with response generation")
    print(f"  Result   : {result_2.message['content'][0]['text']}")

    print("\n" + "=" * 70)
    print("Walkthrough complete: Successfully demonstrated WAIT suppression and RESPOND yield.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
