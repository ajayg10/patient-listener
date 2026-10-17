"""Integration test verifying real strands.Agent with stub model and PatientListener intervention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from strands import Agent
from strands.models.model import Model
from strands.types.content import Messages
from strands_decider.schema import ChoiceQuestion

from patient_listener_pattern import PatientListenerInterventionHandler, TurnState


@dataclass
class MockDecisionResult:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float = 0.0


class StubStrandsModel(Model):
    """Stub model providing offline deterministic text generation without cloud calls."""

    def __init__(self, reply: str = "Medication refill request authorized.") -> None:
        super().__init__()
        self.reply = reply
        self.invocations = 0

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {}

    async def structured_output(self, output_model: Any, prompt: Messages, system_prompt: str | None = None, **kwargs: Any):
        yield {}

    async def stream(self, messages: Messages, tool_specs: Any = None, system_prompt: str | None = None, **kwargs: Any):
        self.invocations += 1
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"contentBlockIndex": 0, "start": {"text": ""}}}
        yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": self.reply}}}
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "end_turn"}}


@pytest.mark.asyncio
async def test_real_strands_agent_offline_wait_and_respond():
    """Verify real strands.Agent offline with stub model: one WAIT turn (denied), one RESPOND turn (proceeds)."""
    current_state = TurnState(transcript="", silence_ms=0.0)

    def scripted_decider(state_text: str, _q: ChoiceQuestion) -> MockDecisionResult:
        if "pause" in state_text.lower():
            return MockDecisionResult("WAIT", {"WAIT": 0.90, "RESPOND": 0.10}, 0.80)
        return MockDecisionResult("RESPOND", {"WAIT": 0.10, "RESPOND": 0.90}, 0.80)

    handler = PatientListenerInterventionHandler(
        decider=scripted_decider,
        get_turn_state=lambda _ev: current_state,
    )

    stub_model = StubStrandsModel()
    agent = Agent(
        model=stub_model,
        interventions=[handler],
        system_prompt="You are a clinical voice assistant.",
    )

    # 1. Turn 1: Speaker pauses mid-thought -> Model call denied (suppressed)
    current_state = TurnState(transcript="I need to pause and think...", silence_ms=300.0)
    result_wait = await agent.invoke_async("I need to pause and think...")
    assert "DENIED:" in result_wait.message["content"][0]["text"]
    assert handler.last_action == "WAIT"
    assert stub_model.invocations == 0

    # 2. Turn 2: Speaker completes turn -> Intervention proceeds, stub model generates response
    current_state = TurnState(transcript="Please refill my blood pressure medication.", silence_ms=400.0)
    result_respond = await agent.invoke_async("Please refill my blood pressure medication.")
    assert result_respond.message["content"][0]["text"] == "Medication refill request authorized."
    assert handler.last_action == "RESPOND"
    assert stub_model.invocations == 1
