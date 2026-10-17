"""Unit tests for patient_listener_pattern.

All tests use mocks only. No network, no weights, no AWS credentials.
Asserts zero imports from backend/.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from typing import Any

import pytest
from strands import Agent
from strands.hooks import BeforeInvocationEvent, BeforeModelCallEvent
from strands.interventions import Deny, Proceed
from strands.models.model import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands_decider.schema import ChoiceQuestion

from patient_listener_pattern import (
    PatientListenerInterventionHandler,
    PolicyThresholds,
    TurnState,
    policy,
)
from patient_listener_pattern.decision import TURN_COMPLETE_QUESTION


@dataclass
class MockDecisionResult:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float = 0.0


class MockStrandsModel(Model):
    """Mock Strands model provider for testing."""

    def __init__(self, reply_text: str = "Test response") -> None:
        super().__init__()
        self.reply_text = reply_text
        self.call_count = 0

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
        self.call_count += 1
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
                "delta": {"text": self.reply_text},
            }
        }
        yield {"contentBlockStop": {"contentBlockIndex": 0}}
        yield {"messageStop": {"stopReason": "end_turn"}}


def test_zero_imports_from_backend():
    """Verify that importing patient_listener_pattern does NOT import backend/."""
    # Check current sys.modules for any leaked backend import from patient_listener_pattern
    import patient_listener_pattern
    import patient_listener_pattern.decision
    import patient_listener_pattern.handler

    # Inspect decision.py and handler.py module objects directly
    for mod in [patient_listener_pattern, patient_listener_pattern.decision, patient_listener_pattern.handler]:
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if hasattr(attr, "__module__") and attr.__module__:
                assert not attr.__module__.startswith("backend"), (
                    f"Found import from backend in {mod.__name__}: {attr_name} -> {attr.__module__}"
                )


def test_pure_policy_rules():
    """Verify policy decision rules: safety cap, cadence floor, and threshold."""
    custom = PolicyThresholds(
        respond_probability_threshold=0.70,
        min_silence_for_respond_ms=250.0,
        max_hold_ms=2200.0,
    )

    # 1. Safety Cap: silence >= max_hold_ms returns RESPOND regardless of probability
    assert policy({"RESPOND": 0.05, "WAIT": 0.95}, silence_ms=2200.0, thresholds=custom) == "RESPOND"
    assert policy({"RESPOND": 0.00, "WAIT": 1.00}, silence_ms=3000.0, thresholds=custom) == "RESPOND"

    # 2. Cadence Floor: silence < min_silence_ms returns WAIT even if P(RESPOND) is high
    assert policy({"RESPOND": 0.99, "WAIT": 0.01}, silence_ms=100.0, thresholds=custom) == "WAIT"
    assert policy({"RESPOND": 0.99, "WAIT": 0.01}, silence_ms=249.0, thresholds=custom) == "WAIT"

    # 3. Model Confidence: silence in [min_silence, max_hold)
    # Above or equal threshold -> RESPOND
    assert policy({"RESPOND": 0.70, "WAIT": 0.30}, silence_ms=500.0, thresholds=custom) == "RESPOND"
    assert policy({"RESPOND": 0.85, "WAIT": 0.15}, silence_ms=400.0, thresholds=custom) == "RESPOND"

    # Below threshold -> WAIT
    assert policy({"RESPOND": 0.69, "WAIT": 0.31}, silence_ms=500.0, thresholds=custom) == "WAIT"
    assert policy({"RESPOND": 0.10, "WAIT": 0.90}, silence_ms=1500.0, thresholds=custom) == "WAIT"

    # Missing RESPOND key defaults to 0.0 -> WAIT
    assert policy({}, silence_ms=500.0, thresholds=custom) == "WAIT"


def test_turn_state_formatting():
    """Verify TurnState serializes into the standard prompt format and budgets recent words."""
    state = TurnState(transcript="Hello doctor, I have a headache.", silence_ms=350.0)
    text = state.to_state_text()
    assert text == 'Speaker: "Hello doctor, I have a headache."\n[Silence: 350 ms]'

    # Verify word budgeting (last 40 words)
    long_words = [f"word{i}" for i in range(60)]
    long_state = TurnState(transcript=" ".join(long_words), silence_ms=500.0)
    formatted = long_state.to_state_text()
    assert "word0" not in formatted
    assert "word20" in formatted
    assert "word59" in formatted


from unittest.mock import MagicMock


def test_handler_wait_suppression():
    """Verify that a WAIT decision generates a Deny action in before_invocation."""
    def mock_decider(state_text: str, q: ChoiceQuestion) -> MockDecisionResult:
        assert q.criteria == TURN_COMPLETE_QUESTION.criteria
        return MockDecisionResult(
            choice="WAIT",
            probabilities={"WAIT": 0.90, "RESPOND": 0.10},
            confidence=0.80,
        )

    current_state = TurnState(transcript="I am thinking...", silence_ms=300.0)
    handler = PatientListenerInterventionHandler(
        decider=mock_decider,
        get_turn_state=lambda _ev: current_state,
    )

    # Simulate before_invocation
    mock_event = BeforeInvocationEvent(agent=MagicMock(), messages=[{"role": "user", "content": "I am thinking..."}])
    action = handler.before_invocation(mock_event)

    assert isinstance(action, Deny)
    assert "pausing mid-thought" in action.reason
    assert handler.last_action == "WAIT"

    # Simulate before_model_call
    model_event = BeforeModelCallEvent(agent=MagicMock())
    model_action = handler.before_model_call(model_event)
    assert isinstance(model_action, Deny)


def test_handler_respond_proceed():
    """Verify that a RESPOND decision yields Proceed action in before_invocation."""
    def mock_decider(state_text: str, q: ChoiceQuestion) -> MockDecisionResult:
        return MockDecisionResult(
            choice="RESPOND",
            probabilities={"WAIT": 0.10, "RESPOND": 0.90},
            confidence=0.80,
        )

    current_state = TurnState(transcript="Please refill my medication.", silence_ms=400.0)
    handler = PatientListenerInterventionHandler(
        decider=mock_decider,
        get_turn_state=lambda _ev: current_state,
    )

    mock_event = BeforeInvocationEvent(agent=MagicMock(), messages=[{"role": "user", "content": "Please refill my medication."}])
    action = handler.before_invocation(mock_event)

    assert isinstance(action, Proceed)
    assert handler.last_action == "RESPOND"

    model_event = BeforeModelCallEvent(agent=MagicMock())
    model_action = handler.before_model_call(model_event)
    assert isinstance(model_action, Proceed)


@pytest.mark.asyncio
async def test_agent_integration_with_handler():
    """Verify end-to-end integration with Strands Agent using mock model and decider."""
    current_state = TurnState(transcript="", silence_ms=0.0)

    def decider_logic(state_text: str, _q: ChoiceQuestion) -> MockDecisionResult:
        if "pause" in state_text:
            return MockDecisionResult("WAIT", {"WAIT": 0.95, "RESPOND": 0.05}, 0.90)
        return MockDecisionResult("RESPOND", {"WAIT": 0.05, "RESPOND": 0.95}, 0.90)

    handler = PatientListenerInterventionHandler(
        decider=decider_logic,
        get_turn_state=lambda _ev: current_state,
    )

    mock_model = MockStrandsModel(reply_text="Prescription confirmed.")
    agent = Agent(
        model=mock_model,
        interventions=[handler],
    )

    # 1. Turn with pause -> Denied, model not called
    current_state = TurnState(transcript="I need to pause and think...", silence_ms=300.0)
    res_1 = await agent.invoke_async("I need to pause and think...")
    assert "DENIED:" in res_1.message["content"][0]["text"]
    assert mock_model.call_count == 0

    # 2. Turn completed -> Proceeded, model called
    current_state = TurnState(transcript="I would like a refill now please.", silence_ms=350.0)
    res_2 = await agent.invoke_async("I would like a refill now please.")
    assert res_2.message["content"][0]["text"] == "Prescription confirmed."
    assert mock_model.call_count == 1


def test_handler_guide_action():
    """Verify that wait_action_type='guide' returns Guide instead of Deny."""
    from strands.interventions import Guide

    def mock_decider(_state_text: str, _q: ChoiceQuestion) -> MockDecisionResult:
        return MockDecisionResult("WAIT", {"WAIT": 0.90, "RESPOND": 0.10}, 0.80)

    current_state = TurnState(transcript="I am thinking...", silence_ms=300.0)
    handler = PatientListenerInterventionHandler(
        decider=mock_decider,
        get_turn_state=lambda _ev: current_state,
        wait_action_type="guide",
    )

    mock_event = BeforeInvocationEvent(agent=MagicMock(), messages=[{"role": "user", "content": "I am thinking..."}])
    action = handler.before_invocation(mock_event)
    assert isinstance(action, Guide)
    assert "Hold response" in action.feedback

    model_event = BeforeModelCallEvent(agent=MagicMock())
    model_action = handler.before_model_call(model_event)
    assert isinstance(model_action, Guide)
    assert "Speaker is pausing mid-thought" in model_action.feedback

