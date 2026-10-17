"""Strands Agent InterventionHandler implementing patient-listener turn-taking.

STANDALONE MODULE: Contains zero imports from backend/.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

from strands.hooks import BeforeInvocationEvent, BeforeModelCallEvent
from strands.interventions import Deny, Guide, InterventionHandler, Proceed
from strands_decider.schema import ChoiceQuestion

from patient_listener_pattern.decision import (
    TURN_COMPLETE_QUESTION,
    Action,
    DecisionProtocol,
    PolicyThresholds,
    TurnState,
    policy,
)

logger = logging.getLogger(__name__)

WaitActionType = Literal["deny", "guide"]


class PatientListenerInterventionHandler(InterventionHandler):
    """Strands InterventionHandler steering agent turn-taking before model execution.

    Hooks into the Strands Agent loop (before_invocation / before_model_call).
    When the user is pausing mid-thought, suppresses response generation via Deny() (default)
    or steers conversational backchannel via Guide().
    Once the conversational turn has finished, yields floor via Proceed().
    """

    name: str = "patient-listener-turn-taking"

    def __init__(
        self,
        decider: Callable[[str, ChoiceQuestion], DecisionProtocol],
        thresholds: PolicyThresholds | None = None,
        get_turn_state: Callable[[BeforeInvocationEvent | BeforeModelCallEvent], TurnState] | None = None,
        wait_action_type: WaitActionType = "deny",
    ) -> None:
        """Initialize handler with an injected decider callable.

        Args:
            decider: Callable taking (state_text, question) and returning a DecisionProtocol object.
            thresholds: Configurable policy thresholds (P(RESPOND) threshold, min silence, max hold).
            get_turn_state: Optional hook extracting custom TurnState (transcript, silence_ms) from event.
            wait_action_type: Action type returned for WAIT ("deny" to halt execution, "guide" to steer).
        """
        self.decider = decider
        self.thresholds = thresholds or PolicyThresholds()
        self.get_turn_state = get_turn_state or self._default_extract_state
        self.wait_action_type: WaitActionType = wait_action_type

        # Latest decision tracking for inspection / telemetry
        self.last_state: TurnState | None = None
        self.last_decision: DecisionProtocol | None = None
        self.last_action: Action | None = None

    def _default_extract_state(self, event: BeforeInvocationEvent | BeforeModelCallEvent) -> TurnState:
        """Extract transcript and silence from event attributes or message context."""
        transcript = ""
        silence_ms = 250.0  # default checkpoint duration if not specified

        # Check for user message in event
        if hasattr(event, "messages") and event.messages:
            last_msg = event.messages[-1]
            if isinstance(last_msg, dict):
                content = last_msg.get("content", [])
                if isinstance(content, list) and content:
                    first_block = content[0]
                    if isinstance(first_block, dict):
                        transcript = first_block.get("text", "")
                    elif isinstance(first_block, str):
                        transcript = first_block
                elif isinstance(content, str):
                    transcript = content
            elif hasattr(last_msg, "content"):
                transcript = str(last_msg.content)

        return TurnState(transcript=transcript, silence_ms=silence_ms)

    def evaluate_turn(self, state: TurnState) -> Action:
        """Evaluate turn state against injected decider and policy."""
        state_text = state.to_state_text()
        decision = self.decider(state_text, TURN_COMPLETE_QUESTION)
        action = policy(decision.probabilities, state.silence_ms, self.thresholds)

        self.last_state = state
        self.last_decision = decision
        self.last_action = action

        return action

    def before_invocation(
        self, event: BeforeInvocationEvent, **kwargs: Any
    ) -> Proceed | Deny | Guide:
        """Called before an agent invocation begins. Suppresses or guides reply if user is pausing."""
        state = self.get_turn_state(event)
        action = self.evaluate_turn(state)

        if action == "WAIT":
            p_resp = self.last_decision.probabilities.get("RESPOND", 0.0) if self.last_decision else 0.0
            reason = (
                f"Speaker is pausing mid-thought (silence {int(state.silence_ms)} ms, "
                f"P(RESPOND)={p_resp:.2f}). Floor retained by speaker."
            )
            if self.wait_action_type == "guide":
                logger.info(f"[PatientListenerHandler] Guiding invocation: {reason}")
                return Guide(feedback=f"Hold response: {reason}", reason=reason)
            logger.info(f"[PatientListenerHandler] Denying invocation: {reason}")
            return Deny(reason=reason)

        logger.info("[PatientListenerHandler] Proceeding with agent response (turn finished).")
        return Proceed()

    def before_model_call(
        self, event: BeforeModelCallEvent, **kwargs: Any
    ) -> Proceed | Deny | Guide:
        """Called before a model call. Ensures no redundant tokens are consumed while paused."""
        if self.last_action == "WAIT":
            if self.wait_action_type == "guide":
                return Guide(
                    feedback="Speaker is pausing mid-thought; do not interrupt or generate a complete answer.",
                    reason="Turn not yielded; guidance injected.",
                )
            return Deny(reason="Turn not yielded; model call suppressed.")
        return Proceed()
