"""Reply Agent wrapping Strands Agent on Amazon Bedrock with token streaming."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from backend.app.config import AppConfig, get_config

logger = logging.getLogger(__name__)


class ReplyAgent:
    """Conversational reply agent powered by Strands Agents SDK and Amazon Bedrock."""

    def __init__(self, cfg: AppConfig | None = None, mock: bool | None = None) -> None:
        self.cfg = cfg or get_config()
        self.mock = mock if mock is not None else self.cfg.env.mock_bedrock

        self.agent = None
        self.model = None

        if not self.mock:
            try:
                from strands import Agent
                from strands.models import BedrockModel

                self.model = BedrockModel(
                    model_id=self.cfg.env.bedrock_model_id,
                    region_name=self.cfg.env.aws_region,
                )
                self.agent = Agent(
                    model=self.model,
                    system_prompt=(
                        "You are an empathetic, patient clinical intake assistant. "
                        "Respond clearly, concisely, and reassuringly to the patient."
                    ),
                )
            except Exception as e:
                logger.warning(
                    f"[reply_agent] Failed to initialize live BedrockModel ({e}). "
                    "Falling back to calibrated mock agent."
                )
                self.mock = True

    async def stream_reply(self, prompt: str) -> AsyncIterator[str]:
        """Stream conversational reply tokens as an asynchronous generator."""
        if self.mock or self.agent is None:
            # Deterministic, conversational mock generator for sub-millisecond testing
            reply = self._generate_mock_reply(prompt)
            for word in reply.split():
                yield word + " "
        else:
            async for event in self.agent.stream_async(prompt):
                if isinstance(event, dict) and "data" in event and event["data"]:
                    yield str(event["data"])

    async def generate_reply(self, prompt: str) -> str:
        """Non-streaming accumulator returning full generated response."""
        chunks: list[str] = []
        async for chunk in self.stream_reply(prompt):
            chunks.append(chunk)
        return "".join(chunks).strip()

    @staticmethod
    def _generate_mock_reply(prompt: str) -> str:
        """Generate contextual mock response based on utterance content."""
        clean = prompt.lower()
        if "appointment" in clean:
            return "I would be happy to help you schedule an appointment. What day and time work best for you?"
        elif "prescription" in clean or "medication" in clean or "refill" in clean:
            return "I can certainly assist with your prescription refill and pharmacy details."
        elif "billing" in clean or "copay" in clean or "insurance" in clean:
            return "I can review your billing statement and insurance copay breakdown right now."
        return "I understand your request. Let me assist you with that right away."

    @staticmethod
    def should_respond(action: str) -> bool:
        """Enforces gating: reply agent triggers strictly when action is RESPOND."""
        return action.upper() == "RESPOND"
