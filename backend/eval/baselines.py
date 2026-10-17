"""Evaluation baselines: Fixed silence timeouts and Amazon Bedrock LLM Judge.

SYNTHETIC EVALUATION MODULE.
Note: A mock judge exists strictly for unit test plumbing and must NEVER appear in results.md.
When AWS credentials are absent, the Bedrock LLM judge row is explicitly marked SKIPPED.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.app.config import get_config

logger = logging.getLogger(__name__)


class FixedTimeoutBaseline:
    """Fixed-threshold baseline triggering RESPOND once silence exceeds timeout_ms."""

    def __init__(self, timeout_ms: int = 800) -> None:
        self.timeout_ms = timeout_ms
        self.name = f"Fixed {timeout_ms}ms"

    def decide(self, silence_ms: float) -> str:
        """Return 'RESPOND' if silence >= timeout_ms, else 'WAIT'."""
        return "RESPOND" if silence_ms >= self.timeout_ms else "WAIT"


class TextHeuristicBaseline:
    """Text-only heuristic baseline:
    - If final word is a trailing conjunction/filler or utterance lacks complete terminal punctuation -> WAIT.
    - Else -> RESPOND once silence >= threshold_ms (threshold tuned on DEV split only).
    """

    TRAILING_CONNECTORS = {
        "and", "or", "but", "because", "so", "then", "if", "when",
        "though", "although", "while", "that", "to", "for", "with",
        "as", "since",
    }
    TRAILING_FILLERS = {
        "um", "uh", "er", "ah", "well", "like", "hmm", "right",
    }

    def __init__(self, threshold_ms: int = 250) -> None:
        self.threshold_ms = threshold_ms
        self.name = f"Text Heuristic ({threshold_ms}ms)"

    def is_incomplete(self, text: str) -> bool:
        import re
        trimmed = text.strip()
        if not trimmed:
            return True
        if trimmed.endswith("...") or trimmed.endswith(",") or trimmed.endswith("-"):
            return True
        tokens = re.findall(r"\b[a-zA-Z']+\b", trimmed.lower())
        if not tokens:
            return True
        last_word = tokens[-1]
        if last_word in self.TRAILING_CONNECTORS or last_word in self.TRAILING_FILLERS:
            return True
        if not trimmed.endswith((".", "?", "!")):
            return True
        return False

    def decide(self, text: str, silence_ms: float) -> str:
        if self.is_incomplete(text):
            return "WAIT"
        return "RESPOND" if silence_ms >= self.threshold_ms else "WAIT"



class LLMJudgeBaseline:
    """LLM Judge baseline using Amazon Bedrock Converse API with Claude 3 Haiku.

    Strict protocol: If AWS credentials are absent or invalid, is_available() returns False,
    and evaluation reports the judge baseline as SKIPPED.
    """

    def __init__(self, model_id: str | None = None, region_name: str | None = None, is_mock: bool = False) -> None:
        cfg = get_config()
        self.model_id = model_id or cfg.env.bedrock_model_id
        self.region_name = region_name or cfg.env.aws_region
        self.is_mock = is_mock
        self.name = "Bedrock LLM Judge (Claude 3 Haiku)"

        self._client: Any = None
        self._available = False

        if not self.is_mock and not cfg.env.mock_bedrock:
            try:
                import boto3
                client = boto3.client("bedrock-runtime", region_name=self.region_name)
                # Verify credentials with lightweight STS probe
                sts = boto3.client("sts", region_name=self.region_name)
                sts.get_caller_identity()
                self._client = client
                self._available = True
            except Exception as e:
                logger.info(f"[LLMJudgeBaseline] Bedrock credentials unavailable ({e}). Marking judge as SKIPPED.")
                self._available = False
        elif self.is_mock:
            # Internal test mock only
            self._available = True

    def is_available(self) -> bool:
        """Return True only if credentials are authenticated or in explicit test mock mode."""
        return self._available

    def decide(self, state_text: str, silence_ms: int) -> str:
        """Evaluate state against prompt. Raises RuntimeError if live judge is unavailable."""
        if not self._available:
            raise RuntimeError(
                "Bedrock LLM Judge is unavailable due to missing/invalid AWS credentials. "
                "Baseline must be marked SKIPPED in results."
            )

        if self.is_mock:
            # Mock plumbing logic for test assertion only
            lower = state_text.lower()
            if "..." in lower or "and" in lower.split()[-1:] or silence_ms < 500:
                return "WAIT"
            return "RESPOND"

        # Live Bedrock Converse execution
        import boto3
        client = self._client or boto3.client("bedrock-runtime", region_name=self.region_name)

        system_prompt = (
            "You are an expert conversation turn-taking classifier. "
            "Given the conversational context and silence duration, determine whether the speaker has "
            "finished their turn and yields the floor (RESPOND), or is pausing mid-thought and still speaking (WAIT). "
            "Reply with exactly one word: RESPOND or WAIT."
        )

        prompt = f"{state_text}\n\nSilence: {silence_ms} ms. Decision (RESPOND or WAIT):"

        response = client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 10},
        )

        output_text = response["output"]["message"]["content"][0]["text"].strip().upper()
        return "RESPOND" if "RESPOND" in output_text else "WAIT"
