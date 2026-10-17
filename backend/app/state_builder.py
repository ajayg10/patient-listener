"""State builder constructing compact context representations for System One Decider."""

from __future__ import annotations

import re

from backend.app.config import StateBuilderConfig


class StateBuilder:
    """Maintains recent utterance buffer and formats compact state texts."""

    def __init__(self, cfg: StateBuilderConfig | None = None) -> None:
        self.cfg = cfg or StateBuilderConfig()
        self._tokens: list[str] = []

    def append_transcript(self, text: str) -> None:
        """Append incremental ASR words to the conversational buffer."""
        words = text.strip().split()
        if words:
            self._tokens.extend(words)

    def get_full_transcript(self) -> str:
        """Return full transcript of current utterance."""
        return " ".join(self._tokens)

    def clear(self) -> None:
        """Reset utterance buffer on turn completion."""
        self._tokens.clear()

    def build_state_text(
        self,
        silence_ms: int,
        pace_summary: str | None = None,
    ) -> str:
        """Build compact prompt state.

        Format:
        Speaker: "<recent ~40 words>"
        [Silence: <silence_ms> ms]
        [Pace profile: <pace_summary>]  # optional
        """
        # Select last ~40 words
        max_words = self.cfg.max_recent_words
        recent_words = self._tokens[-max_words:] if len(self._tokens) > max_words else self._tokens
        speech_text = " ".join(recent_words) if recent_words else "..."

        # Sanitize internal newlines/quotes
        speech_text = re.sub(r'\s+', ' ', speech_text).strip('"')

        parts = [
            f'Speaker: "{speech_text}"',
            f"[Silence: {silence_ms} ms]",
        ]

        if self.cfg.include_pace_summary and pace_summary:
            parts.append(f"Pace profile: {pace_summary.strip()}")

        return "\n".join(parts)
