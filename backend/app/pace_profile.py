"""Pace Profile builder computing per-user rolling conversational statistics.

Constructs personalized pacing stats built ONLY from that user's prior history sessions:
- Pause lengths after which speaker resumed vs finished (median and p75 of each)
- Filler rate
- Typical trailing words before a pause

Renders a deterministic 1-line state representation capped at 30 tokens.
A user with no history gets an empty profile, yielding identical base state text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_HISTORY_PATH = (
    Path(__file__).resolve().parent.parent / "eval" / "data" / "user_histories.jsonl"
)


def count_tokens(text: str) -> int:
    """Approximate word/subword token count for state representation strings."""
    if not text.strip():
        return 0
    # Splits on whitespace and punctuation boundaries
    return len(re.findall(r"\w+|[^\w\s]", text))


@dataclass
class PaceProfile:
    """User conversational pace profile."""

    user_id: str
    resumed_median_ms: float = 0.0
    resumed_p75_ms: float = 0.0
    finished_median_ms: float = 0.0
    finished_p75_ms: float = 0.0
    filler_rate: float = 0.0
    typical_words: list[str] = field(default_factory=list)
    has_history: bool = False
    token_count: int = 0

    def render(self) -> str:
        """Render deterministic 1-line state representation, capped at 30 tokens."""
        if not self.has_history:
            return ""

        words_str = ", ".join(sorted(self.typical_words)[:3]) if self.typical_words else "none"
        line = (
            f'[Pace profile: resume_p50={self.resumed_median_ms:.0f}ms resume_p75={self.resumed_p75_ms:.0f}ms '
            f'finish_p50={self.finished_median_ms:.0f}ms finish_p75={self.finished_p75_ms:.0f}ms '
            f'filler={self.filler_rate:.2f} trail="{words_str}"]'
        )

        # Enforce maximum 30 tokens cap
        toks = re.findall(r"\w+|[^\w\s]", line)
        if len(toks) > 30:
            # Deterministically truncate text if it exceeds 30 tokens
            line = line[:100] + "...]"

        return line


class PaceProfileBuilder:
    """Extracts and maintains rolling pace profiles per user."""

    def __init__(self, history_data_path: Path | str | None = None) -> None:
        self.history_path = Path(history_data_path) if history_data_path else DEFAULT_HISTORY_PATH
        self._cache: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.history_path.exists():
            with open(self.history_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self._cache[entry["user_id"]] = entry
        self._loaded = True

    def build_profile(
        self,
        user_id: str,
        regrets: list[dict[str, Any]] | None = None,
    ) -> PaceProfile:
        """Build pace profile for a user from prior history sessions and regret events."""
        self._ensure_loaded()
        history_entry = self._cache.get(user_id)

        user_regrets = [r for r in (regrets or []) if r.get("user_id") == user_id]

        if not history_entry and not user_regrets:
            return PaceProfile(user_id=user_id, has_history=False, token_count=0)

        resumed_pauses: list[float] = []
        finished_pauses: list[float] = []
        filler_rate = 0.0
        typical_words: list[str] = []

        if history_entry:
            resumed_pauses.extend(float(p) for p in history_entry.get("resumed_pauses_ms", []))
            finished_pauses.extend(float(p) for p in history_entry.get("finished_pauses_ms", []))
            # Fallback for old history files without resumed/finished split
            if not resumed_pauses and not finished_pauses and "observed_pauses_ms" in history_entry:
                all_obs = history_entry["observed_pauses_ms"]
                resumed_pauses.extend(float(p) for p in all_obs[::2])
                finished_pauses.extend(float(p) for p in all_obs[1::2])

            filler_rate = float(history_entry.get("filler_rate", 0.0))
            typical_words = list(history_entry.get("typical_words", []))

        # Integrate regret events: a premature interruption at pause X counts as "resumed after X"
        for reg in user_regrets:
            silence_val = float(reg.get("silence_ms", 0.0))
            if silence_val > 0:
                resumed_pauses.append(silence_val)

        if not resumed_pauses and not finished_pauses:
            return PaceProfile(user_id=user_id, has_history=False, token_count=0)

        r_med = float(np.median(resumed_pauses)) if resumed_pauses else 0.0
        r_p75 = float(np.percentile(resumed_pauses, 75)) if resumed_pauses else 0.0
        f_med = float(np.median(finished_pauses)) if finished_pauses else 0.0
        f_p75 = float(np.percentile(finished_pauses, 75)) if finished_pauses else 0.0

        profile = PaceProfile(
            user_id=user_id,
            resumed_median_ms=round(r_med, 1),
            resumed_p75_ms=round(r_p75, 1),
            finished_median_ms=round(f_med, 1),
            finished_p75_ms=round(f_p75, 1),
            filler_rate=round(filler_rate, 2),
            typical_words=sorted(typical_words),
            has_history=True,
        )

        rendered = profile.render()
        profile.token_count = count_tokens(rendered)
        return profile
