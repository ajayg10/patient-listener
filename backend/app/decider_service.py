"""Decider service abstraction supporting Mock, Local, and Remote execution backends.

Adheres strictly to the Strands Decider API:
- `probabilities`: Softmax distribution across all options (sum to 1.0).
- `confidence`: Normalized concentration score ((N * p_max - 1) / (N - 1)), range [0.0, 1.0].
- `latency_ms`: Server-side decision compute time only.
- `round_trip_ms`: Network round-trip duration (reported separately for RemoteDecider).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import psutil
from strands_decider.schema import ChoiceQuestion, derive_confidence

from backend.app.config import AppConfig, get_config


@dataclass(frozen=True)
class Decision:
    """Standardized decision returned by all Decider backends."""

    choice: str
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float  # Server-side compute time only
    backend: str  # "mock" | "local" | "remote"
    round_trip_ms: float | None = None  # Network transmission round-trip (remote only)

    @property
    def probs(self) -> dict[str, float]:
        """Convenience alias for probabilities."""
        return self.probabilities


@runtime_checkable
class DeciderClient(Protocol):
    """Protocol defining the decider client interface."""

    def decide(self, state_text: str, question: ChoiceQuestion) -> Decision:
        """Evaluate conversational state against a structured choice question."""
        ...


class MockDecider:
    """Deterministic, rule-based decider for fast unit testing and virtual-clock benchmarks."""

    def __init__(self, fake_latency_ms: float = 0.0, simulate_sleep: bool = False) -> None:
        self.fake_latency_ms = fake_latency_ms
        self.simulate_sleep = simulate_sleep
        self.backend = "mock"

    def decide(self, state_text: str, question: ChoiceQuestion) -> Decision:
        if self.simulate_sleep and self.fake_latency_ms > 0:
            time.sleep(self.fake_latency_ms / 1000.0)

        mid_thought = self._detect_mid_thought(state_text)
        options = list(question.criteria.keys())

        # Determine choice based on question schema
        if "WAIT" in options and "RESPOND" in options:
            if mid_thought:
                choice = "WAIT"
                probabilities = {"WAIT": 0.88, "RESPOND": 0.12}
            else:
                choice = "RESPOND"
                probabilities = {"WAIT": 0.15, "RESPOND": 0.85}
        elif "YES" in options and "NO" in options:
            if mid_thought:
                choice = "YES"
                probabilities = {"YES": 0.88, "NO": 0.12}
            else:
                choice = "NO"
                probabilities = {"YES": 0.15, "NO": 0.85}
        else:
            # Fallback for arbitrary choices: option[0] if mid_thought else option[1]
            first, second = options[0], options[1]
            choice = first if mid_thought else second
            p_first = 0.88 if mid_thought else 0.15
            p_second = round(1.0 - p_first, 4)
            probabilities = {first: p_first, second: p_second}

        conf = round(derive_confidence(list(probabilities.values())), 4)

        return Decision(
            choice=choice,
            probabilities=probabilities,
            confidence=conf,
            latency_ms=self.fake_latency_ms,
            backend=self.backend,
        )

    @staticmethod
    def _detect_mid_thought(state_text: str) -> bool:
        """Deterministic heuristic inspecting fillers, trailing words, and silence."""
        clean = state_text.strip()

        # Parse silence duration if present
        silence_match = re.search(r"\[(?:Pause )?silence:\s*(\d+)\s*ms\]", clean, re.IGNORECASE)
        silence_ms = int(silence_match.group(1)) if silence_match else None

        # Extract spoken text from Speaker: "..." if available
        quote_match = re.search(r'Speaker:\s*"([^"]*)"', clean, re.IGNORECASE)
        speech = quote_match.group(1).strip() if quote_match else clean

        speech_lower = speech.lower().rstrip(".,?!")
        words = speech_lower.split()
        last_word = words[-1] if words else ""

        fillers = {"uh", "um", "er", "ah", "hmm", "like"}
        trailing_connectors = {
            "and", "or", "but", "because", "so", "to", "the", "a", "an",
            "that", "with", "if", "when", "for", "as", "at", "by", "of",
        }

        # Ellipsis indicates trailing hesitation
        if speech.endswith("...") or "..." in speech:
            return True

        if last_word in fillers or last_word in trailing_connectors:
            return True

        for filler in fillers:
            if f" {filler} " in f" {speech_lower} ":
                if silence_ms is not None and silence_ms < 900:
                    return True

        # Completed sentence with final punctuation and sufficient pause
        if speech.endswith((".", "?", "!")):
            return False

        # Short pause without completion cues implies still thinking
        if silence_ms is not None and silence_ms < 500:
            return True

        return False


class LocalDecider:
    """Local System One decider engine loaded directly in-process with a strict RAM gate."""

    RAM_GATE_BYTES = 2.0 * (1024**3)  # 2 GB required

    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or get_config()
        self.backend = "local"

        # 1. RAM Gate: Refuse to load if free RAM < 2 GB
        available_ram = psutil.virtual_memory().available
        if available_ram < self.RAM_GATE_BYTES:
            free_gb = available_ram / (1024**3)
            raise RuntimeError(
                f"Free RAM ({free_gb:.2f} GB) is below the required 2.0 GB gate. "
                "Refusing to load Decider model."
            )

        # 2. Ensure offline and cache environment before library imports
        os.environ["HF_HOME"] = self.cfg.hf_home
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        # 3. Configure PyTorch threads
        import torch
        from strands_decider.infer import load_engine

        torch.set_num_threads(self.cfg.decider_torch_threads)

        # 4. Load engine once at startup (resolve local snapshot path to allow offline loading)
        checkpoint_path = self._resolve_checkpoint(self.cfg.decider_model_id)
        self.engine = load_engine(
            checkpoint_path,
            device=self.cfg.decider_device,
        )

        # 5. One warm-up call
        from backend.app.questions import TURN_COMPLETE_QUESTION

        self.engine.ask("Speaker: \"Warmup probe\"\n[Silence: 500 ms]", {"warmup": TURN_COMPLETE_QUESTION})

    @staticmethod
    def _resolve_checkpoint(checkpoint_id: str) -> str:
        """Resolve cached snapshot directory to bypass network verification."""
        if os.path.isdir(checkpoint_id):
            return checkpoint_id
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(checkpoint_id, local_files_only=True)
        except Exception:
            return checkpoint_id

    def decide(self, state_text: str, question: ChoiceQuestion) -> Decision:
        t0 = time.perf_counter()
        resp = self.engine.ask(state_text, {"turn": question})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ans = resp.answers["turn"]
        return Decision(
            choice=ans.choice,
            probabilities=ans.probabilities,
            confidence=ans.confidence,
            latency_ms=round(latency_ms, 2),
            backend=self.backend,
        )


class RemoteDecider:
    """Remote Decider client calling an external HTTP server (e.g., Colab/Kaggle GPU endpoint)."""

    def __init__(self, url: str = "http://localhost:8000", timeout_s: float = 10.0) -> None:
        endpoint = url.rstrip("/")
        if not endpoint.endswith("/decide"):
            endpoint = f"{endpoint}/decide"
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.backend = "remote"

    def decide(self, state_text: str, question: ChoiceQuestion) -> Decision:
        payload = {
            "state": state_text,
            "question": {
                "type": question.type,
                "instructions": question.instructions,
                "criteria": question.criteria,
            },
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                rtt_ms = (time.perf_counter() - t0) * 1000.0
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to remote decider at {self.endpoint}: {e}") from e

        server_latency_ms = float(body.get("latency_ms", 0.0))
        return Decision(
            choice=body["choice"],
            probabilities=body["probabilities"],
            confidence=float(body["confidence"]),
            latency_ms=server_latency_ms,
            backend=self.backend,
            round_trip_ms=round(rtt_ms, 2),
        )


def get_decider_client(cfg: AppConfig | None = None) -> DeciderClient:
    """Factory creating a DeciderClient according to DECIDER_BACKEND."""
    config = cfg or get_config()
    backend = config.decider_backend.lower()

    if backend == "mock":
        return MockDecider()
    elif backend == "local":
        return LocalDecider(config)
    elif backend == "remote":
        return RemoteDecider(url=config.decider_url)
    else:
        raise ValueError(f"Unknown DECIDER_BACKEND: {backend!r}. Choose 'mock', 'local', or 'remote'.")
