"""Unit tests for Decider service backends (Mock only, strictly adheres to Memory Guard)."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from backend.app.config import AppConfig
from backend.app.decider_service import (
    Decision,
    LocalDecider,
    MockDecider,
    RemoteDecider,
    get_decider_client,
)
from backend.app.questions import MID_THOUGHT_QUESTION, TURN_COMPLETE_QUESTION


def test_mock_decider_determinism():
    """MockDecider must produce deterministic, rule-based outputs across repeat calls."""
    decider = MockDecider(fake_latency_ms=10.0)

    # 1. Trailing connector indicates mid-thought -> WAIT
    prompt_wait = "Speaker: \"I would like to order the medication, but...\"\n[Silence: 400 ms]"
    res1 = decider.decide(prompt_wait, TURN_COMPLETE_QUESTION)
    res2 = decider.decide(prompt_wait, TURN_COMPLETE_QUESTION)

    assert res1.choice == "WAIT"
    assert res1 == res2
    assert res1.probabilities["WAIT"] > res1.probabilities["RESPOND"]

    # 2. Completed statement indicates turn yield -> RESPOND
    prompt_respond = "Speaker: \"Thank you for your help today.\"\n[Silence: 950 ms]"
    res3 = decider.decide(prompt_respond, TURN_COMPLETE_QUESTION)
    res4 = decider.decide(prompt_respond, TURN_COMPLETE_QUESTION)

    assert res3.choice == "RESPOND"
    assert res3 == res4
    assert res3.probabilities["RESPOND"] > res3.probabilities["WAIT"]

    # 3. Secondary Question: mid_thought -> YES vs NO
    res_q2_wait = decider.decide(prompt_wait, MID_THOUGHT_QUESTION)
    assert res_q2_wait.choice == "YES"

    res_q2_respond = decider.decide(prompt_respond, MID_THOUGHT_QUESTION)
    assert res_q2_respond.choice == "NO"


def test_mock_decider_probabilities_sum_to_one():
    """Probabilities returned by MockDecider must strictly sum to 1.0 within float tolerance."""
    decider = MockDecider()
    test_cases = [
        "Speaker: \"Um, let me check my calendar...\"\n[Silence: 300 ms]",
        "Speaker: \"Yes, that is correct.\"\n[Silence: 800 ms]",
        "Speaker: \"What is the co-pay for this prescription?\"\n[Silence: 850 ms]",
        "Speaker: \"I need to speak with a representative and...\"\n[Silence: 250 ms]",
    ]

    for text in test_cases:
        decision_q1 = decider.decide(text, TURN_COMPLETE_QUESTION)
        assert abs(sum(decision_q1.probabilities.values()) - 1.0) < 1e-4
        assert 0.0 <= decision_q1.confidence <= 1.0

        decision_q2 = decider.decide(text, MID_THOUGHT_QUESTION)
        assert abs(sum(decision_q2.probabilities.values()) - 1.0) < 1e-4
        assert 0.0 <= decision_q2.confidence <= 1.0


def test_mock_decider_latency_field_and_alias():
    """Decision must report server-side latency_ms and support the .probs alias."""
    decider = MockDecider(fake_latency_ms=42.5)
    decision = decider.decide("Speaker: \"Hello.\"\n[Silence: 600 ms]", TURN_COMPLETE_QUESTION)

    assert decision.latency_ms == 42.5
    assert decision.backend == "mock"
    assert decision.round_trip_ms is None
    # Verify .probs property alias matches .probabilities
    assert decision.probs == decision.probabilities


def test_remote_decider_with_stubbed_http_transport():
    """RemoteDecider must handle HTTP POST with no network, isolating server latency from RTT."""
    fake_body = {
        "choice": "WAIT",
        "probabilities": {"WAIT": 0.84, "RESPOND": 0.16},
        "confidence": 0.68,
        "latency_ms": 35.2,
        "backend": "local",
    }
    fake_response = io.BytesIO(json.dumps(fake_body).encode("utf-8"))

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response.getvalue()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        client = RemoteDecider(url="http://mock-gpu-host:8000/decide", timeout_s=5.0)
        decision = client.decide(
            state_text="Speaker: \"Wait a second...\"\n[Silence: 300 ms]",
            question=TURN_COMPLETE_QUESTION,
        )

        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        assert req.get_full_url() == "http://mock-gpu-host:8000/decide"
        assert req.get_method() == "POST"

        # Verify parsed decision
        assert isinstance(decision, Decision)
        assert decision.choice == "WAIT"
        assert decision.probabilities == {"WAIT": 0.84, "RESPOND": 0.16}
        assert decision.confidence == 0.68
        assert decision.latency_ms == 35.2  # Server-side compute only
        assert decision.backend == "remote"
        assert decision.round_trip_ms is not None
        assert decision.round_trip_ms >= 0.0


def test_local_decider_ram_gate_failure(monkeypatch):
    """LocalDecider must refuse to load and raise RuntimeError if free RAM < 2 GB."""
    fake_mem = MagicMock()
    fake_mem.available = int(1.2 * 1024**3)  # 1.2 GB available (< 2.0 GB gate)

    monkeypatch.setattr("psutil.virtual_memory", lambda: fake_mem)

    cfg = AppConfig(decider_backend="local")

    with pytest.raises(RuntimeError) as exc_info:
        LocalDecider(cfg=cfg)

    assert "below the required 2.0 GB gate" in str(exc_info.value)
    assert "Refusing to load Decider model" in str(exc_info.value)


def test_decider_factory_selection():
    """get_decider_client must instantiate the appropriate backend from AppConfig."""
    mock_client = get_decider_client(AppConfig(decider_backend="mock"))
    assert isinstance(mock_client, MockDecider)

    remote_client = get_decider_client(AppConfig(decider_backend="remote", decider_url="http://remote-decider:8000"))
    assert isinstance(remote_client, RemoteDecider)
    assert remote_client.endpoint == "http://remote-decider:8000/decide"
