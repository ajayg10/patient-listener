"""Integration tests for Patient Listener FastAPI backend API.

Tests (Mock only, no real weights, no network):
- Scripted WS session with decisions and streamed replies
- Regret submission followed by dynamic pace profile updates
- Side-by-side replay with both engines at high speed
- Strict cache miss error event (zero fallback)
- Reply agent fallback flag (reply_source: 'mock')
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.reply_agent import ReplyAgent


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "cache_provenance" in data
    assert "bedrock_status" in data
    assert data["bedrock_status"]["status"] == "SKIPPED"


def test_results_endpoint(client: TestClient) -> None:
    resp = client.get("/results")
    assert resp.status_code == 200
    data = resp.json()
    assert "provenance_flag" in data
    assert data["provenance_flag"] in ("mock", "real")


def test_scenarios_endpoint(client: TestClient) -> None:
    resp = client.get("/scenarios")
    assert resp.status_code == 200
    scenarios = resp.json()
    assert isinstance(scenarios, list)
    assert len(scenarios) > 0
    s0 = scenarios[0]
    assert "id" in s0
    assert "user" in s0
    assert "persona" in s0
    assert "text_preview" in s0
    assert "true_end_time" in s0


def test_regret_and_profile_dynamic_shift(client: TestClient) -> None:
    import uuid

    user_id = f"user_test_dyn_{uuid.uuid4().hex[:8]}"

    # Profile before any regrets
    p0 = client.get(f"/profile/{user_id}").json()
    assert p0["user_id"] == user_id
    assert p0["has_history"] is False
    assert p0["resumed_median_ms"] == 0.0

    # Post regret event at 1350ms silence
    regret_payload = {
        "user_id": user_id,
        "session_id": "sess_dyn_01",
        "state_text": 'Speaker: "Please schedule an appointment"\n[Silence: 1350 ms]',
        "probs": {"WAIT": 0.2, "RESPOND": 0.8},
        "silence_ms": 1350.0,
        "action": "RESPOND",
    }
    r_resp = client.post("/regret", json=regret_payload)
    assert r_resp.status_code == 200
    r_data = r_resp.json()
    assert r_data["status"] == "ok"
    assert r_data["regret_id"] > 0

    # Profile after regret should update dynamically
    p1 = client.get(f"/profile/{user_id}").json()
    assert p1["has_history"] is True
    assert p1["resumed_median_ms"] == 1350.0
    assert p1["regrets_count"] >= 1
    assert "1350ms" in p1["rendered"]


def test_scripted_ws_session_and_reply_stream(client: TestClient) -> None:
    """Test scripted WebSocket turn-taking session with streaming response."""
    with client.websocket_connect("/ws/session?user=default_user&ticker=false") as ws:
        # 1. Speech onset
        ws.send_json({"type": "SpeechStarted", "timestamp_ms": 0.0})

        # 2. Utterance transcription
        ws.send_json({
            "type": "SpeechDelta",
            "text": "Can I get a refill on my blood pressure medication please?",
            "timestamp_ms": 150.0,
        })

        # 3. Speech ends (pause begins)
        ws.send_json({"type": "SpeechEnded", "timestamp_ms": 2500.0})

        # 4. Silence checkpoint tick at 250ms silence (2500 + 250 = 2750ms)
        ws.send_json({"type": "tick", "timestamp_ms": 2750.0})

        # Collect server messages
        received: list[dict] = []
        has_decision = False
        has_reply_chunk = False
        has_reply_done = False

        for _ in range(25):
            msg = ws.receive_json()
            received.append(msg)
            m_type = msg.get("type")

            if m_type == "decision":
                has_decision = True
                assert "probs" in msg
                assert "p_done" in msg
                assert "latency_ms" in msg
                assert "action" in msg
                assert "checkpoint" in msg
                assert msg["checkpoint"] == 250

            elif m_type == "reply_chunk":
                has_reply_chunk = True
                assert "chunk" in msg
                assert msg.get("reply_source") in ("mock", "bedrock")

            elif m_type == "reply_done":
                has_reply_done = True
                assert "full_reply" in msg
                assert msg.get("reply_source") in ("mock", "bedrock")
                break

        assert has_decision, f"Expected decision event, got: {received}"
        assert has_reply_chunk, f"Expected reply chunks, got: {received}"
        assert has_reply_done, f"Expected reply done, got: {received}"


def test_reply_fallback_flag(client: TestClient) -> None:
    """Test that when live Bedrock errors, system falls back to mock reply and sets reply_source: mock."""
    with patch.object(ReplyAgent, "stream_reply") as mock_stream:
        # Simulate live Bedrock exception during streaming
        async def failing_stream(prompt: str):
            raise RuntimeError("Simulated Bedrock client error")
            yield "test"  # pragma: no cover

        mock_stream.side_effect = failing_stream

        with client.websocket_connect("/ws/session?user=default_user&ticker=false") as ws:
            ws.send_json({"type": "SpeechStarted", "timestamp_ms": 0.0})
            ws.send_json({
                "type": "SpeechDelta",
                "text": "Can I get a refill on my blood pressure medication please?",
                "timestamp_ms": 100.0,
            })
            ws.send_json({"type": "SpeechEnded", "timestamp_ms": 1000.0})
            ws.send_json({"type": "tick", "timestamp_ms": 1250.0})

            fallback_flag_observed = False
            for _ in range(25):
                msg = ws.receive_json()
                if msg.get("type") in ("reply_chunk", "reply_done"):
                    assert msg.get("reply_source") == "mock"
                    fallback_flag_observed = True
                    if msg.get("type") == "reply_done":
                        break

            assert fallback_flag_observed


def test_replay_both_engines_high_speed(client: TestClient) -> None:
    """Test POST /replay/{scenario} streams both engines at high speed with correct final metrics."""
    # Use complete scenario that exists in cache
    scenario_id = "turn_user_fast_test_01_tmpl_test_04_004"
    resp = client.post(f"/replay/{scenario_id}?speed=1000")
    assert resp.status_code == 200

    raw_lines = [line.strip() for line in resp.text.split("\n\n") if line.strip()]
    events = [json.loads(line[len("data: "):]) for line in raw_lines if line.startswith("data: ")]

    assert len(events) >= 5

    # First event must be metadata with cache provenance and latency note
    meta = events[0]
    assert meta["event"] == "metadata"
    assert meta["engine"] == "system"
    assert "provenance" in meta
    assert "latency measured on Tesla T4, decisions from cache" in meta["provenance"]["note"]

    # Verify both engines emit events
    engines_seen = {ev.get("engine") for ev in events}
    assert "fixed" in engines_seen
    assert "patient_listener" in engines_seen

    # Verify final events per engine
    final_fixed = [ev for ev in events if ev.get("engine") == "fixed" and ev.get("event") == "final"]
    final_pl = [ev for ev in events if ev.get("engine") == "patient_listener" and ev.get("event") == "final"]

    assert len(final_fixed) == 1
    assert "interrupted" in final_fixed[0]
    assert "response_delay_ms" in final_fixed[0]
    assert final_fixed[0]["interrupted"] is False

    assert len(final_pl) == 1
    assert "interrupted" in final_pl[0]
    assert "response_delay_ms" in final_pl[0]
    assert final_pl[0]["interrupted"] is False


def test_replay_cache_miss_error_event(client: TestClient) -> None:
    """Test that a cache miss returns a clear error event and never falls back."""
    scenario_id = "turn_user_slow_test_01_tmpl_test_01_121"
    with patch("backend.app.main.load_cache", return_value={}):
        resp = client.post(f"/replay/{scenario_id}?speed=1000")
        assert resp.status_code == 200

        raw_lines = [line.strip() for line in resp.text.split("\n\n") if line.strip()]
        events = [json.loads(line[len("data: "):]) for line in raw_lines if line.startswith("data: ")]

        # Check for the clear error event
        error_events = [ev for ev in events if ev.get("event") == "error"]
        assert len(error_events) >= 1
        err = error_events[0]
        assert err["engine"] == "patient_listener"
        assert err["error"] == "cache_miss"
        assert "Cache miss for state key" in err["message"]


def test_replay_unknown_scenario_returns_404(client: TestClient) -> None:
    resp = client.post("/replay/unknown_scenario_id_9999?speed=1000")
    assert resp.status_code == 404


def test_plots_endpoint(client: TestClient) -> None:
    """Test that whitelisted plots are served and unknown plot names return 404."""
    resp1 = client.get("/plots/results_pareto.png")
    assert resp1.status_code == 200
    assert resp1.headers["content-type"] == "image/png"

    resp2 = client.get("/plots/results_reliability.png")
    assert resp2.status_code == 200
    assert resp2.headers["content-type"] == "image/png"

    resp_bad = client.get("/plots/malicious_or_unknown.png")
    assert resp_bad.status_code == 404

