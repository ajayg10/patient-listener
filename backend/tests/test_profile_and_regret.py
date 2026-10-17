"""Unit tests for Pace Profile, State Builder profile formatting, and SQLite Regret Log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.pace_profile import PaceProfileBuilder, count_tokens
from backend.app.regret_log import list_regrets, log_regret, regret_stats
from backend.app.state_builder import StateBuilder

DATA_DIR = Path(__file__).resolve().parent.parent / "eval" / "data"


def test_profile_uses_only_history_sessions_zero_test_turn_leakage():
    """Verify that a test user's profile is built strictly from prior history sessions,

    with zero test-split turn events included in the profile observations.
    """
    builder = PaceProfileBuilder()
    builder._ensure_loaded()

    test_turns_path = DATA_DIR / "test_turns.jsonl"
    assert test_turns_path.exists(), "test_turns.jsonl must exist"

    test_turns: list[dict[str, Any]] = []
    with open(test_turns_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_turns.append(json.loads(line))

    # Collect test turn IDs and pauses
    test_turn_ids = set(t["id"] for t in test_turns)
    test_user_ids = sorted(list(set(t["user_id"] for t in test_turns)))
    assert len(test_user_ids) == 30, f"Expected 30 test users, got {len(test_user_ids)}"

    for uid in test_user_ids:
        prof = builder.build_profile(uid)
        assert prof.has_history is True
        assert prof.user_id == uid

        # Check raw history entry in builder cache
        entry = builder._cache[uid]
        assert "turn_id" not in entry
        # Assert no test turn ID is referenced anywhere in history
        for tid in test_turn_ids:
            assert tid not in str(entry), f"Test turn ID {tid} leaked into profile for user {uid}"


def test_cold_start_equals_base_state():
    """A user with no history gets an empty profile, and the state text is identical to base state."""
    builder = PaceProfileBuilder()
    cold_user = "brand_new_user_without_history"

    prof = builder.build_profile(cold_user)
    assert prof.has_history is False
    assert prof.render() == ""

    sb = StateBuilder()
    sb.append_transcript("I need to check my appointment status please")

    base_state = sb.build_state_text(silence_ms=500)
    profile_state = sb.build_state_text(silence_ms=500, pace_summary=prof.render())

    assert base_state == profile_state
    assert "Pace profile:" not in profile_state
    assert "[Pace" not in profile_state


def test_regret_events_shift_profile_in_expected_direction(tmp_path: Path):
    """Premature interruption regret events at pause X count as 'resumed after X' and increase resume median/p75."""
    db_file = tmp_path / "test_regrets.db"

    uid = "test_user_adaptive"
    # Create artificial history in a temp history file
    hist_file = tmp_path / "hist.jsonl"
    hist_entry = {
        "user_id": uid,
        "persona_name": "Fast Pauser",
        "archetype": "fast_pauser",
        "resumed_pauses_ms": [250, 300, 350],
        "finished_pauses_ms": [600, 700],
        "filler_rate": 0.05,
        "typical_words": ["please", "now"],
    }
    with open(hist_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(hist_entry) + "\n")

    builder = PaceProfileBuilder(history_data_path=hist_file)
    initial_prof = builder.build_profile(uid)
    assert initial_prof.resumed_median_ms == 300.0
    assert initial_prof.resumed_p75_ms == 325.0

    # Log two premature interruptions at 800ms and 1000ms
    log_regret(uid, "s1", 'Speaker: "..."', {"RESPOND": 0.9}, 800.0, action="RESPOND", db_path=db_file)
    log_regret(uid, "s2", 'Speaker: "..."', {"RESPOND": 0.85}, 1000.0, action="RESPOND", db_path=db_file)

    regrets = list_regrets(user_id=uid, db_path=db_file)
    assert len(regrets) == 2

    # Rebuild profile with regrets integrated
    shifted_prof = builder.build_profile(uid, regrets=regrets)
    # Resumed pauses now: [250, 300, 350, 800, 1000] -> median is 350, p75 is 800
    assert shifted_prof.resumed_median_ms > initial_prof.resumed_median_ms
    assert shifted_prof.resumed_p75_ms > initial_prof.resumed_p75_ms
    assert shifted_prof.resumed_median_ms == 350.0
    assert shifted_prof.resumed_p75_ms == 800.0


def test_profile_summary_is_deterministic():
    """Given the same history and regrets, rendered text is identical across invocations."""
    builder = PaceProfileBuilder()
    prof1 = builder.build_profile("user_fast_test_01")
    prof2 = builder.build_profile("user_fast_test_01")

    assert prof1.render() == prof2.render()
    assert prof1.token_count == prof2.token_count


def test_profile_line_under_token_ceiling():
    """Rendered profile line must strictly not exceed 30 tokens, and total state under 100 tokens."""
    builder = PaceProfileBuilder()
    builder._ensure_loaded()

    sb = StateBuilder()
    long_speech = (
        "Can I please verify whether my prescription for lisinopril is ready "
        "and if the physician approved the ninety day supply because I will be traveling "
        "next month for three weeks and cannot visit the pharmacy while out of town"
    )
    sb.append_transcript(long_speech)

    for uid in list(builder._cache.keys())[:15]:
        prof = builder.build_profile(uid)
        line = prof.render()
        if line:
            # Token cap for line alone is <= 30 tokens
            tok_count = count_tokens(line)
            assert tok_count <= 30, f"Profile line for {uid} has {tok_count} tokens > 30: {line}"

            state_text = sb.build_state_text(silence_ms=500, pace_summary=line)
            words = state_text.split()
            # 1 word ~ 1.3 tokens -> total words should stay under 70 words
            assert len(words) < 70, f"Total state words {len(words)} too large for {uid}"
            est_tokens = len(words) * 1.3
            assert est_tokens < 100, f"Estimated tokens {est_tokens:.1f} >= 100 ceiling for {uid}"


def test_sqlite_regret_log_crud_and_stats(tmp_path: Path):
    """Verify sqlite3 log_regret, list_regrets, and regret_stats functions."""
    db_file = tmp_path / "regrets_crud.db"

    # Initially empty
    assert len(list_regrets(db_path=db_file)) == 0
    stats = regret_stats(db_path=db_file)
    assert stats["total_regrets"] == 0

    # Log 3 regrets across two users
    row1 = log_regret("user_a", "sess_1", "state1", {"RESPOND": 0.8}, 500.0, db_path=db_file)
    row2 = log_regret("user_a", "sess_2", "state2", {"RESPOND": 0.9}, 700.0, db_path=db_file)
    row3 = log_regret("user_b", "sess_3", "state3", {"RESPOND": 0.75}, 300.0, db_path=db_file)

    assert row1 == 1
    assert row2 == 2
    assert row3 == 3

    # List all
    all_rows = list_regrets(db_path=db_file)
    assert len(all_rows) == 3

    # Filter user_a
    user_a_rows = list_regrets(user_id="user_a", db_path=db_file)
    assert len(user_a_rows) == 2
    assert user_a_rows[0]["silence_ms"] == 500.0
    assert user_a_rows[1]["silence_ms"] == 700.0
    assert user_a_rows[0]["probs"] == {"RESPOND": 0.8}

    # Stats for user_a
    st_a = regret_stats(user_id="user_a", db_path=db_file)
    assert st_a["total_regrets"] == 2
    assert st_a["median_silence_ms"] == 600.0
    assert st_a["min_silence_ms"] == 500.0
    assert st_a["max_silence_ms"] == 700.0
