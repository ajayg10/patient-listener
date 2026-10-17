"""SQLite3 regret log storing premature interruptions and turn decision regrets.

Uses standard library sqlite3 with raw SQL (no ORM).
Records events where the system prematurely interrupted a speaker (or executed an erroneous turn action).
In pace profile construction, each regret event at silence X counts as 'resumed after X'.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from backend.app.config import get_config


def get_default_db_path() -> Path:
    """Return default regret log database path from application config."""
    cfg = get_config()
    db_path = Path(cfg.regret_db_path)
    if not db_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent.parent
        db_path = repo_root / db_path
    return db_path


def init_db(db_path: str | Path | None = None) -> Path:
    """Initialize SQLite database schema with raw SQL."""
    path = Path(db_path) if db_path else get_default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS regrets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                state_text TEXT NOT NULL,
                probs TEXT NOT NULL,
                silence_ms REAL NOT NULL,
                action TEXT NOT NULL
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_regrets_user ON regrets(user_id);"
        )
        conn.commit()
    return path


def log_regret(
    user_id: str,
    session_id: str,
    state_text: str,
    probs: dict[str, float] | str,
    silence_ms: float,
    action: str = "RESPOND",
    ts: float | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Log a turn regret event (e.g. premature interruption) to SQLite database."""
    path = init_db(db_path)
    event_ts = ts if ts is not None else time.time()
    probs_str = json.dumps(probs) if isinstance(probs, dict) else str(probs)

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO regrets (ts, user_id, session_id, state_text, probs, silence_ms, action)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (event_ts, user_id, session_id, state_text, probs_str, float(silence_ms), action),
        )
        conn.commit()
        return cursor.lastrowid or 0


def list_regrets(
    user_id: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Query logged regrets, optionally filtered by user_id."""
    path = init_db(db_path)
    records: list[dict[str, Any]] = []

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT * FROM regrets WHERE user_id = ? ORDER BY ts ASC;",
                (user_id,),
            )
        else:
            cursor.execute("SELECT * FROM regrets ORDER BY ts ASC;")

        for row in cursor.fetchall():
            probs_raw = row["probs"]
            try:
                probs_val = json.loads(probs_raw)
            except Exception:
                probs_val = probs_raw

            records.append({
                "id": row["id"],
                "ts": row["ts"],
                "user_id": row["user_id"],
                "session_id": row["session_id"],
                "state_text": row["state_text"],
                "probs": probs_val,
                "silence_ms": row["silence_ms"],
                "action": row["action"],
            })
    return records


def regret_stats(
    user_id: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute summary statistics over recorded regrets."""
    regrets = list_regrets(user_id=user_id, db_path=db_path)
    if not regrets:
        return {
            "total_regrets": 0,
            "user_id": user_id,
            "median_silence_ms": 0.0,
            "min_silence_ms": 0.0,
            "max_silence_ms": 0.0,
        }

    silences = sorted(r["silence_ms"] for r in regrets)
    mid = len(silences) // 2
    median_val = (
        silences[mid]
        if len(silences) % 2 != 0
        else (silences[mid - 1] + silences[mid]) / 2.0
    )

    return {
        "total_regrets": len(regrets),
        "user_id": user_id,
        "median_silence_ms": round(median_val, 1),
        "min_silence_ms": round(silences[0], 1),
        "max_silence_ms": round(silences[-1], 1),
    }
