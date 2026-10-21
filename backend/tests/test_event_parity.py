"""Test type parity between backend Python event dataclasses and frontend TypeScript event interfaces."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVENTS_PY_PATH = REPO_ROOT / "backend" / "app" / "events.py"
EVENTS_TS_PATH = REPO_ROOT / "frontend" / "src" / "events.ts"


def get_backend_event_names() -> list[str]:
    """Parse backend/app/events.py and extract all defined event dataclass names."""
    with open(EVENTS_PY_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(EVENTS_PY_PATH))

    event_classes: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # Check for @dataclass decorator
            is_dataclass = any(
                (isinstance(d, ast.Name) and d.id == "dataclass")
                or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
                for d in node.decorator_list
            )
            if is_dataclass:
                event_classes.append(node.name)
    return event_classes


def test_events_ts_file_exists():
    assert EVENTS_TS_PATH.exists(), f"Expected {EVENTS_TS_PATH} to exist."


def test_event_type_parity():
    """Assert every dataclass in events.py appears as an interface/type in events.ts."""
    backend_events = get_backend_event_names()
    assert len(backend_events) >= 9, f"Expected at least 9 event classes in events.py, found {len(backend_events)}"

    ts_content = EVENTS_TS_PATH.read_text(encoding="utf-8")

    missing = []
    for event_name in backend_events:
        # Check that the interface or type definition exists in events.ts
        pattern = f"interface {event_name}"
        if pattern not in ts_content:
            missing.append(event_name)

    assert not missing, f"The following backend events are missing from {EVENTS_TS_PATH}: {missing}"


def test_turn_event_union_parity():
    """Assert TurnEvent union exists in events.ts and references all event types."""
    backend_events = get_backend_event_names()
    ts_content = EVENTS_TS_PATH.read_text(encoding="utf-8")

    assert "export type TurnEvent" in ts_content, "Expected 'export type TurnEvent' in events.ts"
    for event_name in backend_events:
        assert event_name in ts_content, f"Expected {event_name} to be referenced in events.ts"
