"""Clock abstractions providing real-time wall clock and injectable virtual clock."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Protocol for timekeeping across live voice pipelines and virtual-clock evaluation."""

    def now_ms(self) -> float:
        """Return current timestamp in milliseconds."""
        ...

    def sleep(self, ms: float) -> None:
        """Pause or advance time by specified milliseconds."""
        ...


class SystemClock:
    """Real wall-clock using high-resolution performance counters."""

    def __init__(self) -> None:
        self._epoch = time.perf_counter()

    def now_ms(self) -> float:
        return (time.perf_counter() - self._epoch) * 1000.0

    def sleep(self, ms: float) -> None:
        if ms > 0:
            time.sleep(ms / 1000.0)


class VirtualClock:
    """Simulated virtual clock with deterministic time advancement for benchmarks and tests."""

    def __init__(self, start_ms: float = 0.0, simulated_decision_latency_ms: float = 0.0) -> None:
        self._current_ms = start_ms
        self.simulated_decision_latency_ms = simulated_decision_latency_ms

    def now_ms(self) -> float:
        return self._current_ms

    def advance_ms(self, delta_ms: float) -> float:
        """Explicitly advance clock by delta_ms and return the updated timestamp."""
        if delta_ms < 0:
            raise ValueError(f"Clock cannot move backward: {delta_ms} ms")
        self._current_ms += delta_ms
        return self._current_ms

    def set_time_ms(self, timestamp_ms: float) -> None:
        """Set absolute timestamp."""
        self._current_ms = timestamp_ms

    def sleep(self, ms: float) -> None:
        """Virtual sleep advances simulated time instantly without blocking wall-clock time."""
        if ms > 0:
            self._current_ms += ms
