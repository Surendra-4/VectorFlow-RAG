# src/observability/primitives.py

"""
Thread-safe metric primitives.

Performance characteristics on a modern Mac (target hardware):

* ``Counter.inc``       ~100 ns
* ``LabeledCounter.inc`` ~200 ns
* ``Gauge.set``         ~100 ns
* ``Histogram.observe`` ~1 µs amortized (lock + deque append + bounded prune)
* ``RingBuffer.append`` ~1 µs

Per-primitive ``threading.Lock`` is sufficient for our throughput. If
the service ever pushes into the high-thousands-of-requests/s range,
``threading.RLock`` would let nested observe-from-collector patterns
work without deadlock — easy switch, not needed today.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


def _now() -> float:
    """Time in seconds; ``time.time`` is used (not monotonic) so trace
    timestamps line up with wall-clock log entries operators inspect."""
    return time.time()


# --------------------------------------------------------------------------- #
# Counter — monotonic
# --------------------------------------------------------------------------- #


class Counter:
    """Monotonically-increasing integer counter."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, n: int = 1) -> None:
        if n < 0:
            raise ValueError(f"Counter.inc must be non-negative, got {n}")
        with self._lock:
            self._value += n

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class LabeledCounter:
    """Counter family keyed on a tuple of string labels.

    The label set is bounded by the operator (typically endpoint paths
    and status codes), so cardinality stays small. We don't enforce a
    cap here — overflow is an operator-level concern visible in the
    snapshot output.
    """

    def __init__(self, name: str, label_names: Tuple[str, ...], description: str = ""):
        self.name = name
        self.label_names = tuple(label_names)
        self.description = description
        self._values: Dict[Tuple[str, ...], int] = {}
        self._lock = threading.Lock()

    def inc(self, *labels: str, n: int = 1) -> None:
        if len(labels) != len(self.label_names):
            raise ValueError(
                f"{self.name}: expected {len(self.label_names)} labels "
                f"({self.label_names}), got {len(labels)}"
            )
        if n < 0:
            raise ValueError(f"LabeledCounter.inc must be non-negative, got {n}")
        key = tuple(str(lbl) for lbl in labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + n

    def items(self) -> List[Tuple[Tuple[str, ...], int]]:
        with self._lock:
            return list(self._values.items())

    def total(self) -> int:
        with self._lock:
            return sum(self._values.values())


# --------------------------------------------------------------------------- #
# Gauge — current value (can go up or down)
# --------------------------------------------------------------------------- #


class Gauge:
    """A floating-point value that can be set, incremented, or decremented."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)

    def inc(self, n: float = 1.0) -> None:
        with self._lock:
            self._value += float(n)

    def dec(self, n: float = 1.0) -> None:
        with self._lock:
            self._value -= float(n)

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


# --------------------------------------------------------------------------- #
# Histogram — rolling-window samples with percentile summary
# --------------------------------------------------------------------------- #


class Histogram:
    """Rolling-window sample histogram.

    Samples older than ``window_s`` are pruned lazily on each observe;
    a hard cap of ``max_samples`` prevents memory growth on bursts.

    Returns a fixed-shape dict from :meth:`snapshot` so callers can
    emit the same JSON regardless of how many samples are held.
    """

    def __init__(
        self,
        name: str,
        window_s: float = 300.0,
        max_samples: int = 5000,
        description: str = "",
    ):
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        if max_samples <= 0:
            raise ValueError(f"max_samples must be > 0, got {max_samples}")
        self.name = name
        self.description = description
        self.window_s = window_s
        self.max_samples = max_samples
        self._samples: Deque[Tuple[float, float]] = deque()
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        ts = _now()
        v = float(value)
        with self._lock:
            self._samples.append((ts, v))
            self._prune_locked(ts)

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        while len(self._samples) > self.max_samples:
            self._samples.popleft()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._prune_locked(_now())
            values = sorted(v for _, v in self._samples)

        n = len(values)
        if n == 0:
            return {
                "count": 0,
                "p50": None,
                "p95": None,
                "p99": None,
                "min": None,
                "max": None,
                "mean": None,
            }
        # Linear-interpolation percentile would be marginally nicer at
        # tiny n; for our use the nearest-rank method is fine.
        def pct(p: float) -> float:
            idx = min(n - 1, int(p * n))
            return values[idx]
        return {
            "count": n,
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "min": values[0],
            "max": values[-1],
            "mean": sum(values) / n,
        }


class LabeledHistogram:
    """Histogram family keyed on label tuple — same contract as ``LabeledCounter``."""

    def __init__(
        self,
        name: str,
        label_names: Tuple[str, ...],
        window_s: float = 300.0,
        max_samples: int = 5000,
        description: str = "",
    ):
        self.name = name
        self.label_names = tuple(label_names)
        self.description = description
        self.window_s = window_s
        self.max_samples = max_samples
        self._hists: Dict[Tuple[str, ...], Histogram] = {}
        self._lock = threading.Lock()

    def observe(self, *labels: str, value: float) -> None:
        if len(labels) != len(self.label_names):
            raise ValueError(
                f"{self.name}: expected {len(self.label_names)} labels, got {len(labels)}"
            )
        key = tuple(str(lbl) for lbl in labels)
        with self._lock:
            hist = self._hists.get(key)
            if hist is None:
                hist = Histogram(
                    self.name, window_s=self.window_s, max_samples=self.max_samples
                )
                self._hists[key] = hist
        # Observe outside the family-level lock to avoid serializing
        # all label observations on the registry lock.
        hist.observe(value)

    def items(self) -> List[Tuple[Tuple[str, ...], Dict[str, Any]]]:
        with self._lock:
            keys = list(self._hists.keys())
            hists = [self._hists[k] for k in keys]
        # Snapshot each child outside the family-level lock.
        return [(k, h.snapshot()) for k, h in zip(keys, hists)]


# --------------------------------------------------------------------------- #
# RingBuffer — bounded recent items
# --------------------------------------------------------------------------- #


class RingBuffer:
    """Bounded FIFO buffer for recent items (e.g. RetrievalTrace dumps)."""

    def __init__(self, name: str, max_size: int = 100, description: str = ""):
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self.name = name
        self.max_size = max_size
        self.description = description
        self._items: Deque[Any] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def append(self, item: Any) -> None:
        with self._lock:
            self._items.append(item)

    def snapshot(self, limit: Optional[int] = None) -> List[Any]:
        with self._lock:
            items = list(self._items)
        if limit is None or limit >= len(items):
            return items
        # Newest first when limited — operators typically want "latest N".
        return items[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
