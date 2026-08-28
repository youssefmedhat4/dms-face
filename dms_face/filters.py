"""
Temporal filtering primitives.

Applied in a deliberate order (see review, Section 10):
    median(3) -> hysteresis -> minimum dwell -> debounce

with EMA branching off for display and PERCLOS only. The blink FSM runs on the
MEDIAN-filtered signal, never the EMA one: an EMA with alpha=0.4 has a time
constant of 2-3 frames, a meaningful fraction of a 100 ms blink, so smoothing
before the FSM would systematically stretch the very durations we classify on.
"""

import time
from collections import deque


class MedianFilter:
    """Odd-length running median. Kills single-frame landmark spikes with no
    lag on genuine transitions -- unlike a mean, which smears them."""

    def __init__(self, window=3):
        if window % 2 == 0:
            raise ValueError("median window must be odd")
        self.window = window
        self._buf = deque(maxlen=window)

    def update(self, value):
        if value is None:
            return None
        self._buf.append(float(value))
        ordered = sorted(self._buf)
        return ordered[len(ordered) // 2]

    def reset(self):
        self._buf.clear()


class EMA:
    """Exponential moving average. Display and PERCLOS only."""

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.value = None

    def update(self, value):
        if value is None:
            return self.value
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * float(value) + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class Hysteresis:
    """Schmitt trigger.

    Enters the 'low' state only when the signal falls below `low_threshold`,
    and leaves it only when the signal rises above `high_threshold`. With a
    single threshold, a signal resting near it toggles on sensor noise.
    """

    def __init__(self, low_threshold, high_threshold, start_low=False):
        if low_threshold >= high_threshold:
            raise ValueError("low_threshold must be < high_threshold")
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.is_low = start_low

    def update(self, value):
        """Returns True while in the 'low' (eye-closed) state."""
        if value is None:
            return self.is_low
        if self.is_low:
            if value > self.high_threshold:
                self.is_low = False
        else:
            if value < self.low_threshold:
                self.is_low = True
        return self.is_low

    def reset(self, start_low=False):
        self.is_low = start_low


class Debounce:
    """Rate-limits an event to at most once per `cooldown_s`."""

    def __init__(self, cooldown_s=5.0):
        self.cooldown_s = cooldown_s
        self._last = None

    def allow(self, now=None):
        now = time.monotonic() if now is None else now
        if self._last is None or (now - self._last) >= self.cooldown_s:
            self._last = now
            return True
        return False

    def remaining(self, now=None):
        now = time.monotonic() if now is None else now
        if self._last is None:
            return 0.0
        return max(0.0, self.cooldown_s - (now - self._last))

    def reset(self):
        self._last = None


class TimeWindow:
    """Sliding time window of (timestamp, value) pairs.

    Backs PERCLOS, blink rate and baseline adaptation. Eviction is amortised
    O(1) per sample, so updating every frame costs nothing measurable.
    """

    def __init__(self, window_s):
        self.window_s = window_s
        self._items = deque()

    def push(self, timestamp, value):
        self._items.append((timestamp, value))
        self._evict(timestamp)

    def _evict(self, now):
        cutoff = now - self.window_s
        while self._items and self._items[0][0] < cutoff:
            self._items.popleft()

    def values(self, now=None):
        if now is not None:
            self._evict(now)
        return [v for _, v in self._items]

    def span_s(self):
        if len(self._items) < 2:
            return 0.0
        return self._items[-1][0] - self._items[0][0]

    def __len__(self):
        return len(self._items)

    def clear(self):
        self._items.clear()
