"""
PERCLOS -- percentage of eyelid closure over the pupil.

Implements the actual Wierwille (1994) / NHTSA definition rather than the folk
version: P80 over a 60 s moving window, computed on a normalised openness scale
so that "80% closed" means 80% closed.

Two properties matter and both are easy to get wrong:

  1. The metric is about slow eyelid DROOPS, not blinks. Because a normal
     100-400 ms blink contributes only a few frames to a 60 s window, correctly
     implemented PERCLOS does not fire on ordinary blinking. That property is
     built into the metric; it is not something to bolt on afterwards.

  2. Invalid frames must be excluded from the numerator AND the denominator.
     Counting "no face detected" as "eyes open" is the common bug that makes
     PERCLOS silently under-report exactly when the driver has slumped out of
     frame.
"""

from dataclasses import dataclass

from .filters import TimeWindow


@dataclass
class PerclosResult:
    primary: float  # 60 s window -- the standards-compliant figure
    fast: float  # short window  -- demo responsiveness only
    validity: float  # fraction of frames in the primary window that were usable
    valid: bool  # False -> report MONITORING_UNAVAILABLE, not a number
    n_frames: int
    window_span_s: float
    warming_up: bool = False


class PerclosMeter:
    """
    Ring buffers of (timestamp, sample). Each sample is one of:
        1   -> valid frame, eye at or below P80 (closed)
        0   -> valid frame, eye above P80 (open)
        None-> invalid frame (no face / low confidence / head turned too far)

    Push every frame -- it is O(1). Decide at 1 Hz.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._primary = TimeWindow(cfg.primary_window_s)
        self._fast = TimeWindow(cfg.fast_window_s)

    def update(self, openness, valid, timestamp):
        if not valid or openness is None:
            sample = None
        else:
            sample = 1 if openness < self.cfg.perclos_p80_openness else 0
        self._primary.push(timestamp, sample)
        self._fast.push(timestamp, sample)
        return sample

    def _ratio(self, window, now):
        samples = window.values(now)
        if not samples:
            return 0.0, 0.0, 0
        usable = [s for s in samples if s is not None]
        validity = len(usable) / len(samples)
        if not usable:
            return 0.0, validity, 0
        return sum(usable) / len(usable), validity, len(usable)

    def compute(self, now):
        primary, validity, n = self._ratio(self._primary, now)
        fast, _, _ = self._ratio(self._fast, now)
        span = self._primary.span_s()

        # Three independent conditions must hold before this number means
        # anything. The span/sample checks are the cold-start guard: without
        # them the first frame gives a 1-sample ratio of 100% and the system
        # announces DROWSY the moment it launches.
        warm = span >= self.cfg.min_window_s and n >= self.cfg.min_samples
        usable = validity >= self.cfg.min_validity_ratio and n > 0

        return PerclosResult(
            primary=primary,
            fast=fast,
            validity=validity,
            valid=warm and usable,
            n_frames=n,
            window_span_s=span,
            warming_up=not warm,
        )

    def level(self, result):
        """'ok' | 'warn' | 'drowsy' | 'warming' | 'unavailable'."""
        if result.warming_up:
            return "warming"
        if not result.valid:
            return "unavailable"
        if result.primary >= self.cfg.drowsy_threshold:
            return "drowsy"
        if result.primary >= self.cfg.warn_threshold:
            return "warn"
        return "ok"

    def reset(self):
        self._primary.clear()
        self._fast.clear()


class PerclosMeterWithEyeConfig(PerclosMeter):
    """PerclosMeter needs the P80 threshold, which lives in EyeConfig rather
    than PerclosConfig. This adapter keeps both configs where they belong."""

    def __init__(self, perclos_cfg, eye_cfg):
        super().__init__(perclos_cfg)
        self.cfg = _MergedConfig(perclos_cfg, eye_cfg)


class _MergedConfig:
    def __init__(self, perclos_cfg, eye_cfg):
        self._p = perclos_cfg
        self.perclos_p80_openness = eye_cfg.perclos_p80_openness

    def __getattr__(self, name):
        return getattr(self._p, name)
