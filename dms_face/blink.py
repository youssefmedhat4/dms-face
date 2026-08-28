"""
Blink / closure finite state machine.

Durations are measured from WALL-CLOCK TIMESTAMPS, not frame counts. The
near-universal `EYE_AR_CONSEC_FRAMES = 16` pattern silently encodes a frame
rate: 16 frames is 0.53 s at 30 fps but 3.2 s at 5 fps. On a Pi under load the
frame rate moves, so a frame-count threshold changes meaning underneath you.
"""

import time
from dataclasses import dataclass
from enum import Enum

from .filters import Hysteresis, TimeWindow


class EyeState(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    OPENING = "opening"
    UNKNOWN = "unknown"  # eye channel invalid: no face, or head turned too far


class ClosureKind(Enum):
    BLINK = "blink"  # < 400 ms  -- normal, never alerts
    LONG_BLINK = "long_blink"  # 400-800 ms -- fatigue evidence
    PROLONGED = "prolonged"  # 800-1500 ms
    MICROSLEEP = "microsleep"  # > 1500 ms -- immediate alert


@dataclass
class ClosureEvent:
    kind: ClosureKind
    duration_ms: float
    start_time: float
    end_time: float


class BlinkDetector:
    """
    Four-state machine with hysteresis:

        OPEN --(openness < 0.20)--> CLOSING --(dwell met)--> CLOSED  [t_start]
          ^                                                    |
          +---------(openness > 0.30)---- OPENING <------------+     [emit]

    CLOSING exists so that a single noisy frame cannot open a closure event,
    and OPENING so that a flicker mid-closure cannot end one early.
    """

    def __init__(self, eye_cfg, blink_cfg):
        self.eye_cfg = eye_cfg
        self.blink_cfg = blink_cfg
        self.hysteresis = Hysteresis(eye_cfg.close_openness, eye_cfg.open_openness)

        self.state = EyeState.UNKNOWN
        self._closure_start = None
        self._candidate_start = None
        self._candidate_frames = 0

        self.current_closure_ms = 0.0
        self.last_event = None
        self.events = []
        self._blink_times = TimeWindow(blink_cfg.rate_window_s)

    def update(self, openness, valid=True, now=None):
        """Feed one frame. `openness` should be the MEDIAN-filtered signal.

        Returns a ClosureEvent when a closure completes, else None.
        """
        now = time.monotonic() if now is None else now

        if not valid or openness is None:
            # Do not guess. An invalid frame mid-closure aborts the measurement
            # rather than inventing a duration.
            self.state = EyeState.UNKNOWN
            self._closure_start = None
            self._candidate_start = None
            self._candidate_frames = 0
            self.current_closure_ms = 0.0
            return None

        is_low = self.hysteresis.update(openness)
        event = None

        if self.state in (EyeState.UNKNOWN, EyeState.OPEN, EyeState.OPENING):
            if is_low:
                if self._candidate_start is None:
                    self._candidate_start = now
                    self._candidate_frames = 0
                self._candidate_frames += 1
                elapsed_ms = (now - self._candidate_start) * 1000.0
                self.state = EyeState.CLOSING
                if (
                    self._candidate_frames >= self.eye_cfg.min_closure_frames
                    and elapsed_ms >= self.eye_cfg.min_closure_ms
                ):
                    self.state = EyeState.CLOSED
                    self._closure_start = self._candidate_start
            else:
                self.state = EyeState.OPEN
                self._candidate_start = None
                self._candidate_frames = 0
                self.current_closure_ms = 0.0

        elif self.state == EyeState.CLOSING:
            if is_low:
                self._candidate_frames += 1
                elapsed_ms = (now - self._candidate_start) * 1000.0
                if (
                    self._candidate_frames >= self.eye_cfg.min_closure_frames
                    and elapsed_ms >= self.eye_cfg.min_closure_ms
                ):
                    self.state = EyeState.CLOSED
                    self._closure_start = self._candidate_start
            else:
                # Too short to be a blink -- discard, do not count it.
                self.state = EyeState.OPEN
                self._candidate_start = None
                self._candidate_frames = 0

        elif self.state == EyeState.CLOSED:
            if is_low:
                self.current_closure_ms = (now - self._closure_start) * 1000.0
            else:
                duration_ms = (now - self._closure_start) * 1000.0
                event = ClosureEvent(
                    kind=classify_closure(duration_ms, self.blink_cfg),
                    duration_ms=duration_ms,
                    start_time=self._closure_start,
                    end_time=now,
                )
                self.last_event = event
                self.events.append(event)
                if event.kind is ClosureKind.BLINK:
                    self._blink_times.push(now, 1)
                self.state = EyeState.OPENING
                self._closure_start = None
                self._candidate_start = None
                self._candidate_frames = 0
                self.current_closure_ms = 0.0

        return event

    @property
    def is_closed(self):
        return self.state is EyeState.CLOSED

    def ongoing_closure_ms(self, now=None):
        """Duration of a closure still in progress -- this is what triggers the
        microsleep alert, since waiting for the eye to REOPEN before alerting
        would defeat the purpose."""
        if self.state is not EyeState.CLOSED or self._closure_start is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return (now - self._closure_start) * 1000.0

    def blink_rate_per_min(self, now=None):
        now = time.monotonic() if now is None else now
        n = len(self._blink_times.values(now))
        span = self._blink_times.span_s()
        if span < 5.0:  # too short a sample to extrapolate honestly
            return None
        return n * 60.0 / span

    def recent_mean_blink_ms(self, count=10):
        blinks = [e for e in self.events if e.kind is ClosureKind.BLINK][-count:]
        if not blinks:
            return None
        return sum(e.duration_ms for e in blinks) / len(blinks)

    def reset(self):
        self.state = EyeState.UNKNOWN
        self._closure_start = None
        self._candidate_start = None
        self._candidate_frames = 0
        self.current_closure_ms = 0.0
        self.hysteresis.reset()


def classify_closure(duration_ms, cfg):
    if duration_ms < cfg.normal_blink_max_ms:
        return ClosureKind.BLINK
    if duration_ms < cfg.long_blink_max_ms:
        return ClosureKind.LONG_BLINK
    if duration_ms < cfg.prolonged_max_ms:
        return ClosureKind.PROLONGED
    return ClosureKind.MICROSLEEP
