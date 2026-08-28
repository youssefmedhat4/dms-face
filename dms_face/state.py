"""
Driver-state fusion.

Combines four independent channels into one reported state:

    prolonged closure  (fast,  seconds)     -- microsleep, fires immediately
    PERCLOS            (slow,  60 s window) -- gradual fatigue
    blink character    (medium)             -- lengthening blinks
    head pose + gaze   (fast)               -- distraction

The channels are deliberately independent. PERCLOS is a lagging indicator by
construction and is weakest at early-stage sleepiness, so the closure and blink
channels exist to cover the window it misses. Conversely PERCLOS catches slow
drooping that never becomes a discrete "closure event".
"""

import time
from dataclasses import dataclass, field
from enum import Enum

from .blink import ClosureKind
from .filters import Debounce


class DriverState(Enum):
    UNAVAILABLE = "unavailable"  # cannot see the driver well enough to judge
    ALERT = "alert"
    DISTRACTED = "distracted"
    WARNING = "warning"
    DROWSY = "drowsy"

    @property
    def severity(self):
        return {
            "unavailable": 0,
            "alert": 0,
            "distracted": 2,
            "warning": 2,
            "drowsy": 3,
        }[self.value]


@dataclass
class StateReport:
    state: DriverState
    reasons: list = field(default_factory=list)
    perclos: float = 0.0
    perclos_fast: float = 0.0
    perclos_valid: bool = False
    perclos_warming: bool = False
    perclos_span: float = 0.0
    validity: float = 0.0
    closure_ms: float = 0.0
    blink_rate: float = None
    mean_blink_ms: float = None
    gaze_off_road_s: float = 0.0
    eye_channel_available: bool = True
    should_alert: bool = False

    def summary(self):
        head = self.state.value.upper()
        if self.reasons:
            return f"{head}: {'; '.join(self.reasons)}"
        return head


class StateMachine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = DriverState.UNAVAILABLE
        self._debounce = Debounce(cfg.state.alert_cooldown_s)
        self._off_road_since = None
        self._on_road_since = None
        self._distracted = False

    def update(
        self,
        *,
        now,
        perclos_result,
        perclos_level,
        blink_detector,
        head_angles,
        gaze_deviation,
        eye_channel_valid,
        face_present,
        profile,
    ):
        reasons = []
        report = StateReport(
            state=DriverState.UNAVAILABLE,
            perclos=perclos_result.primary,
            perclos_fast=perclos_result.fast,
            perclos_valid=perclos_result.valid,
            perclos_warming=perclos_result.warming_up,
            perclos_span=perclos_result.window_span_s,
            validity=perclos_result.validity,
            eye_channel_available=eye_channel_valid,
        )

        # ---- distraction channel: works even when the eyes do not ----------
        off_road = self._evaluate_gaze(now, head_angles, gaze_deviation, profile)
        report.gaze_off_road_s = (
            0.0 if self._off_road_since is None else now - self._off_road_since
        )

        # ---- eye channel ---------------------------------------------------
        closure_ms = blink_detector.ongoing_closure_ms(now)
        report.closure_ms = closure_ms
        report.blink_rate = blink_detector.blink_rate_per_min(now)
        report.mean_blink_ms = blink_detector.recent_mean_blink_ms()

        candidate = DriverState.ALERT

        if not face_present:
            # Short-circuit. Everything below this point is derived from the
            # eyes, and with no face there are no eyes to derive it from. Left
            # to fall through, a blink event from a few seconds ago would still
            # be "recent" and would escalate a no-face frame to WARNING --
            # reporting a driver state we demonstrably cannot observe.
            report.state = DriverState.UNAVAILABLE
            report.reasons = ["no face detected"]
            self.state = DriverState.UNAVAILABLE
            return report

        if not eye_channel_valid:
            # Degrade gracefully rather than guessing: head pose still works,
            # so distraction detection continues while drowsiness does not.
            candidate = DriverState.ALERT
            reasons.append("eye channel unavailable - head pose only")

        # Sanity guard, checked BEFORE the closure logic. A real microsleep
        # ends. An unbroken closure lasting tens of seconds means the openness
        # scale is wrong -- nearly always ear_open calibrated above the
        # driver's true open EAR, so every frame reads as closed.
        implausible = closure_ms >= self.cfg.state.implausible_closure_s * 1000.0
        if implausible:
            report.state = DriverState.UNAVAILABLE
            report.reasons = [
                f"implausible unbroken closure ({closure_ms / 1000.0:.0f}s) - "
                "profile is probably uncalibrated; run with --calibrate"
            ]
            report.closure_ms = closure_ms
            self.state = DriverState.UNAVAILABLE
            return report

        # Prolonged closure: fires on the ONGOING closure, not on the completed
        # event. Waiting for the eye to reopen before alerting would defeat the
        # entire purpose of detecting a microsleep.
        if eye_channel_valid and closure_ms >= self.cfg.blink.prolonged_max_ms:
            candidate = DriverState.DROWSY
            reasons.append(f"eyes closed {closure_ms / 1000.0:.1f}s")
        elif eye_channel_valid and closure_ms >= self.cfg.blink.long_blink_max_ms:
            candidate = max(candidate, DriverState.WARNING, key=lambda s: s.severity)
            reasons.append(f"prolonged closure {closure_ms:.0f}ms")

        # PERCLOS channel
        if perclos_level == "drowsy":
            candidate = DriverState.DROWSY
            reasons.append(f"PERCLOS {perclos_result.primary:.0%}")
        elif perclos_level == "warn":
            candidate = max(candidate, DriverState.WARNING, key=lambda s: s.severity)
            reasons.append(f"PERCLOS {perclos_result.primary:.0%}")
        elif perclos_level == "warming":
            reasons.append(f"PERCLOS warming up ({perclos_result.window_span_s:.0f}s)")
        elif perclos_level == "unavailable":
            reasons.append(f"PERCLOS unavailable (validity {perclos_result.validity:.0%})")

        # Blink-character channel: lengthening blinks precede overt closure.
        # Alert 265+/-57 ms vs sleep-deprived 586+/-592 ms in the literature.
        last = blink_detector.last_event
        if last is not None and (now - last.end_time) < 5.0:
            if last.kind is ClosureKind.MICROSLEEP:
                candidate = DriverState.DROWSY
                reasons.append(f"microsleep {last.duration_ms:.0f}ms")
            elif last.kind is ClosureKind.PROLONGED:
                candidate = max(candidate, DriverState.WARNING, key=lambda s: s.severity)
                reasons.append(f"prolonged blink {last.duration_ms:.0f}ms")

        mean_blink = report.mean_blink_ms
        if mean_blink is not None and mean_blink > self.cfg.blink.normal_blink_max_ms:
            candidate = max(candidate, DriverState.WARNING, key=lambda s: s.severity)
            reasons.append(f"mean blink {mean_blink:.0f}ms")

        # Distraction never outranks drowsiness: a driver asleep AND looking
        # away is asleep, and reporting "distracted" would understate it.
        if off_road and candidate.severity < DriverState.DROWSY.severity:
            if candidate is DriverState.ALERT or candidate is DriverState.UNAVAILABLE:
                candidate = DriverState.DISTRACTED
            reasons.append(f"looking away {report.gaze_off_road_s:.1f}s")

        report.state = candidate
        report.reasons = reasons
        self.state = candidate

        if candidate.severity >= 2 and self._debounce.allow(now):
            report.should_alert = True

        return report

    def _evaluate_gaze(self, now, head_angles, gaze_deviation, profile):
        """Head pose first, iris ratio second.

        Head pose catches the large deviations; the iris ratio catches the one
        head pose misses -- eyes down at a phone while the head stays forward.
        That single case is the whole reason to carry 478 landmarks rather
        than 468.

        NOTE both angles are taken RELATIVE to the driver's calibrated neutral.
        Absolute pitch is badly contaminated by the assumed focal length (see
        headpose.py); the difference from neutral is not.
        """
        looking_away = False

        if head_angles is not None:
            yaw, pitch, _ = head_angles
            d_yaw = abs(yaw - profile.yaw0)
            d_pitch = abs(pitch - profile.pitch0)
            if (
                d_yaw > self.cfg.head.distraction_yaw_deg
                or d_pitch > self.cfg.head.distraction_pitch_deg
            ):
                looking_away = True

        if not looking_away and gaze_deviation is not None:
            if gaze_deviation > self.cfg.gaze.deviation_threshold:
                looking_away = True

        if looking_away:
            self._on_road_since = None
            if self._off_road_since is None:
                self._off_road_since = now
        else:
            if self._on_road_since is None:
                self._on_road_since = now
            # Require a sustained look back before clearing, so a glance
            # through the mirror does not reset the timer.
            if (now - self._on_road_since) >= self.cfg.state.distracted_clear_s:
                self._off_road_since = None

        if self._off_road_since is None:
            return False
        return (now - self._off_road_since) >= self.cfg.gaze.off_road_alert_s

    @property
    def should_freeze_adaptation(self):
        """Baseline adaptation must not run while we believe the driver is
        impaired -- otherwise the reference chases the impairment."""
        return self.state.severity >= 2
