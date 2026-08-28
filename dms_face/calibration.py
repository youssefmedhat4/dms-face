"""
Per-driver calibration.

Fixed EAR thresholds do not transfer between people -- published values range
0.20/0.25/0.30 precisely because no single number works for everyone, and the
fixed threshold specifically penalises people with smaller eyes.

We capture BOTH endpoints (open and closed), not just the open one as most
published calibration schemes do. Both are required, because P80 is defined on
a 0..1 opening scale and you cannot construct that scale from one endpoint.
"""

import json
import os
import statistics
import time
from dataclasses import dataclass, asdict, field

from .filters import TimeWindow


@dataclass
class DriverProfile:
    name: str = "default"
    ear_open: float = 0.30
    ear_closed: float = 0.10
    gaze_h0: float = 0.5
    gaze_v0: float = 0.5
    yaw0: float = 0.0
    pitch0: float = 0.0
    roll0: float = 0.0
    # Quality of this driver's eye signal. Recorded so that a marginal driver
    # is visible in the results rather than silently producing worse detection.
    separation: float = 0.0
    noise: float = 0.0
    snr: float = 0.0
    quality: str = "unknown"
    created: float = field(default_factory=time.time)
    notes: str = ""

    @property
    def is_default(self):
        return self.notes == "" and self.created == 0

    def save(self, directory="profiles"):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.name}.json")
        with open(path, "w", encoding="utf8") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, name, directory="profiles"):
        path = os.path.join(directory, f"{name}.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf8") as f:
            return cls(**json.load(f))

    @classmethod
    def load_or_default(cls, name, directory="profiles"):
        profile = cls.load(name, directory)
        if profile is not None:
            return profile, True
        return cls(name=name), False

    def describe(self):
        line = (
            f"profile '{self.name}': EAR open={self.ear_open:.4f} closed={self.ear_closed:.4f} "
            f"(range {self.ear_open - self.ear_closed:.4f})  "
            f"gaze centre=({self.gaze_h0:.3f}, {self.gaze_v0:.3f})  "
            f"neutral pose=({self.yaw0:.1f}, {self.pitch0:.1f}, {self.roll0:.1f})"
        )
        if self.snr:
            line += f"  signal quality={self.quality} (SNR {self.snr:.1f})"
        return line

    def quality_advice(self):
        """Human-readable guidance, or None when the signal is fine."""
        if self.quality == "good":
            return None
        if self.quality == "fair":
            return (
                f"Eye signal is FAIR (SNR {self.snr:.1f}). Usable, but expect occasional\n"
                "  flicker in the eye state. Improving the lighting or moving the camera\n"
                "  closer will widen the margin more than any threshold change will."
            )
        if self.quality == "poor":
            return (
                f"Eye signal is POOR (SNR {self.snr:.1f}): this driver's eye-opening range\n"
                f"  ({self.separation:.4f}) is small relative to landmark jitter ({self.noise:.4f}).\n"
                "  Blink and PERCLOS results for this driver will be unreliable, and that is\n"
                "  a property of the measurement, not a threshold that can be tuned around.\n"
                "  Try: better/more even lighting, a closer or higher-resolution camera,\n"
                "  removing glasses. If it stays poor, this driver needs an appearance-based\n"
                "  eye-state classifier rather than EAR geometry - see README limitations."
            )
        return None


class CalibrationCollector:
    """Accumulates samples for one phase and reduces them robustly.

    Uses the MEDIAN, never the mean: a single blink during the eyes-open phase
    would drag a mean downward and quietly compress the whole openness scale.
    """

    def __init__(self, settle_s=2.0):
        self.settle_s = settle_s
        self.ear = []
        self.h = []
        self.v = []
        self.yaw = []
        self.pitch = []
        self.roll = []
        self._t0 = None

    def start(self):
        self._t0 = time.monotonic()
        return self

    @property
    def elapsed(self):
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    @property
    def settling(self):
        return self.elapsed < self.settle_s

    def push(self, ear=None, h=None, v=None, yaw=None, pitch=None, roll=None):
        if self.settling:  # discard: people settle into position
            return False
        for target, value in (
            (self.ear, ear),
            (self.h, h),
            (self.v, v),
            (self.yaw, yaw),
            (self.pitch, pitch),
            (self.roll, roll),
        ):
            if value is not None:
                target.append(float(value))
        return True

    @staticmethod
    def _median(values, fallback=None):
        return statistics.median(values) if values else fallback

    @staticmethod
    def _mad(values):
        """Median absolute deviation -- a robust spread estimate.

        Robust matters here: during the eyes-open phase the driver is told to
        blink normally, so the samples contain genuine dips. A standard
        deviation would count those blinks as "noise" and massively overstate
        the jitter. The median and MAD both shrug them off.
        """
        if len(values) < 5:
            return None
        med = statistics.median(values)
        return statistics.median([abs(v - med) for v in values])

    def _open_state_noise(self):
        """Frame-to-frame jitter of the OPEN eye, with blinks excluded.

        Only samples at or above the median are used: below the median sits the
        blink population, which is signal, not noise.
        """
        if len(self.ear) < 10:
            return None
        med = statistics.median(self.ear)
        upper = [v for v in self.ear if v >= med]
        return self._mad(upper)

    def result(self):
        return {
            "ear": self._median(self.ear),
            "h": self._median(self.h),
            "v": self._median(self.v),
            "yaw": self._median(self.yaw),
            "pitch": self._median(self.pitch),
            "roll": self._median(self.roll),
            "n": len(self.ear),
            "noise": self._open_state_noise(),
            "mad": self._mad(self.ear),
        }


def assess_quality(separation, noise, hysteresis_band=0.10):
    """Rate how usable this driver's eye signal is.

    Separation alone is not enough. What decides whether the eye channel works
    for a given person is their open-to-closed range against their landmark
    jitter:

        SNR = (EAR_open - EAR_closed) / noise

    Two drivers can both clear a separation check and behave completely
    differently. A range of 0.21 with 0.005 jitter is clean. A range of 0.06
    with the same jitter is not -- once normalised onto the 0..1 openness
    scale, that noise is 8% of the entire range.

    The thresholds are derived, not invented. After normalisation the noise
    occupies 1/SNR of the openness scale, and it has to stay small compared to
    the hysteresis dead band (default 0.30 - 0.20 = 0.10), which is the only
    thing standing between jitter and a phantom blink:

        good : noise fits 4x inside the band  ->  SNR >= 4 / band  = 40
        fair : noise fits 2x inside the band  ->  SNR >= 2 / band  = 20
        poor : anything less

    This is where inter-person variation actually bites. Palpebral aperture
    varies substantially between individuals -- epicanthic folds, monolid
    eyelid structure, ptosis, deep-set eyes and heavy lashes all compress the
    range -- so a system that reports one number for everybody is concealing a
    real difference in how well it serves different drivers.
    """
    if noise is None or noise <= 1e-9:
        return "unknown", None
    snr = separation / noise
    if snr >= 4.0 / hysteresis_band:
        return "good", snr
    if snr >= 2.0 / hysteresis_band:
        return "fair", snr
    return "poor", snr


def build_profile(name, open_phase, closed_phase, min_separation=0.04):
    """Combine both phases into a profile, validating that they are distinct."""
    ear_open = open_phase["ear"]
    ear_closed = closed_phase["ear"]

    if ear_open is None or ear_closed is None:
        raise ValueError("calibration failed: no face detected during one of the phases")

    separation = ear_open - ear_closed
    if separation < min_separation:
        raise ValueError(
            f"calibration rejected: open EAR ({ear_open:.4f}) and closed EAR "
            f"({ear_closed:.4f}) are too close together (separation "
            f"{separation:.4f} < {min_separation}).\n"
            "Usually this means the eyes were not actually shut during phase 2, "
            "or the face was too far from the camera. Re-run the calibration."
        )

    noise = open_phase.get("noise")
    quality, snr = assess_quality(separation, noise)

    notes = f"open n={open_phase['n']}, closed n={closed_phase['n']}"
    if snr is not None:
        notes += f", separation={separation:.4f}, noise={noise:.4f}, snr={snr:.1f} ({quality})"

    return DriverProfile(
        name=name,
        ear_open=ear_open,
        ear_closed=ear_closed,
        gaze_h0=open_phase["h"] if open_phase["h"] is not None else 0.5,
        gaze_v0=open_phase["v"] if open_phase["v"] is not None else 0.5,
        yaw0=open_phase["yaw"] or 0.0,
        pitch0=open_phase["pitch"] or 0.0,
        roll0=open_phase["roll"] or 0.0,
        separation=separation,
        noise=noise if noise is not None else 0.0,
        snr=snr if snr is not None else 0.0,
        quality=quality,
        notes=notes,
    )


class BaselineAdapter:
    """Slow drift correction on the open-EAR baseline.

    Absorbs posture and lighting changes over a drive. The one rule that makes
    this safe: adaptation is FROZEN while the system is warning or drowsy.
    Without that freeze the baseline chases the drowsiness downward and the
    system quietly stops detecting the thing it was built to detect.
    """

    def __init__(self, cfg, profile):
        self.cfg = cfg
        self.profile = profile
        self._window = TimeWindow(cfg.adapt_window_s)
        self._last_update = 0.0
        self.adapted_ear_open = profile.ear_open
        self.frozen = False

    def push(self, ear, timestamp):
        if ear is not None:
            self._window.push(timestamp, ear)

    def maybe_adapt(self, timestamp, frozen):
        self.frozen = frozen
        if not self.cfg.adapt_enabled or frozen:
            return self.adapted_ear_open
        if timestamp - self._last_update < 1.0 / self.cfg.adapt_hz:
            return self.adapted_ear_open
        self._last_update = timestamp

        values = self._window.values(timestamp)
        if len(values) < 100:
            return self.adapted_ear_open

        values = sorted(values)
        idx = int(len(values) * self.cfg.adapt_percentile / 100.0)
        candidate = values[min(idx, len(values) - 1)]

        # Move gently, and never below the calibrated closed value plus a
        # margin -- a hard clamp against runaway drift.
        floor = self.profile.ear_closed + 0.05
        self.adapted_ear_open = max(floor, 0.9 * self.adapted_ear_open + 0.1 * candidate)
        return self.adapted_ear_open
