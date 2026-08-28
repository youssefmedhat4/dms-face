"""
Every tunable in one place.

Values are the starting points argued for in the architecture review. They are
not magic numbers: each carries the reasoning that produced it, because you will
have to defend them.
"""

from dataclasses import dataclass, asdict, field
import json


@dataclass
class EyeConfig:
    # Hysteresis (Schmitt trigger) on normalised openness, not raw EAR.
    # Two thresholds, not one: a single threshold makes the state chatter every
    # time the signal hovers near it, producing dozens of phantom blinks/minute.
    close_openness: float = 0.20  # openness must fall BELOW this to close
    open_openness: float = 0.30  # and rise ABOVE this to reopen

    # P80: "eyelid occludes the pupil more than 80%" (Wierwille 1994).
    # On the normalised openness scale that is openness < 0.20.
    perclos_p80_openness: float = 0.20

    # Reject sub-blink noise. Expressed in BOTH frames and milliseconds so the
    # behaviour does not silently change when frame rate drops under load.
    min_closure_frames: int = 2
    min_closure_ms: float = 60.0

    median_window: int = 3  # spike rejection, feeds the blink FSM
    ema_alpha: float = 0.4  # display + PERCLOS only, NEVER upstream of the FSM


@dataclass
class BlinkConfig:
    # Boundaries from the sleep-research literature (see review, Section 05):
    # normal blink 100-400 ms; alert total blink duration 265 +/- 57 ms;
    # sleep-deprived 586 +/- 592 ms; "extremely drowsy" closures > 2 s.
    normal_blink_max_ms: float = 400.0
    long_blink_max_ms: float = 800.0
    prolonged_max_ms: float = 1500.0
    # Beyond prolonged_max_ms -> microsleep -> immediate alert.
    # NOTE: deliberately 1.5 s, not the 2-3 s in the original project brief.
    # At 100 km/h, 2 s of closed eyes is 55 m of road travelled blind.

    # Blink rate is computed over this window (alert adults: 10-15/min).
    rate_window_s: float = 60.0


@dataclass
class PerclosConfig:
    # Primary window is the NHTSA/Wierwille standard. The short window exists
    # purely for demo responsiveness -- nobody wants to wait 60 s in a viva.
    primary_window_s: float = 60.0
    fast_window_s: float = 20.0

    warn_threshold: float = 0.15  # commonly cited operating point
    drowsy_threshold: float = 0.25

    # If fewer than this fraction of frames in the window are usable, report
    # MONITORING_UNAVAILABLE instead of a number. An honest "I cannot see you"
    # beats a confidently wrong 3%.
    min_validity_ratio: float = 0.60

    # Cold-start guard. PERCLOS is a RATIO over a window, so on the very first
    # frame a single closed sample reads as 100% and the system announces
    # DROWSY the instant it launches. The metric is only meaningful once the
    # window holds a representative span of time -- a driver blinks roughly
    # every 4-6 s, so under ~10 s we cannot distinguish "drowsy" from "caught
    # mid-blink at startup".
    min_window_s: float = 10.0
    min_samples: int = 60

    decide_hz: float = 1.0  # re-deciding 30x/s adds flicker and gains nothing


@dataclass
class HeadPoseConfig:
    # EAR is scale-invariant but NOT rotation-invariant: yaw foreshortens the
    # horizontal denominator (closed eye reads open), pitch-down shrinks the
    # vertical numerator (open eye reads closed -- the classic false alarm when
    # a driver glances at the gear stick). Past these angles we mark the eye
    # channel invalid rather than trusting it.
    max_yaw_for_ear_deg: float = 30.0
    max_pitch_for_ear_deg: float = 25.0

    # Distraction gate (relative to the driver's calibrated neutral pose).
    distraction_yaw_deg: float = 25.0
    distraction_pitch_deg: float = 20.0

    ema_alpha: float = 0.35  # angles are noisy; smooth for display and gating


@dataclass
class GazeConfig:
    # Binary on-road / off-road ONLY. Vora et al. measured head pose + landmark
    # geometry at 68.76% over 7 zones vs 95.18% for a CNN -- geometry is not
    # good enough for zone-level gaze, but the confusions it makes (windshield
    # vs speedometer) collapse away in a binary decision.
    deviation_threshold: float = 0.16  # normalised units inside the eye socket
    ema_alpha: float = 0.4
    off_road_alert_s: float = 2.0  # sustained off-road before alerting


@dataclass
class StateConfig:
    alert_cooldown_s: float = 5.0  # debounce; stops alarm storms
    distracted_clear_s: float = 1.0  # must look back this long to clear

    # Sanity guard. A real microsleep ENDS -- the eye reopens. An unbroken
    # "closure" lasting this long is not a driver state, it is a broken
    # openness scale: almost always an uncalibrated or stale profile whose
    # ear_open sits above the driver's actual open EAR. Reporting DROWSY in
    # that situation is confidently wrong, which is worse than reporting
    # nothing, so we downgrade to UNAVAILABLE and say why.
    implausible_closure_s: float = 30.0


@dataclass
class CameraConfig:
    width: int = 640  # models resize to 192x192 / 256x256 internally --
    height: int = 480  # feeding more than this is wasted work
    fps: int = 30
    index: int = 0  # OpenCV device index (dev machine only)
    flip_horizontal: bool = True  # mirror view is more natural to sit in front of
    ir_mode: bool = False  # apply CLAHE; IR cabin images are low-contrast


@dataclass
class LandmarkerConfig:
    model_path: str = "models/face_landmarker.task"
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    output_facial_transformation_matrixes: bool = True  # free head-pose cross-check
    output_face_blendshapes: bool = False  # costs compute; we use EAR instead


@dataclass
class CalibrationConfig:
    open_phase_s: float = 10.0
    closed_phase_s: float = 3.0
    settle_s: float = 2.0  # discard the first samples; people settle in
    min_samples: int = 30

    # Slow adaptation absorbs posture and lighting drift, but is FROZEN during
    # WARNING/DROWSY -- otherwise the baseline chases the drowsiness downward
    # and the system quietly stops detecting it.
    adapt_enabled: bool = True
    adapt_window_s: float = 300.0
    adapt_percentile: float = 95.0
    adapt_hz: float = 0.1


@dataclass
class Config:
    eye: EyeConfig = field(default_factory=EyeConfig)
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    perclos: PerclosConfig = field(default_factory=PerclosConfig)
    head: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    state: StateConfig = field(default_factory=StateConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    landmarker: LandmarkerConfig = field(default_factory=LandmarkerConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def to_json(self, path):
        with open(path, "w", encoding="utf8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf8") as f:
            raw = json.load(f)
        return cls(
            eye=EyeConfig(**raw.get("eye", {})),
            blink=BlinkConfig(**raw.get("blink", {})),
            perclos=PerclosConfig(**raw.get("perclos", {})),
            head=HeadPoseConfig(**raw.get("head", {})),
            gaze=GazeConfig(**raw.get("gaze", {})),
            state=StateConfig(**raw.get("state", {})),
            camera=CameraConfig(**raw.get("camera", {})),
            landmarker=LandmarkerConfig(**raw.get("landmarker", {})),
            calibration=CalibrationConfig(**raw.get("calibration", {})),
        )
