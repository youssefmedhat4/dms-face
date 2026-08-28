"""
The pipeline. One inference, then mathematics.

Per frame:
    landmark inference -> EAR (both eyes) -> openness -> median filter
                       -> blink FSM -> PERCLOS buffer
                       -> iris ratios -> gaze deviation
                       -> solvePnP -> yaw/pitch/roll -> EAR validity gate

At 1 Hz:
    PERCLOS decision, state fusion, baseline adaptation.
"""

import time
from dataclasses import dataclass, field

from . import metrics as M
from .blink import BlinkDetector, EyeState
from .calibration import BaselineAdapter, DriverProfile
from .filters import EMA, MedianFilter
from .headpose import HeadPoseEstimator, angles_from_transformation_matrix
from .landmarker import FaceLandmarkerWrapper
from .perclos import PerclosMeterWithEyeConfig
from .state import DriverState, StateMachine


@dataclass
class FrameResult:
    timestamp: float
    face_present: bool = False
    landmarks: object = None

    ear_raw: float = None
    ear_right: float = None
    ear_left: float = None
    ear_filtered: float = None
    openness: float = None
    openness_smooth: float = None

    eye_state: object = EyeState.UNKNOWN
    eye_channel_valid: bool = False
    closure_event: object = None

    angles: tuple = None
    angles_mp: tuple = None

    gaze_h: float = None
    gaze_v: float = None
    gaze_dev: float = None

    report: object = None
    inference_ms: float = 0.0
    total_ms: float = 0.0
    fps: float = 0.0
    stats: dict = field(default_factory=dict)


class FacePipeline:
    def __init__(self, cfg, profile=None, width=None, height=None, camera_calib=None):
        self.cfg = cfg
        self.width = width or cfg.camera.width
        self.height = height or cfg.camera.height
        self.profile = profile or DriverProfile()

        self.landmarker = FaceLandmarkerWrapper(cfg.landmarker, self.width, self.height)

        if camera_calib:
            self.headpose = HeadPoseEstimator.from_calibration_file(
                camera_calib, self.width, self.height
            )
        else:
            self.headpose = HeadPoseEstimator(self.width, self.height)

        self.ear_median = MedianFilter(cfg.eye.median_window)
        self.openness_ema = EMA(cfg.eye.ema_alpha)
        self.yaw_ema = EMA(cfg.head.ema_alpha)
        self.pitch_ema = EMA(cfg.head.ema_alpha)
        self.roll_ema = EMA(cfg.head.ema_alpha)
        self.gaze_ema_h = EMA(cfg.gaze.ema_alpha)
        self.gaze_ema_v = EMA(cfg.gaze.ema_alpha)

        self.blink = BlinkDetector(cfg.eye, cfg.blink)
        self.perclos = PerclosMeterWithEyeConfig(cfg.perclos, cfg.eye)
        self.machine = StateMachine(cfg)
        self.adapter = BaselineAdapter(cfg.calibration, self.profile)

        self._last_decision = 0.0
        self._last_report = None
        self._fps_ema = EMA(0.1)
        self._last_frame_time = None

        self.frames_processed = 0
        self.frames_with_face = 0

    # -- main entry ---------------------------------------------------------
    def process(self, frame_rgb, timestamp):
        t_start = time.perf_counter()
        result = FrameResult(timestamp=timestamp)

        if self._last_frame_time is not None:
            dt = timestamp - self._last_frame_time
            if dt > 1e-6:
                result.fps = self._fps_ema.update(1.0 / dt) or 0.0
        self._last_frame_time = timestamp

        t_inf = time.perf_counter()
        frame = self.landmarker.process(frame_rgb, timestamp)
        result.inference_ms = (time.perf_counter() - t_inf) * 1000.0

        self.frames_processed += 1

        if frame is None:
            self._handle_no_face(result, timestamp)
        else:
            self.frames_with_face += 1
            self._handle_face(result, frame, timestamp)

        result.total_ms = (time.perf_counter() - t_start) * 1000.0
        result.stats = {
            "frames": self.frames_processed,
            "with_face": self.frames_with_face,
            "detection_rate": self.frames_with_face / max(1, self.frames_processed),
            "ear_open": self.adapter.adapted_ear_open,
            "ear_closed": self.profile.ear_closed,
            "adaptation_frozen": self.adapter.frozen,
            "calibrated_camera": self.headpose.calibrated,
        }
        return result

    # -- branches -----------------------------------------------------------
    def _handle_no_face(self, result, timestamp):
        result.face_present = False
        self.ear_median.reset()
        self.headpose.reset()
        self.blink.update(None, valid=False, now=timestamp)
        # Push an INVALID sample, not a zero. Counting "no face" as "eyes open"
        # is the bug that makes PERCLOS under-report exactly when the driver
        # has slumped out of frame.
        self.perclos.update(None, valid=False, timestamp=timestamp)
        result.eye_state = self.blink.state
        self._decide(result, timestamp, face_present=False, eye_valid=False)

    def _handle_face(self, result, frame, timestamp):
        result.face_present = True
        result.landmarks = frame

        pts = frame.pixels

        # --- geometry branch (needed first: it gates the eye branch) --------
        angles = self.headpose.estimate(pts)
        if angles is not None:
            angles = (
                self.yaw_ema.update(angles[0]),
                self.pitch_ema.update(angles[1]),
                self.roll_ema.update(angles[2]),
            )
        result.angles = angles

        if frame.transform_matrix is not None:
            result.angles_mp = angles_from_transformation_matrix(frame.transform_matrix)

        # EAR is scale-invariant but NOT rotation-invariant. Past these angles
        # the measurement is not trustworthy, so we mark it invalid rather than
        # feeding a wrong number into PERCLOS.
        eye_valid = True
        if angles is not None:
            yaw, pitch, _ = angles
            if abs(yaw - self.profile.yaw0) > self.cfg.head.max_yaw_for_ear_deg:
                eye_valid = False
            if abs(pitch - self.profile.pitch0) > self.cfg.head.max_pitch_for_ear_deg:
                eye_valid = False
        result.eye_channel_valid = eye_valid

        # --- eye branch -----------------------------------------------------
        r_ear, l_ear = M.both_ear(pts)
        result.ear_right, result.ear_left = r_ear, l_ear
        result.ear_raw = M.combine_ear(r_ear, l_ear)
        result.ear_filtered = self.ear_median.update(result.ear_raw)

        ear_open = self.adapter.adapted_ear_open
        result.openness = M.openness(result.ear_filtered, ear_open, self.profile.ear_closed)
        result.openness_smooth = self.openness_ema.update(result.openness)

        # Blink FSM runs on the MEDIAN-filtered signal, never the EMA one:
        # an EMA time constant of 2-3 frames is a large fraction of a 100 ms
        # blink and would stretch the durations we classify on.
        result.closure_event = self.blink.update(result.openness, valid=eye_valid, now=timestamp)
        result.eye_state = self.blink.state

        self.perclos.update(result.openness, valid=eye_valid, timestamp=timestamp)
        self.adapter.push(result.ear_raw if eye_valid else None, timestamp)

        # --- iris branch ----------------------------------------------------
        h, v = M.iris_ratios(pts, frame.has_iris)
        if h is not None:
            h = self.gaze_ema_h.update(h)
            v = self.gaze_ema_v.update(v)
        result.gaze_h, result.gaze_v = h, v
        result.gaze_dev = M.gaze_deviation(h, v, self.profile.gaze_h0, self.profile.gaze_v0)

        self._decide(result, timestamp, face_present=True, eye_valid=eye_valid)

    def _decide(self, result, timestamp, face_present, eye_valid):
        """State fusion at 1 Hz -- re-deciding 30x/s adds flicker, not accuracy.

        The exception below matters: a microsleep in progress must escalate
        immediately, not wait up to a second for the next decision tick.
        """
        interval = 1.0 / self.cfg.perclos.decide_hz
        urgent = self.blink.ongoing_closure_ms(timestamp) >= self.cfg.blink.long_blink_max_ms

        if (timestamp - self._last_decision) < interval and not urgent:
            result.report = self._last_report
            return

        self._last_decision = timestamp
        perclos_result = self.perclos.compute(timestamp)
        level = self.perclos.level(perclos_result)

        report = self.machine.update(
            now=timestamp,
            perclos_result=perclos_result,
            perclos_level=level,
            blink_detector=self.blink,
            head_angles=result.angles,
            gaze_deviation=result.gaze_dev,
            eye_channel_valid=eye_valid,
            face_present=face_present,
            profile=self.profile,
        )
        self.adapter.maybe_adapt(timestamp, frozen=self.machine.should_freeze_adaptation)

        result.report = report
        self._last_report = report

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
