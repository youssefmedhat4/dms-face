"""
Unit tests for the deterministic half of the pipeline.

These matter more than usual here. You cannot ethically induce real drowsiness
in a test subject, so end-to-end "drowsiness accuracy" is not measurable. What
IS measurable is whether each component computes what it claims to compute --
so that is what we test, with synthetic signals where ground truth is exact.

Run:  python -m pytest tests/ -v      (or: python tests/test_core.py)
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dms_face import metrics as M
from dms_face.blink import BlinkDetector, ClosureKind, EyeState
from dms_face.calibration import build_profile
from dms_face.config import BlinkConfig, EyeConfig, PerclosConfig
from dms_face.filters import Debounce, EMA, Hysteresis, MedianFilter, TimeWindow
from dms_face.headpose import normalise_angles
from dms_face.perclos import PerclosMeterWithEyeConfig


# --------------------------------------------------------------- geometry --
def synthetic_eye(width=40.0, height=10.0, cx=100.0, cy=100.0):
    """Six landmarks in EAR order for an idealised eye of known dimensions.

    With both lid pairs at the full height, EAR = (h + h) / (2 * w) = h / w,
    so the expected value is exact and independent of the implementation.
    """
    half_w, half_h = width / 2.0, height / 2.0
    return np.array(
        [
            [cx - half_w, cy],  # p1 outer
            [cx - half_w / 2, cy - half_h],  # p2 upper-outer
            [cx + half_w / 2, cy - half_h],  # p3 upper-inner
            [cx + half_w, cy],  # p4 inner
            [cx + half_w / 2, cy + half_h],  # p5 lower-inner
            [cx - half_w / 2, cy + half_h],  # p6 lower-outer
        ]
    )


def test_ear_matches_closed_form():
    pts = synthetic_eye(width=40.0, height=10.0)
    ear = M.eye_aspect_ratio(pts, (0, 1, 2, 3, 4, 5))
    assert abs(ear - 0.25) < 1e-9, ear


def test_ear_is_scale_invariant():
    """The defining property: distance from the camera must not change EAR."""
    small = M.eye_aspect_ratio(synthetic_eye(40.0, 10.0), (0, 1, 2, 3, 4, 5))
    large = M.eye_aspect_ratio(synthetic_eye(120.0, 30.0), (0, 1, 2, 3, 4, 5))
    assert abs(small - large) < 1e-9


def test_ear_is_not_rotation_invariant():
    """Documents the known failure mode we gate against.

    Yaw foreshortens the horizontal axis while the lids keep their separation,
    so EAR RISES -- a closed eye can read as open. This is why the pipeline
    marks the eye channel invalid past 30 degrees rather than trusting it.
    """
    frontal = synthetic_eye(40.0, 10.0)
    yawed = frontal.copy()
    yawed[:, 0] = 100.0 + (yawed[:, 0] - 100.0) * math.cos(math.radians(45))

    ear_frontal = M.eye_aspect_ratio(frontal, (0, 1, 2, 3, 4, 5))
    ear_yawed = M.eye_aspect_ratio(yawed, (0, 1, 2, 3, 4, 5))
    assert ear_yawed > ear_frontal * 1.3, (ear_frontal, ear_yawed)


def test_degenerate_eye_returns_none():
    pts = synthetic_eye(width=0.0, height=10.0)
    assert M.eye_aspect_ratio(pts, (0, 1, 2, 3, 4, 5)) is None


def test_combine_ear_uses_mean_not_min():
    """Taking the minimum would let one jittery landmark drive a false closure."""
    assert abs(M.combine_ear(0.30, 0.10) - 0.20) < 1e-12
    assert abs(M.combine_ear(0.30, None) - 0.30) < 1e-12
    assert M.combine_ear(None, None) is None


def test_openness_normalisation():
    assert abs(M.openness(0.30, 0.30, 0.10) - 1.0) < 1e-9
    assert abs(M.openness(0.10, 0.30, 0.10) - 0.0) < 1e-9
    assert abs(M.openness(0.20, 0.30, 0.10) - 0.5) < 1e-9
    # Clamped, so a driver opening wider than baseline cannot exceed 1.0
    assert M.openness(0.40, 0.30, 0.10) == 1.0
    assert M.openness(0.05, 0.30, 0.10) == 0.0
    # Degenerate calibration must not divide by zero
    assert M.openness(0.2, 0.2, 0.2) is None


# ---------------------------------------------------------------- filters --
def test_median_rejects_single_frame_spike():
    f = MedianFilter(3)
    f.update(0.30)
    f.update(0.30)
    assert abs(f.update(0.02) - 0.30) < 1e-9  # spike absorbed entirely


def test_median_still_follows_a_real_transition():
    f = MedianFilter(3)
    for v in (0.30, 0.30):
        f.update(v)
    f.update(0.02)
    assert f.update(0.02) < 0.1  # two samples is enough to move


def test_hysteresis_prevents_chatter():
    """A signal resting between the thresholds must not toggle."""
    h = Hysteresis(0.20, 0.30)
    assert h.update(0.50) is False
    assert h.update(0.19) is True  # crossed the low threshold -> closed
    for v in (0.21, 0.25, 0.29, 0.22):  # in the dead band
        assert h.update(v) is True, v  # stays closed
    assert h.update(0.31) is False  # only reopens above the high threshold


def test_single_threshold_would_chatter():
    """Control case showing why the dead band exists."""
    naive = [v < 0.25 for v in (0.24, 0.26, 0.24, 0.26)]
    assert naive == [True, False, True, False]  # 4 state changes

    h = Hysteresis(0.20, 0.30)
    states = [h.update(v) for v in (0.24, 0.26, 0.24, 0.26)]
    assert len(set(states)) == 1  # zero state changes


def test_debounce():
    d = Debounce(5.0)
    assert d.allow(now=100.0) is True
    assert d.allow(now=102.0) is False
    assert d.allow(now=105.5) is True


def test_time_window_evicts():
    w = TimeWindow(10.0)
    for t in range(20):
        w.push(float(t), t)
    assert len(w) == 11  # t=9..19 inclusive
    assert min(w.values()) == 9


def test_ema_converges():
    e = EMA(0.5)
    for _ in range(30):
        e.update(1.0)
    assert abs(e.value - 1.0) < 1e-6


# ------------------------------------------------------------------ blink --
def drive_closure(detector, closed_seconds, fps=30.0, t0=1000.0, lead=0.5, tail=0.5):
    """Feed a scripted open->closed->open sequence at a fixed frame rate."""
    dt = 1.0 / fps
    t = t0
    events = []

    def step(openness, duration):
        nonlocal t
        for _ in range(max(1, int(round(duration / dt)))):
            ev = detector.update(openness, valid=True, now=t)
            if ev:
                events.append(ev)
            t += dt

    step(0.90, lead)
    step(0.05, closed_seconds)
    step(0.90, tail)
    return events


def make_detector():
    return BlinkDetector(EyeConfig(), BlinkConfig())


def test_normal_blink_is_classified_as_blink():
    events = drive_closure(make_detector(), 0.20)
    assert len(events) == 1
    assert events[0].kind is ClosureKind.BLINK
    assert 150 < events[0].duration_ms < 300


def test_long_blink():
    events = drive_closure(make_detector(), 0.60)
    assert events[0].kind is ClosureKind.LONG_BLINK


def test_prolonged_closure():
    events = drive_closure(make_detector(), 1.0)
    assert events[0].kind is ClosureKind.PROLONGED


def test_microsleep():
    events = drive_closure(make_detector(), 2.0)
    assert events[0].kind is ClosureKind.MICROSLEEP


def test_sub_blink_noise_is_discarded():
    """A one-frame dropout must not be counted as a blink."""
    d = make_detector()
    events = drive_closure(d, closed_seconds=0.02)  # ~0.6 frames at 30 fps
    assert events == []


def test_duration_is_measured_in_time_not_frames():
    """The same real-world blink must measure the same at 10 fps and 30 fps.

    This is the bug in every `EYE_AR_CONSEC_FRAMES = 16` implementation: the
    threshold silently encodes a frame rate, so the classification changes when
    the Pi drops frames under load.
    """
    slow = drive_closure(make_detector(), 0.60, fps=10.0)[0].duration_ms
    fast = drive_closure(make_detector(), 0.60, fps=30.0)[0].duration_ms
    assert abs(slow - fast) < 120, (slow, fast)


def test_ongoing_closure_reports_before_the_eye_reopens():
    """A microsleep alert must not wait for the eye to open again."""
    d = make_detector()
    t = 1000.0
    dt = 1 / 30.0
    for _ in range(10):
        d.update(0.9, valid=True, now=t)
        t += dt
    for _ in range(60):  # 2 s closed, still closed
        d.update(0.05, valid=True, now=t)
        t += dt
    assert d.state is EyeState.CLOSED
    assert d.ongoing_closure_ms(t) > 1800


def test_invalid_frame_aborts_measurement_rather_than_guessing():
    d = make_detector()
    t = 1000.0
    for _ in range(10):
        d.update(0.9, valid=True, now=t)
        t += 1 / 30
    for _ in range(10):
        d.update(0.05, valid=True, now=t)
        t += 1 / 30
    d.update(None, valid=False, now=t)
    assert d.state is EyeState.UNKNOWN
    assert d.ongoing_closure_ms(t) == 0.0


# ---------------------------------------------------------------- PERCLOS --
def make_meter():
    return PerclosMeterWithEyeConfig(PerclosConfig(), EyeConfig())


def test_perclos_does_not_fire_on_normal_blinking():
    """The central claim of the design: a correctly implemented PERCLOS is
    about slow droops, so ordinary blinking must leave it far below threshold.

    15 blinks/min at 250 ms each = 3.75 s of closure in 60 s = 6.25%, which is
    below the 15% warning threshold by construction, not by tuning.
    """
    m = make_meter()
    fps, dt = 30.0, 1 / 30.0
    t = 1000.0
    blink_every = int(fps * 4)  # 15 blinks per minute
    blink_frames = int(0.25 * fps)  # 250 ms each

    for i in range(int(fps * 60)):
        closed = (i % blink_every) < blink_frames
        m.update(0.05 if closed else 0.9, valid=True, timestamp=t)
        t += dt

    res = m.compute(t)
    assert res.valid
    assert res.primary < PerclosConfig().warn_threshold, res.primary
    assert 0.04 < res.primary < 0.09, res.primary


def test_perclos_detects_sustained_droop():
    m = make_meter()
    fps, dt = 30.0, 1 / 30.0
    t = 1000.0
    for i in range(int(fps * 60)):
        closed = i > int(fps * 42)  # eyes shut for the last 18 s
        m.update(0.05 if closed else 0.9, valid=True, timestamp=t)
        t += dt
    res = m.compute(t)
    assert res.primary >= PerclosConfig().drowsy_threshold, res.primary


def test_invalid_frames_leave_both_numerator_and_denominator():
    """Counting 'no face' as 'eyes open' is the classic under-reporting bug."""
    m = make_meter()
    t = 1000.0
    dt = 1 / 30.0
    for _ in range(300):  # 10 s closed, valid
        m.update(0.05, valid=True, timestamp=t)
        t += dt
    for _ in range(300):  # 10 s of no face
        m.update(None, valid=False, timestamp=t)
        t += dt

    res = m.compute(t)
    # Of the frames we could actually see, 100% were closed.
    assert abs(res.primary - 1.0) < 1e-6, res.primary
    assert abs(res.validity - 0.5) < 0.02, res.validity


def test_low_validity_reports_unavailable_not_a_number():
    m = make_meter()
    t = 1000.0
    for _ in range(300):
        m.update(0.9, valid=True, timestamp=t)
        t += 1 / 30
    for _ in range(900):
        m.update(None, valid=False, timestamp=t)
        t += 1 / 30
    res = m.compute(t)
    assert res.validity < PerclosConfig().min_validity_ratio
    assert res.valid is False
    assert m.level(res) == "unavailable"


# -------------------------------------------------------------- head pose --
def test_angle_normalisation_folds_the_flip_solution():
    """solvePnP's degenerate solution puts pitch/roll past +/-90 degrees."""
    _, pitch, _ = normalise_angles(0.0, 175.0, 0.0)
    assert abs(pitch - 5.0) < 1e-6, pitch
    _, _, roll = normalise_angles(0.0, 0.0, -170.0)
    assert abs(roll - 10.0) < 1e-6, roll


def test_angle_normalisation_leaves_normal_angles_alone():
    y, p, r = normalise_angles(-20.0, 15.0, -8.0)
    assert (round(y), round(p), round(r)) == (-20, 15, -8)


# ------------------------------------------------------------ calibration --
def test_calibration_rejects_indistinct_endpoints():
    """If the driver did not actually close their eyes, fail loudly rather
    than producing a compressed openness scale that silently never reaches 0."""
    open_phase = {"ear": 0.25, "h": 0.5, "v": 0.5, "yaw": 0, "pitch": 0, "roll": 0, "n": 200}
    closed_phase = {"ear": 0.24, "h": None, "v": None, "yaw": 0, "pitch": 0, "roll": 0, "n": 60}
    try:
        build_profile("x", open_phase, closed_phase)
    except ValueError as exc:
        assert "too close together" in str(exc)
    else:
        raise AssertionError("should have rejected the calibration")


def test_calibration_accepts_good_endpoints():
    open_phase = {"ear": 0.28, "h": 0.52, "v": 0.49, "yaw": 1.0, "pitch": 20.0, "roll": 0, "n": 250}
    closed_phase = {"ear": 0.09, "h": None, "v": None, "yaw": 0, "pitch": 0, "roll": 0, "n": 80}
    p = build_profile("driver1", open_phase, closed_phase)
    assert p.ear_open == 0.28 and p.ear_closed == 0.09
    assert p.pitch0 == 20.0  # neutral pose captured -> cancels the pitch bias





# ------------------------------------------------------- state-machine guard --
def _guard_machine():
    from dms_face.config import Config
    from dms_face.state import StateMachine

    return StateMachine(Config()), Config()


class _FakePerclos:
    primary = 1.0
    fast = 1.0
    validity = 1.0
    valid = True
    warming_up = False
    n_frames = 1000
    window_span_s = 60.0


class _FakeBlink:
    def __init__(self, closure_ms):
        self._c = closure_ms
        self.last_event = None

    def ongoing_closure_ms(self, now=None):
        return self._c

    def blink_rate_per_min(self, now=None):
        return None

    def recent_mean_blink_ms(self, count=10):
        return None


def _run_guard(closure_ms):
    from dms_face.calibration import DriverProfile

    machine, cfg = _guard_machine()
    return machine.update(
        now=1000.0,
        perclos_result=_FakePerclos(),
        perclos_level="drowsy",
        blink_detector=_FakeBlink(closure_ms),
        head_angles=(0.0, 0.0, 0.0),
        gaze_deviation=0.0,
        eye_channel_valid=True,
        face_present=True,
        profile=DriverProfile(),
    )


def test_genuine_microsleep_still_reports_drowsy():
    from dms_face.state import DriverState

    report = _run_guard(2500.0)  # 2.5 s -- a real microsleep
    assert report.state is DriverState.DROWSY


def test_implausible_unbroken_closure_is_flagged_not_reported_as_drowsy():
    """A real microsleep ends. A 60 s unbroken 'closure' means the openness
    scale is wrong -- almost always an uncalibrated profile. Reporting DROWSY
    there is confidently wrong, which is worse than reporting nothing."""
    from dms_face.state import DriverState

    report = _run_guard(60_000.0)
    assert report.state is DriverState.UNAVAILABLE
    assert "uncalibrated" in report.reasons[0]


def test_perclos_cold_start_does_not_report_drowsy():
    """The bug this guards against, observed live: on the very first frame the
    window holds one sample, so a single closed frame reads as 100% and the
    system announces DROWSY the instant it launches."""
    m = make_meter()
    t = 1000.0
    m.update(0.05, valid=True, timestamp=t)  # one closed frame
    res = m.compute(t)
    assert res.primary == 1.0          # the raw ratio really is 100%
    assert res.warming_up is True      # ...but it is not yet meaningful
    assert res.valid is False
    assert m.level(res) == "warming"


def test_perclos_becomes_valid_once_the_window_fills():
    m = make_meter()
    t, dt = 1000.0, 1 / 30.0
    for _ in range(int(30 * 15)):  # 15 s, past the 10 s minimum span
        m.update(0.9, valid=True, timestamp=t)
        t += dt
    res = m.compute(t)
    assert res.warming_up is False
    assert res.valid is True
    assert m.level(res) == "ok"


def test_no_face_does_not_inherit_a_stale_blink_verdict():
    """Also observed live: a prolonged blink from 3 s ago was still 'recent',
    so a frame with no face at all reported WARNING -- a driver state the
    system demonstrably could not observe."""
    from dms_face.blink import ClosureEvent, ClosureKind as CK
    from dms_face.calibration import DriverProfile
    from dms_face.state import DriverState

    machine, _ = _guard_machine()
    blink = _FakeBlink(0.0)
    blink.last_event = ClosureEvent(
        kind=CK.PROLONGED, duration_ms=1200.0, start_time=997.0, end_time=998.0
    )
    report = machine.update(
        now=1000.0,
        perclos_result=_FakePerclos(),
        perclos_level="drowsy",
        blink_detector=blink,
        head_angles=None,
        gaze_deviation=None,
        eye_channel_valid=False,
        face_present=False,
        profile=DriverProfile(),
    )
    assert report.state is DriverState.UNAVAILABLE
    assert report.reasons == ["no face detected"]



# ------------------------------------------------- signal-quality assessment --
def test_quality_separates_wide_and_narrow_eyes():
    """Two drivers can both pass the separation check and behave completely
    differently. What decides usability is range against jitter, not range."""
    from dms_face.calibration import assess_quality

    wide, snr_wide = assess_quality(separation=0.21, noise=0.005)
    narrow, snr_narrow = assess_quality(separation=0.06, noise=0.005)

    assert wide == "good", (wide, snr_wide)
    assert narrow == "poor", (narrow, snr_narrow)
    # Both would sail past a naive `separation > 0.04` check, which is exactly
    # why separation alone is not a sufficient acceptance test.
    assert 0.21 > 0.04 and 0.06 > 0.04

    # The bands track the hysteresis dead band rather than being fixed: widen
    # the dead band and the same driver's signal becomes more acceptable,
    # because a wider band tolerates proportionally more jitter.
    assert assess_quality(0.06, 0.005, hysteresis_band=0.30)[0] == "fair"
    assert assess_quality(0.06, 0.005, hysteresis_band=0.40)[0] == "good"


def test_quality_handles_missing_noise_estimate():
    from dms_face.calibration import assess_quality

    q, snr = assess_quality(0.2, None)
    assert q == "unknown" and snr is None


def test_open_state_noise_ignores_blinks():
    """The eyes-open phase deliberately contains blinks. A standard deviation
    would count them as noise and overstate jitter several-fold."""
    from dms_face.calibration import CalibrationCollector
    import statistics

    c = CalibrationCollector(settle_s=0.0).start()
    samples = [0.250, 0.252, 0.248, 0.251, 0.249, 0.250, 0.253, 0.247] * 5
    samples += [0.06, 0.05, 0.07]  # three blinks
    for v in samples:
        c.ear.append(v)

    noise = c._open_state_noise()
    assert noise is not None and noise < 0.005, noise
    assert statistics.pstdev(samples) > 0.03  # what a naive stdev would report


def test_poor_quality_profile_advises_the_user():
    from dms_face.calibration import DriverProfile

    p = DriverProfile(quality="poor", snr=3.2, separation=0.05, noise=0.016)
    advice = p.quality_advice()
    assert advice is not None and "POOR" in advice

    assert DriverProfile(quality="good", snr=20.0).quality_advice() is None



# ---------------------------------------------------------- lean-install shims --
def test_matplotlib_stub_lets_the_import_succeed_but_fails_loudly_on_use():
    """MediaPipe hard-imports matplotlib.pyplot for one plotting helper. The
    lean install has no matplotlib, so the stub must (a) let that import work
    and (b) raise a clear error if anything actually uses it -- never silently
    return garbage."""
    from dms_face import _lean

    names = ("matplotlib", "matplotlib.pyplot")
    saved = {n: sys.modules.get(n) for n in names}
    try:
        for n in names:
            sys.modules.pop(n, None)

        assert _lean.install_matplotlib_stub(force=True) is True

        import matplotlib.pyplot as plt  # must not raise

        try:
            plt.figure()
        except AttributeError as exc:
            assert "lean" in str(exc) and "pip install matplotlib" in str(exc)
        else:
            raise AssertionError("using the stub must raise, not succeed")

        # A feature probe must get a truthful "no", not an exception. This is
        # the regression: the first version raised ImportError here.
        assert hasattr(plt, "figure") is False
        assert hasattr(sys.modules["matplotlib"], "rcParams") is False
    finally:
        for n in names:
            sys.modules.pop(n, None)
            if saved[n] is not None:
                sys.modules[n] = saved[n]


def test_matplotlib_stub_is_a_noop_when_the_real_package_is_present():
    from dms_face import _lean

    if not _lean.matplotlib_available():
        return  # nothing to prove on a machine without it
    before = sys.modules.get("matplotlib")
    assert _lean.install_matplotlib_stub() is False
    assert sys.modules.get("matplotlib") is before


def test_gui_check_refuses_without_a_display_on_linux():
    """Over SSH, a GUI build of OpenCV does not raise -- Qt calls abort() and
    the process vanishes with no traceback. So this has to be decided from the
    environment, before any window function is touched."""
    import run as app

    saved = (sys.platform, os.environ.pop("DISPLAY", None), os.environ.pop("WAYLAND_DISPLAY", None))
    try:
        sys.platform = "linux"
        ok, why = app.gui_available()
    finally:
        sys.platform = saved[0]
        for key, val in (("DISPLAY", saved[1]), ("WAYLAND_DISPLAY", saved[2])):
            if val is not None:
                os.environ[key] = val

    assert ok is False
    assert "DISPLAY" in why


if __name__ == "__main__":
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print("  PASS  " + name)
            passed += 1
        except Exception:
            print("  FAIL  " + name)
            traceback.print_exc()
            failed += 1
    print()
    print("%d passed, %d failed, %d total" % (passed, failed, len(tests)))
    sys.exit(1 if failed else 0)
