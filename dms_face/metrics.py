"""
Geometric measurements read off the landmark tensor. No neural network here --
this is the whole point of the architecture: one inference, then mathematics.
"""

import math
import numpy as np

from . import landmark_ids as L


def to_pixels(landmarks, width, height):
    """Normalised landmarks -> (N, 2) pixel array.

    This conversion matters. MediaPipe normalises x by frame WIDTH and y by
    frame HEIGHT, so on a 640x480 frame the two axes are scaled differently.
    Computing EAR directly on normalised coordinates bakes in a 4:3 distortion
    and makes the metric change meaning if you ever change resolution. Many
    published implementations get this wrong; calibration hides it, but only
    until the resolution changes.
    """
    pts = np.empty((len(landmarks), 2), dtype=np.float64)
    for i, lm in enumerate(landmarks):
        pts[i, 0] = lm.x * width
        pts[i, 1] = lm.y * height
    return pts


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(pts, idx):
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Soukupova & Cech, CVWW 2016. The ratio form makes it scale-invariant: it
    does not care how far the face is from the camera. It is NOT rotation
    invariant -- see HeadPoseConfig for why we gate it on yaw and pitch.
    """
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return None
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def both_ear(pts):
    """(right_ear, left_ear) from the driver's point of view."""
    return (
        eye_aspect_ratio(pts, L.RIGHT_EYE_EAR),
        eye_aspect_ratio(pts, L.LEFT_EYE_EAR),
    )


def combine_ear(right, left):
    """Mean of the two eyes when both are valid, otherwise whichever survives.

    Deliberately the MEAN, not the minimum. Taking the minimum lets a single
    jittery landmark on one eye drive the entire system into a false closure.
    """
    vals = [v for v in (right, left) if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def openness(ear, ear_open, ear_closed):
    """Map raw EAR onto a true 0..1 eyelid-opening scale.

        openness = (EAR - EAR_closed) / (EAR_open - EAR_closed)

    This is the step almost every student implementation skips, and it is what
    makes the P80 threshold mean what Wierwille says it means. Raw EAR does not
    reach zero when the eye shuts -- the landmarks retain separation -- so
    dividing raw EAR by a fixed constant produces an arbitrary number, not P80.
    """
    if ear is None:
        return None
    denom = ear_open - ear_closed
    if denom < 1e-6:
        return None
    return float(np.clip((ear - ear_closed) / denom, 0.0, 1.0))


def iris_ratios(pts, has_iris=True):
    """Normalised iris position inside each eye socket.

    Returns (h_ratio, v_ratio) averaged over both eyes, or (None, None).
    h_ratio ~0.5 means the iris is horizontally centred; v_ratio ~0.5 vertically.

    Head pose catches the large deviations (driver turns to the passenger).
    This catches the one head pose misses: eyes down at a phone while the head
    stays forward. That single case is the entire reason to use 478 landmarks
    rather than 468.
    """
    if not has_iris or len(pts) < L.N_LANDMARKS_WITH_IRIS:
        return None, None

    def one_eye(iris_c, inner, outer, upper, lower):
        cx, cy = pts[iris_c]
        ix = pts[inner][0]
        ox = pts[outer][0]
        uy = pts[upper][1]
        ly = pts[lower][1]
        wide = ox - ix
        tall = ly - uy
        if abs(wide) < 1e-6 or abs(tall) < 1e-6:
            return None, None
        return (cx - ix) / wide, (cy - uy) / tall

    rh, rv = one_eye(
        L.RIGHT_IRIS_CENTER,
        L.RIGHT_EYE_CORNER_INNER,
        L.RIGHT_EYE_CORNER_OUTER,
        L.RIGHT_EYE_LID_UPPER,
        L.RIGHT_EYE_LID_LOWER,
    )
    lh, lv = one_eye(
        L.LEFT_IRIS_CENTER,
        L.LEFT_EYE_CORNER_INNER,
        L.LEFT_EYE_CORNER_OUTER,
        L.LEFT_EYE_LID_UPPER,
        L.LEFT_EYE_LID_LOWER,
    )

    hs = [v for v in (rh, lh) if v is not None]
    vs = [v for v in (rv, lv) if v is not None]
    if not hs or not vs:
        return None, None
    return sum(hs) / len(hs), sum(vs) / len(vs)


def gaze_deviation(h_ratio, v_ratio, h0, v0):
    """Euclidean distance from the driver's calibrated on-road gaze centre."""
    if h_ratio is None or v_ratio is None:
        return None
    return math.hypot(h_ratio - h0, v_ratio - v0)


def interocular_distance(pts):
    """Outer-corner to outer-corner, in pixels. Useful as a crude proxy for
    face distance and for sanity-checking that a detection is plausible."""
    return _dist(pts[L.RIGHT_EYE_CORNER_OUTER], pts[L.LEFT_EYE_CORNER_OUTER])
