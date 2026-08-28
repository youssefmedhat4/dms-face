"""
Landmark index sets and the canonical 3D face model.

Index conventions follow MediaPipe Face Landmarker (478 points: 468 mesh + 10 iris).
"Left" / "right" are from the DRIVER's point of view, not the image's.

The 3D coordinates below are read verbatim from Google's
`canonical_face_model.obj` (mediapipe/modules/face_geometry/data/), in the
model's own metric units (roughly centimetres). They are not hand-authored.
"""

# --- Eye Aspect Ratio -------------------------------------------------------
# Order is the EAR convention: p1..p6 = [outer, upper-outer, upper-inner,
# inner, lower-inner, lower-outer]. EAR uses ||p2-p6||, ||p3-p5||, ||p1-p4||.
# VERIFY THESE VISUALLY ONCE with `python run.py --show-indices` before trusting
# them: they are a community convention, not an official Google specification.
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)

# --- Eye corners and lids (gaze ratios) -------------------------------------
RIGHT_EYE_CORNER_OUTER = 33
RIGHT_EYE_CORNER_INNER = 133
RIGHT_EYE_LID_UPPER = 159
RIGHT_EYE_LID_LOWER = 145

LEFT_EYE_CORNER_INNER = 362
LEFT_EYE_CORNER_OUTER = 263
LEFT_EYE_LID_UPPER = 386
LEFT_EYE_LID_LOWER = 374

# --- Iris (present only when the model bundle emits 478 points) -------------
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_RING = (469, 470, 471, 472)
LEFT_IRIS_RING = (474, 475, 476, 477)

N_LANDMARKS_WITH_IRIS = 478
N_LANDMARKS_MESH_ONLY = 468

# --- Head pose --------------------------------------------------------------
# Rigid points only. Jaw, mouth and eyebrow landmarks are deliberately excluded:
# they move independently of the skull and violate solvePnP's rigid-body
# assumption. Yawning would otherwise corrupt the pitch estimate.
CANONICAL_FACE_3D = {
     10: (  0.000000,   8.261778,   4.481535),
    151: (  0.000000,   6.545390,   5.027311),
      9: (  0.000000,   4.885979,   5.385258),
      8: (  0.000000,   4.019042,   5.284764),
    168: (  0.000000,   3.271027,   5.236015),
      6: (  0.000000,   2.473255,   5.788627),
    197: (  0.000000,   1.728369,   6.316750),
    195: (  0.000000,   1.059413,   6.774605),
      5: (  0.000000,   0.365669,   7.242870),
      4: (  0.000000,  -0.463170,   7.586580),
      1: (  0.000000,  -1.126865,   7.475604),
      2: (  0.000000,  -2.089024,   6.058267),
     98: ( -1.405627,  -1.714196,   5.241087),
    327: (  1.405627,  -1.714196,   5.241087),
     33: ( -4.445859,   2.663991,   3.173422),
    133: ( -1.856432,   2.585245,   3.757904),
    362: (  1.856432,   2.585245,   3.757904),
    263: (  4.445859,   2.663991,   3.173422),
    234: ( -7.664182,   0.673132,  -2.435867),
    454: (  7.664182,   0.673132,  -2.435867),
    116: ( -6.465170,   0.937119,   1.689873),
    345: (  6.465170,   0.937119,   1.689873),
}

POSE_LANDMARK_IDS = tuple(CANONICAL_FACE_3D.keys())


def pose_model_points():
    """(N, 3) float64 canonical points, ordered to match POSE_LANDMARK_IDS,
    converted into OpenCV's camera convention.

    The canonical model is authored Y-up, Z-out-of-the-face. OpenCV expects
    Y-down, Z-into-the-scene. The two differ by a 180-degree rotation about X,
    applied here as (x, -y, -z).

    Skipping this conversion does not throw -- solvePnP happily returns a
    solution roughly 180 degrees out, which then surfaces as a nonsensical
    pitch (measured: -158.9 instead of 21.1 on the same frame). Many published
    implementations carry this bug and mask it by only ever looking at yaw.
    """
    import numpy as np

    pts = np.array([CANONICAL_FACE_3D[i] for i in POSE_LANDMARK_IDS], dtype=np.float64)
    pts[:, 1] *= -1.0
    pts[:, 2] *= -1.0
    return pts
