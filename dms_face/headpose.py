"""
Head pose by solvePnP on rigid landmarks.

Costs microseconds and is fully derivable on paper, which is exactly why it
beats adding a second neural network here.
"""

import math
import numpy as np
import cv2

from . import landmark_ids as L


class HeadPoseEstimator:
    def __init__(self, width, height, camera_matrix=None, dist_coeffs=None):
        self.width = width
        self.height = height
        self.model_points = L.pose_model_points()
        self.ids = L.POSE_LANDMARK_IDS

        if camera_matrix is None:
            # Fallback approximation: focal length ~= image width, principal
            # point at centre.
            #
            # Measured on tests/assets/portrait.jpg, sweeping the assumed focal
            # length while holding the landmarks fixed:
            #
            #     focal    yaw    pitch    roll
            #       400   -1.4     38.1    -1.4
            #       820   -1.6     21.1    -1.0
            #      2000   -1.7      9.4    -0.7
            #      3000   -1.7      6.5    -0.6
            #
            # PITCH is almost entirely an artefact of the guessed focal length;
            # yaw and roll are nearly immune to it. So:
            #   * yaw and roll are usable uncalibrated,
            #   * ABSOLUTE pitch is not,
            #   * pitch RELATIVE to the driver's calibrated neutral still is,
            #     because a constant bias cancels in the subtraction.
            # This is why the distraction gate uses (pitch - pitch0), never raw
            # pitch. Run `python calibrate_camera.py` to remove the guess
            # entirely; on the 160-degree module that is mandatory, not optional.
            f = float(width)
            camera_matrix = np.array(
                [[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64
            )
            self.calibrated = False
        else:
            self.calibrated = True

        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = (
            np.zeros((4, 1)) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
        )
        self._last_rvec = None
        self._last_tvec = None

    @classmethod
    def from_calibration_file(cls, path, width, height):
        import json

        with open(path, encoding="utf8") as f:
            data = json.load(f)
        return cls(
            width,
            height,
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64),
        )

    def estimate(self, pts):
        """pts: (N,2) pixel landmarks. Returns (yaw, pitch, roll) in degrees."""
        try:
            image_points = np.array([pts[i] for i in self.ids], dtype=np.float64)
        except IndexError:
            return None

        use_guess = self._last_rvec is not None
        ok, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            rvec=self._last_rvec.copy() if use_guess else None,
            tvec=self._last_tvec.copy() if use_guess else None,
            useExtrinsicGuess=use_guess,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            self._last_rvec = None
            self._last_tvec = None
            return None

        self._last_rvec, self._last_tvec = rvec, tvec

        rmat, _ = cv2.Rodrigues(rvec)
        angles = decompose_rotation(rmat)
        return normalise_angles(*angles)

    def reset(self):
        self._last_rvec = None
        self._last_tvec = None


def decompose_rotation(rmat):
    """Rotation matrix -> (yaw, pitch, roll) in degrees.

    Axis assignment in the OpenCV camera frame (X right, Y down, Z into scene):

        yaw   = rotation about Y  -- turning to look left/right
        pitch = rotation about X  -- nodding up/down
        roll  = rotation about Z  -- tilting an ear toward a shoulder

    Getting this mapping wrong is easy and silent: the numbers still look
    plausible, they are just attached to the wrong names, so a head tilt reads
    as a head turn and the distraction gate fires on the wrong motion.
    """
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        yaw = math.atan2(-rmat[2, 0], sy)
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:  # gimbal lock: roll and yaw are degenerate, pin roll to 0
        yaw = math.atan2(-rmat[2, 0], sy)
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        roll = 0.0

    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def normalise_angles(yaw, pitch, roll):
    """Wrap to [-180, 180] and fold the known solvePnP flip solution back.

    solvePnP has a documented degenerate solution in which pitch or roll jumps
    past +/-90 degrees. Left unhandled it makes the reading oscillate wildly at
    moderate head angles.
    """

    def wrap(a):
        while a > 180.0:
            a -= 360.0
        while a < -180.0:
            a += 360.0
        return a

    yaw, pitch, roll = wrap(yaw), wrap(pitch), wrap(roll)

    if abs(pitch) > 90.0:
        pitch = wrap(180.0 - pitch) if pitch > 0 else wrap(-180.0 - pitch)
    if abs(roll) > 90.0:
        roll = wrap(roll - 180.0) if roll > 0 else wrap(roll + 180.0)

    return yaw, pitch, roll


def angles_from_transformation_matrix(matrix):
    """Independent head pose from MediaPipe's own facial transformation matrix.

    Free cross-check: no solvePnP, no camera intrinsics. Agreement between this
    and the solvePnP result is a strong validation to put in your report;
    persistent disagreement usually means your intrinsics are wrong.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4):
        return None
    rmat = m[:3, :3]
    # Strip any scale the fit introduced before decomposing.
    for c in range(3):
        n = np.linalg.norm(rmat[:, c])
        if n > 1e-9:
            rmat[:, c] /= n
    return normalise_angles(*decompose_rotation(rmat))
