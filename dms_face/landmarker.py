"""
MediaPipe Face Landmarker wrapper -- the ONLY neural network in this pipeline.

Uses the Tasks API (`mediapipe.tasks.python.vision.FaceLandmarker`), not the
legacy `mp.solutions.face_mesh`. That is not a style preference:

    mediapipe <= 0.10.18   ARM64 wheels, has `mediapipe.solutions`
    mediapipe 0.10.21-0.10.35  NO ARM64 wheels at all
    mediapipe >= 1.0.0     ARM64 wheel returns, `mediapipe.solutions` REMOVED

So on any current install, every tutorial that starts with
`mp.solutions.face_mesh.FaceMesh(...)` raises AttributeError. The Tasks API is
the only forward-compatible surface, and it additionally gives us the facial
transformation matrix for a free head-pose cross-check.
"""

import os
from dataclasses import dataclass

import numpy as np

from . import landmark_ids as L


@dataclass
class LandmarkFrame:
    landmarks: object  # list of NormalizedLandmark
    pixels: np.ndarray  # (N, 2) float64, pixel coordinates
    has_iris: bool
    transform_matrix: object  # 4x4 or None
    blendshapes: object  # list or None

    @property
    def n(self):
        return len(self.pixels)


class FaceLandmarkerWrapper:
    def __init__(self, cfg, width, height, running_mode="video"):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._vision = vision
        model_path = cfg.model_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"model bundle not found: {model_path}\n"
                "Download it with:\n"
                "  curl -L -o models/face_landmarker.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task"
            )

        mode = {
            "image": vision.RunningMode.IMAGE,
            "video": vision.RunningMode.VIDEO,
        }[running_mode]

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mode,
            # num_faces=1 also enables MediaPipe's internal landmark smoothing,
            # and keeps the detector from re-running on passengers.
            num_faces=1,
            min_face_detection_confidence=cfg.min_face_detection_confidence,
            min_face_presence_confidence=cfg.min_face_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            output_face_blendshapes=cfg.output_face_blendshapes,
            output_facial_transformation_matrixes=cfg.output_facial_transformation_matrixes,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mode = running_mode
        self.width = width
        self.height = height
        self._last_ts_ms = -1
        self.iris_available = None  # resolved on the first successful detection

    def process(self, frame_rgb, timestamp_s):
        """Run one inference. Returns a LandmarkFrame, or None if no face."""
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._mode == "video":
            # VIDEO mode enables detection-then-tracking: the 192x192 detector
            # only re-runs when tracking confidence drops, so steady-state cost
            # is the mesh alone. Timestamps MUST be monotonically increasing.
            ts_ms = int(timestamp_s * 1000.0)
            if ts_ms <= self._last_ts_ms:
                ts_ms = self._last_ts_ms + 1
            self._last_ts_ms = ts_ms
            result = self._landmarker.detect_for_video(mp_image, ts_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        has_iris = len(landmarks) >= L.N_LANDMARKS_WITH_IRIS
        if self.iris_available is None:
            self.iris_available = has_iris

        pixels = np.empty((len(landmarks), 2), dtype=np.float64)
        for i, lm in enumerate(landmarks):
            pixels[i, 0] = lm.x * self.width
            pixels[i, 1] = lm.y * self.height

        matrix = None
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if matrices:
            matrix = matrices[0]

        blend = None
        shapes = getattr(result, "face_blendshapes", None)
        if shapes:
            blend = shapes[0]

        return LandmarkFrame(
            landmarks=landmarks,
            pixels=pixels,
            has_iris=has_iris,
            transform_matrix=matrix,
            blendshapes=blend,
        )

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def blendshape_eye_blink(blendshapes):
    """(left, right) eyeBlink scores, if blendshapes were requested.

    An independent, appearance-based estimate of eye closure that does not use
    EAR at all. Off by default because it costs extra compute, but it is a
    genuinely useful cross-check when EAR behaves oddly -- particularly under
    IR, where landmark geometry degrades before appearance does.
    """
    if not blendshapes:
        return None, None
    scores = {c.category_name: c.score for c in blendshapes}
    return scores.get("eyeBlinkLeft"), scores.get("eyeBlinkRight")
