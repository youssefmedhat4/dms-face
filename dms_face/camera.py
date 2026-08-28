"""
Camera abstraction.

Picamera2 on the Raspberry Pi, OpenCV VideoCapture on a development machine.
Same interface, so every downstream module is identical on both -- you develop
and tune on a laptop webcam, then deploy to the Pi without touching the logic.

Capture runs on its own thread. Decoupling capture from processing is the
single cheapest throughput win available: the main loop always gets the newest
frame instead of blocking on the camera, and a slow frame never stalls the grab.
"""

import platform
import threading
import time

import cv2
import numpy as np


def _apply_clahe(frame):
    """Adaptive histogram equalisation on luminance only.

    IR cabin images are low-contrast; CLAHE costs a few milliseconds and
    measurably improves landmark confidence. Applied to L in LAB space so it
    does not shift colour.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def ensure_three_channel(frame):
    """MediaPipe REQUIRES 3-channel input and raises otherwise.

    A monochrome IR sensor delivering a single plane must be replicated across
    three channels before inference. This is the first thing that breaks when
    you move from a colour webcam to the IR module.
    """
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    return frame


class FrameSource:
    """Common interface. Frames are delivered as RGB uint8, HxWx3."""

    def start(self):
        raise NotImplementedError

    def read(self):
        """Returns (frame_rgb, timestamp_monotonic) or (None, None)."""
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


class ThreadedCapture(FrameSource):
    """Background grab loop shared by both backends."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._frame = None
        self._timestamp = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._frames_grabbed = 0

    def _grab(self):
        raise NotImplementedError

    def _loop(self):
        while self._running:
            frame = self._grab()
            if frame is None:
                time.sleep(0.002)
                continue
            frame = ensure_three_channel(frame)
            if self.cfg.flip_horizontal:
                frame = cv2.flip(frame, 1)
            if self.cfg.ir_mode:
                frame = _apply_clahe(frame)
            with self._lock:
                self._frame = frame
                self._timestamp = time.monotonic()
                self._frames_grabbed += 1

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="capture")
        self._thread.start()
        # Wait for the first frame so callers never get a spurious None.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self
            time.sleep(0.01)
        raise RuntimeError("camera produced no frames within 5 s")

    def read(self):
        with self._lock:
            if self._frame is None:
                return None, None
            return self._frame, self._timestamp

    @property
    def size(self):
        """(width, height) of the frames actually delivered -- which is not
        always what was requested; webcams silently substitute sizes."""
        with self._lock:
            if self._frame is None:
                return self.cfg.width, self.cfg.height
            h, w = self._frame.shape[:2]
            return w, h

    @property
    def frames_grabbed(self):
        with self._lock:
            return self._frames_grabbed

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class OpenCVSource(ThreadedCapture):
    """Development machine: laptop or USB webcam."""

    def __init__(self, cfg):
        super().__init__(cfg)
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(cfg.index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"could not open camera index {cfg.index}. "
                "Close any app already using the webcam, or pass --camera-index 1."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _grab(self):
        ok, bgr = self._cap.read()
        if not ok:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def stop(self):
        super().stop()
        self._cap.release()

    def describe(self):
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return f"OpenCV webcam #{self.cfg.index} @ {w}x{h}"


class PiCameraSource(ThreadedCapture):
    """Raspberry Pi. Uses the `lores` stream so the ISP does the downscale in
    hardware -- resizing a 1280x720 frame in OpenCV on the CPU is pure waste
    when the models only consume 192x192 and 256x256 anyway."""

    def __init__(self, cfg):
        super().__init__(cfg)
        from picamera2 import Picamera2  # noqa: import guarded, Pi only

        self._picam = Picamera2()
        config = self._picam.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"},
            lores={"size": (cfg.width, cfg.height), "format": "RGB888"},
            controls={"FrameRate": cfg.fps},
        )
        self._picam.configure(config)
        self._picam.start()
        time.sleep(0.5)  # let AE/AWB settle

    def _grab(self):
        return self._picam.capture_array("lores")

    def stop(self):
        super().stop()
        self._picam.stop()
        self._picam.close()

    def describe(self):
        return f"Picamera2 lores @ {self.cfg.width}x{self.cfg.height}"


class VideoFileSource(FrameSource):
    """Replay a recorded clip. This is how you get REPEATABLE evaluation
    numbers -- a live webcam gives a different answer every run, which is
    useless for a results chapter."""

    def __init__(self, path, cfg=None, realtime=False):
        self.path = path
        self.cfg = cfg
        self.realtime = realtime
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video file: {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._index = 0
        self._t0 = None
        self._size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def size(self):
        return self._size

    def start(self):
        self._t0 = time.monotonic()
        return self

    def read(self):
        ok, bgr = self._cap.read()
        if not ok:
            return None, None
        frame = ensure_three_channel(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self.cfg is not None and self.cfg.ir_mode:
            frame = _apply_clahe(frame)
        # Synthetic timestamps derived from the file's own frame rate, so
        # blink durations are measured against the video clock rather than how
        # fast this machine happens to decode.
        ts = self._t0 + self._index / self.fps
        self._index += 1
        if self.realtime:
            target = self._t0 + self._index / self.fps
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return frame, ts

    def stop(self):
        self._cap.release()

    def describe(self):
        return f"video file {self.path} ({self.frame_count} frames @ {self.fps:.1f} fps)"


def is_raspberry_pi():
    try:
        with open("/proc/device-tree/model", "rb") as f:
            return b"Raspberry Pi" in f.read()
    except OSError:
        return False


def open_source(cfg, video=None, realtime=True):
    """Pick the right backend automatically.

    Accepts either a CameraConfig or a full Config (whose .camera it uses), so
    callers do not have to remember which one to hand over.
    """
    cfg = getattr(cfg, "camera", cfg)
    if video is not None:
        return VideoFileSource(video, cfg, realtime=realtime)
    if is_raspberry_pi():
        try:
            return PiCameraSource(cfg)
        except Exception as exc:  # falls back to a USB cam on the Pi
            print(f"[camera] Picamera2 unavailable ({exc}); falling back to OpenCV")
    return OpenCVSource(cfg)
