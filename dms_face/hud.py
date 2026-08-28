"""
Debug overlay.

Deliberately dense: while tuning thresholds you need to see the raw signal, the
filtered signal, the state machine and the validity flags simultaneously.
Disable it with --no-display when measuring real throughput -- drawing and
imshow cost more than the inference does.
"""

import cv2
import numpy as np

from . import landmark_ids as L
from .blink import EyeState
from .state import DriverState

# BGR, because OpenCV.
COL_BG = (26, 20, 16)
COL_TEXT = (235, 235, 230)
COL_DIM = (150, 150, 145)
COL_OK = (110, 190, 110)
COL_WARN = (60, 180, 235)
COL_CRIT = (70, 70, 235)
COL_ACCENT = (200, 170, 60)
COL_INVALID = (110, 110, 200)

STATE_COLOURS = {
    DriverState.ALERT: COL_OK,
    DriverState.UNAVAILABLE: COL_DIM,
    DriverState.DISTRACTED: COL_WARN,
    DriverState.WARNING: COL_WARN,
    DriverState.DROWSY: COL_CRIT,
}


def _text(img, s, org, colour=COL_TEXT, scale=0.45, thickness=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def _panel(img, x, y, w, h, alpha=0.72):
    sub = img[y : y + h, x : x + w]
    if sub.size == 0:
        return
    overlay = np.full(sub.shape, COL_BG, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)


class Hud:
    def __init__(self, width, height, history=180):
        self.width = width
        self.height = height
        self.history = history
        self._openness_trace = []
        self._show_mesh = False
        self._show_indices = False

    def toggle_mesh(self):
        self._show_mesh = not self._show_mesh

    def toggle_indices(self):
        self._show_indices = not self._show_indices

    def draw(self, frame_rgb, result, profile, cfg):
        img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if result.landmarks is not None:
            self._draw_eyes(img, result)
            if self._show_mesh:
                self._draw_mesh(img, result.landmarks.pixels)
            if self._show_indices:
                self._draw_indices(img, result.landmarks.pixels)

        self._openness_trace.append(result.openness if result.openness is not None else -1.0)
        if len(self._openness_trace) > self.history:
            self._openness_trace.pop(0)

        self._draw_left_panel(img, result, profile, cfg)
        self._draw_state_banner(img, result)
        self._draw_trace(img, cfg)
        return img

    # -- components ---------------------------------------------------------
    def _draw_eyes(self, img, result):
        pts = result.landmarks.pixels
        colour = COL_OK if result.eye_channel_valid else COL_INVALID
        if result.eye_state is EyeState.CLOSED:
            colour = COL_CRIT

        for idx in (L.RIGHT_EYE_EAR, L.LEFT_EYE_EAR):
            poly = np.array([pts[i] for i in idx], dtype=np.int32)
            cv2.polylines(img, [poly], True, colour, 1, cv2.LINE_AA)

        if result.landmarks.has_iris:
            for c in (L.RIGHT_IRIS_CENTER, L.LEFT_IRIS_CENTER):
                x, y = pts[c]
                cv2.circle(img, (int(x), int(y)), 2, COL_ACCENT, -1, cv2.LINE_AA)

        # Head-pose axes from the nose tip: the fastest way to spot a bad
        # solvePnP solution is to watch the axes flip.
        if result.angles is not None:
            self._draw_axes(img, pts[1], result.angles)

    def _draw_axes(self, img, origin, angles, length=60):
        yaw, pitch, roll = (np.radians(a) for a in angles)
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        axes = np.float32([[length, 0, 0], [0, length, 0], [0, 0, length]])
        ox, oy = int(origin[0]), int(origin[1])
        for vec, colour in zip(axes, [(80, 80, 240), (80, 240, 80), (240, 160, 80)]):
            p = R @ vec
            cv2.line(img, (ox, oy), (ox + int(p[0]), oy + int(p[1])), colour, 2, cv2.LINE_AA)

    def _draw_mesh(self, img, pts):
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), 1, (90, 90, 90), -1)

    def _draw_indices(self, img, pts):
        """Prove the EAR landmark indices are the right ones. The index sets
        are a community convention, not an official spec -- verify once."""
        for name, idx in (("R", L.RIGHT_EYE_EAR), ("L", L.LEFT_EYE_EAR)):
            for n, i in enumerate(idx, start=1):
                x, y = pts[i]
                cv2.circle(img, (int(x), int(y)), 3, COL_ACCENT, -1, cv2.LINE_AA)
                _text(img, f"{name}p{n}:{i}", (int(x) + 5, int(y) - 4), COL_ACCENT, 0.32)

    def _draw_left_panel(self, img, result, profile, cfg):
        lines = []
        st = result.stats
        lines.append(("FPS", f"{result.fps:5.1f}", COL_TEXT))
        lines.append(("inference", f"{result.inference_ms:5.1f} ms", COL_TEXT))
        lines.append(("total", f"{result.total_ms:5.1f} ms", COL_TEXT))
        lines.append((None, None, None))

        if result.ear_raw is not None:
            lines.append(("EAR raw", f"{result.ear_raw:.4f}", COL_DIM))
            lines.append(("EAR filt", f"{result.ear_filtered:.4f}", COL_TEXT))
        else:
            lines.append(("EAR", "--", COL_DIM))

        if result.openness is not None:
            oc = COL_CRIT if result.openness < cfg.eye.close_openness else COL_TEXT
            lines.append(("openness", f"{result.openness:.3f}", oc))
        else:
            lines.append(("openness", "--", COL_DIM))

        lines.append(
            (
                "baseline",
                f"{st.get('ear_open', 0):.3f}/{st.get('ear_closed', 0):.3f}",
                COL_WARN if st.get("adaptation_frozen") else COL_DIM,
            )
        )
        lines.append(("eye state", result.eye_state.value, COL_TEXT))

        gate = "valid" if result.eye_channel_valid else "GATED (head angle)"
        lines.append(("eye channel", gate, COL_OK if result.eye_channel_valid else COL_INVALID))
        lines.append((None, None, None))

        rep = result.report
        if rep is not None:
            if rep.perclos_warming:
                pc = f"warming {rep.perclos_span:.0f}/{cfg.perclos.min_window_s:.0f}s"
            elif rep.perclos_valid:
                pc = f"{rep.perclos:.1%}"
            else:
                pc = "unavailable"
            pcol = COL_CRIT if rep.perclos >= cfg.perclos.drowsy_threshold else (
                COL_WARN if rep.perclos >= cfg.perclos.warn_threshold else COL_TEXT
            )
            lines.append(("PERCLOS 60s", pc, pcol if rep.perclos_valid else COL_DIM))
            lines.append(("PERCLOS 20s", f"{rep.perclos_fast:.1%}", COL_DIM))
            lines.append(("validity", f"{rep.validity:.0%}", COL_DIM))
            if rep.blink_rate is not None:
                lines.append(("blink rate", f"{rep.blink_rate:.1f}/min", COL_DIM))
            if rep.mean_blink_ms is not None:
                lines.append(("mean blink", f"{rep.mean_blink_ms:.0f} ms", COL_DIM))
            if rep.closure_ms > 0:
                lines.append(("closed for", f"{rep.closure_ms:.0f} ms", COL_CRIT))
        lines.append((None, None, None))

        if result.angles is not None:
            y, p, r = result.angles
            lines.append(("yaw", f"{y - profile.yaw0:+6.1f} deg", COL_TEXT))
            lines.append(("pitch", f"{p - profile.pitch0:+6.1f} deg", COL_TEXT))
            lines.append(("roll", f"{r - profile.roll0:+6.1f} deg", COL_TEXT))
        if result.angles_mp is not None:
            lines.append(("mp yaw/pitch", "%.0f/%.0f" % result.angles_mp[:2], COL_DIM))
        if result.gaze_dev is not None:
            gc = COL_WARN if result.gaze_dev > cfg.gaze.deviation_threshold else COL_TEXT
            lines.append(("gaze dev", f"{result.gaze_dev:.3f}", gc))

        if not st.get("calibrated_camera", False):
            lines.append((None, None, None))
            lines.append(("camera", "UNCALIBRATED", COL_WARN))

        pad = 10
        row = 17
        h = pad * 2 + row * len(lines)
        _panel(img, 0, 0, 210, min(h, self.height))

        y = pad + 12
        for label, value, colour in lines:
            if label is None:
                y += 6
                continue
            _text(img, label, (pad, y), COL_DIM, 0.38)
            _text(img, value, (pad + 96, y), colour, 0.40)
            y += row

    def _draw_state_banner(self, img, result):
        rep = result.report
        state = rep.state if rep is not None else DriverState.UNAVAILABLE
        colour = STATE_COLOURS.get(state, COL_DIM)
        h = 34
        y0 = self.height - h
        _panel(img, 0, y0, self.width, h, alpha=0.82)
        cv2.rectangle(img, (0, y0), (6, self.height), colour, -1)
        label = state.value.upper()
        _text(img, label, (16, y0 + 23), colour, 0.62, 2)
        if rep is not None and rep.reasons:
            _text(img, " | ".join(rep.reasons[:3]), (150, y0 + 22), COL_TEXT, 0.42)

    def _draw_trace(self, img, cfg):
        """Openness over time, with both hysteresis thresholds drawn in.

        Watching the signal cross the two lines is the fastest way to tell
        whether your thresholds are right for this driver and this lighting.
        """
        w, h = 240, 74
        x0 = self.width - w - 8
        y0 = 8
        _panel(img, x0, y0, w, h)
        _text(img, "openness", (x0 + 8, y0 + 14), COL_DIM, 0.36)

        plot_y = y0 + 20
        plot_h = h - 28

        for value, colour, tag in (
            (cfg.eye.close_openness, COL_CRIT, "close"),
            (cfg.eye.open_openness, COL_OK, "open"),
        ):
            yy = int(plot_y + plot_h * (1.0 - value))
            cv2.line(img, (x0 + 6, yy), (x0 + w - 40, yy), colour, 1, cv2.LINE_AA)
            _text(img, tag, (x0 + w - 36, yy + 4), colour, 0.32)

        trace = self._openness_trace
        if len(trace) > 1:
            step = (w - 46) / max(1, self.history - 1)
            pts = []
            for i, v in enumerate(trace):
                if v < 0:
                    pts.append(None)
                    continue
                pts.append((int(x0 + 6 + i * step), int(plot_y + plot_h * (1.0 - v))))
            for a, b in zip(pts, pts[1:]):
                if a is not None and b is not None:
                    cv2.line(img, a, b, COL_ACCENT, 1, cv2.LINE_AA)
