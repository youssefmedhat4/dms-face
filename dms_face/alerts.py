"""
Alert output.

Three backends, picked automatically: GPIO buzzer on the Pi, system beep on a
development machine, console otherwise. Alerts are raised on a worker thread so
a 300 ms tone never stalls the vision loop.
"""

import platform
import queue
import threading


class AlertBackend:
    def emit(self, severity):
        raise NotImplementedError

    def close(self):
        pass

    name = "none"


class ConsoleBackend(AlertBackend):
    name = "console"

    def emit(self, severity):
        print("\a" + ("!" * severity) + f" ALERT (severity {severity})", flush=True)


class WinsoundBackend(AlertBackend):
    name = "winsound"

    def __init__(self):
        import winsound

        self._ws = winsound

    def emit(self, severity):
        if severity >= 3:
            for _ in range(3):
                self._ws.Beep(1200, 180)
        else:
            self._ws.Beep(800, 250)


class GpioBuzzerBackend(AlertBackend):
    name = "gpio"

    def __init__(self, pin=18):
        from gpiozero import TonalBuzzer  # noqa: Pi only
        from gpiozero.tones import Tone

        self._Tone = Tone
        self._buzzer = TonalBuzzer(pin)

    def emit(self, severity):
        import time

        if severity >= 3:
            pattern = [(1200, 0.18), (0, 0.08)] * 3
        else:
            pattern = [(800, 0.25)]
        for freq, dur in pattern:
            if freq:
                self._buzzer.play(self._Tone(frequency=freq))
            else:
                self._buzzer.stop()
            time.sleep(dur)
        self._buzzer.stop()

    def close(self):
        try:
            self._buzzer.close()
        except Exception:
            pass


def _pick_backend(prefer=None):
    if prefer == "console":
        return ConsoleBackend()
    if prefer in (None, "auto", "gpio"):
        try:
            return GpioBuzzerBackend()
        except Exception:
            if prefer == "gpio":
                print("[alerts] GPIO buzzer unavailable; falling back")
    if platform.system() == "Windows":
        try:
            return WinsoundBackend()
        except Exception:
            pass
    return ConsoleBackend()


class Alerter:
    """Non-blocking alert dispatcher."""

    def __init__(self, prefer=None, enabled=True):
        self.enabled = enabled
        self.backend = _pick_backend(prefer) if enabled else AlertBackend()
        self._q = queue.Queue(maxsize=4)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="alerts")
        self._thread.start()
        self.count = 0

    def _loop(self):
        while self._running:
            try:
                severity = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if severity is None:
                break
            try:
                self.backend.emit(severity)
            except Exception as exc:
                print(f"[alerts] backend failed: {exc}")

    def alert(self, severity=2):
        if not self.enabled:
            return
        self.count += 1
        try:
            self._q.put_nowait(severity)
        except queue.Full:
            pass  # already saturated with alerts; dropping one is correct

    def close(self):
        self._running = False
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)
        self.backend.close()
