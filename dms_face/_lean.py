"""
Lean-install shim: let MediaPipe import without matplotlib.

Why this exists
---------------
`import mediapipe` executes `mediapipe/tasks/python/vision/drawing_utils.py`,
whose line 21 is a hard `import matplotlib.pyplot as plt`. That module is used
by exactly one helper (`plot_landmarks`, a 3-D debug plot) which this project
never calls. But because the import is unconditional, pip declares matplotlib
a required dependency of mediapipe, and matplotlib drags in pillow, fonttools,
kiwisolver, contourpy, cycler, pyparsing, python-dateutil, packaging and six.

Measured in a clean venv, dropping that chain is ~77 MB of installed size
(323 MB -> 246 MB), which matters on a Raspberry Pi's SD card.

What it does
------------
If matplotlib is genuinely absent, register two stub modules so the import
succeeds. Using anything on them raises an AttributeError that says exactly
what happened, so a future MediaPipe release that starts using matplotlib for
real fails loudly instead of misbehaving silently.

It is an AttributeError rather than an ImportError on purpose. Code probes
modules with hasattr(module, "feature") all the time, and hasattr only treats
AttributeError as "not there". Raising anything else makes the stub crash a
harmless feature check instead of answering it truthfully. (A first version
raised ImportError and did exactly that.)

If matplotlib IS installed, this does nothing at all.

The install script's end-to-end check exercises the real FaceLandmarker, so a
MediaPipe upgrade that breaks this shim is caught at install time.
"""

import importlib.util
import sys
import types

_MESSAGE = (
    "matplotlib is not installed and was replaced by a stub (dms_face lean "
    "install). MediaPipe's plotting helpers are unavailable. "
    "If you need them: pip install matplotlib"
)


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        # Dunder lookups (__path__, __wrapped__, ...) come from the import
        # machinery and inspection tools; they must behave like a normal
        # module or unrelated code breaks.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise AttributeError(_MESSAGE)


def matplotlib_available():
    """True when a real matplotlib can be imported."""
    if "matplotlib" in sys.modules and not isinstance(sys.modules["matplotlib"], _Stub):
        return True
    try:
        return importlib.util.find_spec("matplotlib") is not None
    except (ImportError, ValueError):
        return False


def install_matplotlib_stub(force=False):
    """Register the stubs. Returns True if they were installed."""
    if not force and matplotlib_available():
        return False

    root = _Stub("matplotlib")
    root.__path__ = []  # mark as a package so `import matplotlib.pyplot` resolves
    pyplot = _Stub("matplotlib.pyplot")
    root.pyplot = pyplot
    sys.modules["matplotlib"] = root
    sys.modules["matplotlib.pyplot"] = pyplot
    return True
