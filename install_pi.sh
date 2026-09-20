#!/usr/bin/env bash
#
# Raspberry Pi 5 setup for the DMS face subsystem -- the lean install.
#
#   chmod +x install_pi.sh && ./install_pi.sh
#
# Options (environment variables):
#   WITH_GUI=1        install the GUI build of OpenCV, for the live HUD window
#                     on a Pi with a desktop. Default is the headless build:
#                     smaller, and the right choice over SSH.
#   SKIP_CAMERA=1     do not install picamera2 (nothing here needs it until a
#                     camera is attached).
#
# ---------------------------------------------------------------------------
# What "lean" means, and why each choice was made (measured, not assumed):
#
#  * mediapipe is installed with --no-deps. Its declared dependencies include
#    matplotlib (+ pillow, fonttools, kiwisolver, ...) and a second copy of
#    OpenCV (opencv-contrib-python). We need neither; see requirements-lean.txt
#    and dms_face/_lean.py. Clean-venv size: 323 MB full -> 246 MB lean.
#  * OpenCV is chosen explicitly, once. Installing opencv-python on top of
#    mediapipe's own opencv-contrib-python (what the previous version of this
#    script did) puts two packages' files in one directory.
#  * The headless OpenCV wheel is ~36-40 MB against ~50 MB for the GUI one on
#    aarch64 (PyPI), and it drops the bundled Qt.
#  * --no-cache-dir: pip would otherwise keep a copy of every wheel in
#    ~/.cache/pip -- roughly another 100 MB on the SD card, for nothing.
#  * --no-install-recommends: python3-picamera2 recommends python3-pyqt5 and
#    python3-opengl (Raspberry Pi apt index), which only serve its preview
#    window. We never open one.
#  * No system python3-pip: the venv brings its own pip.
#
# MediaPipe's ARM64 packaging, for anyone wondering why this pins nothing:
#   <= 0.10.18   aarch64 wheels, has the legacy `mp.solutions` API
#   0.10.21-0.10.35   NO aarch64 wheels at all
#   >= 1.0.0     aarch64 wheel returns, `mp.solutions` REMOVED
# This project uses the Tasks API, so it wants >= 1.0.0, and works on both
# Bookworm (Python 3.11) and Trixie (Python 3.13).
# ---------------------------------------------------------------------------

set -euo pipefail

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$1"; }
die()  { printf '\n\033[1;31m!! %s\033[0m\n' "$1" >&2; exit 1; }

cd "$(dirname "$0")"

WITH_GUI="${WITH_GUI:-0}"
SKIP_CAMERA="${SKIP_CAMERA:-0}"

# --- sanity checks ---------------------------------------------------------
say "Checking the machine"
ARCH="$(uname -m)"
echo "    architecture : $ARCH"
[ "$ARCH" = "aarch64" ] || warn "expected aarch64. On a 32-bit Pi OS there is no mediapipe wheel."

if [ -r /proc/device-tree/model ]; then
    echo "    board        : $(tr -d '\0' < /proc/device-tree/model)"
fi
echo "    python       : $(python3 --version)"
echo "    free disk    : $(df -h --output=avail . | tail -1 | tr -d ' ')"

# --- system packages -------------------------------------------------------
#
# Only touches the system when something is actually missing. The previous
# version ran `sudo apt update` unconditionally, which fails outright on a Pi
# where sudo needs a password -- even when every package was already installed
# (as on a stock Raspberry Pi OS Desktop image).
say "Checking system packages"
NEED_APT=()

# Do not trust `import venv`: on Debian the module imports fine without the
# python3-venv package, and only creating a venv shows whether it really works.
probe="$(mktemp -d)"
if python3 -m venv "$probe/v" >/dev/null 2>&1; then
    echo "    python venv support : present"
else
    NEED_APT+=(python3-venv)
fi
rm -rf "$probe"

if [ "$SKIP_CAMERA" = "1" ]; then
    echo "    picamera2           : skipped (SKIP_CAMERA=1)"
elif python3 -c "import picamera2" 2>/dev/null; then
    echo "    picamera2           : present"
else
    NEED_APT+=(python3-picamera2)
fi

if [ "${#NEED_APT[@]}" -gt 0 ]; then
    echo "    missing             : ${NEED_APT[*]}  (needs sudo)"
    sudo apt update
    sudo apt install -y --no-install-recommends "${NEED_APT[@]}"
else
    echo "    nothing to install - no sudo needed"
fi

# --- virtualenv ------------------------------------------------------------
# --system-site-packages is required: picamera2 ships C++ bindings via apt and
# pip cannot build it, so the venv has to be able to see the system copy.
say "Creating the virtual environment"
if [ ! -d .venv ]; then
    python3 -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- python packages -------------------------------------------------------
say "Installing Python packages (this is the slow part)"
warn "Expect pip to print 'ERROR: pip's dependency resolver ... mediapipe requires"
warn "matplotlib / opencv-contrib-python / sounddevice, which is not installed'."
warn "That is intended: those are the packages this lean install skips on purpose."
warn "It exits 0, and the verification below proves nothing was needed."
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir --no-deps "mediapipe>=1.0.0"
pip install --no-cache-dir -r requirements-lean.txt

if [ "$WITH_GUI" = "1" ]; then
    echo "    OpenCV: GUI build (WITH_GUI=1)"
    pip install --no-cache-dir opencv-python
else
    echo "    OpenCV: headless build. For the live HUD on a desktop, re-run with WITH_GUI=1."
    pip install --no-cache-dir opencv-python-headless
fi

# --- model bundle ----------------------------------------------------------
say "Checking the model bundle"
if [ -f models/face_landmarker.task ]; then
    echo "    already present ($(du -h models/face_landmarker.task | cut -f1))"
else
    echo "    downloading..."
    curl -fL -o models/face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
      || die "model download failed - check the network connection"
fi

# --- verification ----------------------------------------------------------
say "Verifying the install"
python - <<'PY'
import sys
sys.path.insert(0, ".")

import numpy
import cv2

# Order matters: importing dms_face.landmarker installs the matplotlib stub,
# and `import mediapipe` fails on a lean install without it.
from dms_face import landmarker  # noqa: F401
from dms_face._lean import matplotlib_available

import mediapipe as mp
from mediapipe.tasks.python import vision

assert vision.FaceLandmarker
print(f"    mediapipe    : {mp.__version__}")
print(f"    FaceLandmarker: OK")
print(f"    opencv       : {cv2.__version__}")
print(f"    numpy        : {numpy.__version__}")
print(f"    matplotlib   : {'installed' if matplotlib_available() else 'not installed (stubbed, as intended)'}")

import run
ok, why = run.gui_available()
print(f"    window support: {'yes' if ok else 'no - ' + why}")
PY

say "Running the unit tests (no camera needed)"
python tests/test_core.py

say "Running an end-to-end check with no camera"
python tests/make_test_video.py
if ! OUT="$(python run.py --video tests/assets/static_test.mp4 --fast --no-display \
                          --no-alerts --no-calibrate 2>&1)"; then
    echo "$OUT" | tail -30
    die "end-to-end check crashed - the output above is the traceback"
fi
echo "$OUT" | grep -E "frames|mean FPS|face detected|blinks|alerts raised"

DETECTED="$(echo "$OUT" | awk '/face detected/ {gsub(/%/,"",$3); print int($3)}')"
if [ -z "$DETECTED" ] || [ "$DETECTED" -lt 90 ]; then
    die "the landmark model found a face in only ${DETECTED:-0}% of a clip where one is always present"
fi

say "Footprint"
echo "    virtualenv   : $(du -sh .venv | cut -f1)"
echo "    project      : $(du -sh --exclude=.venv --exclude=.git . | cut -f1)"
echo "    pip cache    : $(du -sh "$HOME/.cache/pip" 2>/dev/null | cut -f1 || echo none) (untouched: --no-cache-dir)"

cat <<'EOF'

============================================================
  Install finished.

  Every session starts with:      source .venv/bin/activate

  1. Baseline performance (no camera needed):
         pip install --no-cache-dir psutil     # optional, for CPU/RSS
         python bench.py --source image --resolutions --json bench_pi5.json

  2. Live, with the camera. Calibrates automatically on the
     first run -- 10 s eyes open, then 3 s eyes closed:
         python run.py

  3. Camera intrinsics. Do this before trusting any pitch
     angle, and treat it as MANDATORY on a wide-angle lens:
         python calibrate_camera.py

  Over SSH there is no window: run.py detects that and prints
  state changes to the terminal instead.

  Press q to quit a window. It will not close on its own.
============================================================
EOF
