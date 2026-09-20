#!/usr/bin/env bash
#
# Raspberry Pi 5 setup for the DMS face subsystem.
#
#   chmod +x install_pi.sh && ./install_pi.sh
#
# Takes a few minutes, almost all of it downloading wheels.
#
# ---------------------------------------------------------------------------
# Why this script pins nothing and installs plain `mediapipe`:
#
# MediaPipe's ARM64 packaging changed twice, and it breaks nearly every
# tutorial you will find online:
#
#   <= 0.10.18   aarch64 wheels, has the legacy `mp.solutions` API
#   0.10.21-0.10.35   NO aarch64 wheels at all
#   >= 1.0.0     aarch64 wheel returns, `mp.solutions` REMOVED
#
# This project uses the Tasks API (`mediapipe.tasks`), so it wants >= 1.0.0 and
# works on both Bookworm (Python 3.11) and Trixie (Python 3.13). If you find a
# tutorial starting with `mp.solutions.face_mesh.FaceMesh(...)`, it is written
# against a version that no longer exists on this architecture.
# ---------------------------------------------------------------------------

set -euo pipefail

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$1"; }
die()  { printf '\n\033[1;31m!! %s\033[0m\n' "$1" >&2; exit 1; }

cd "$(dirname "$0")"

# --- sanity checks ---------------------------------------------------------
say "Checking the machine"
ARCH="$(uname -m)"
echo "    architecture : $ARCH"
[ "$ARCH" = "aarch64" ] || warn "expected aarch64. On a 32-bit Pi OS there is no mediapipe wheel."

if [ -r /proc/device-tree/model ]; then
    echo "    board        : $(tr -d '\0' < /proc/device-tree/model)"
fi
echo "    python       : $(python3 --version)"

# --- system packages -------------------------------------------------------
say "Installing system packages"
sudo apt update
sudo apt install -y python3-venv python3-pip python3-picamera2

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
pip install --upgrade pip
pip install "mediapipe>=1.0.0" numpy psutil

# opencv-python includes the GUI, which you need for the HUD window and for
# camera calibration. Swap to opencv-python-headless only if the Pi runs with
# no desktop -- then always pass --no-display.
pip install opencv-python

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
import mediapipe as mp

print(f"    mediapipe    : {mp.__version__}")
if hasattr(mp, "solutions"):
    print("    NOTE: this build still has the legacy mp.solutions API.")
    print("          Harmless - this project does not use it.")
from mediapipe.tasks.python import vision
assert vision.FaceLandmarker
print("    FaceLandmarker: OK")

import cv2
print(f"    opencv       : {cv2.__version__}")
print(f"    GUI support  : {'yes' if hasattr(cv2, 'imshow') else 'NO - use --no-display'}")
PY

say "Running the unit tests (no camera needed)"
python tests/test_core.py

say "Running an end-to-end check with no camera"
python tests/make_test_video.py
python run.py --video tests/assets/static_test.mp4 --fast --no-display \
              --no-alerts --no-calibrate 2>/dev/null | tail -8

cat <<'EOF'

============================================================
  Install finished.

  Every session starts with:      source .venv/bin/activate

  1. Baseline performance (no camera needed):
         python bench.py --source image --resolutions --json bench_pi5.json

  2. Live, with the camera. Calibrates automatically on the
     first run -- 10 s eyes open, then 3 s eyes closed:
         python run.py

  3. Camera intrinsics. Do this before trusting any pitch
     angle, and treat it as MANDATORY on a wide-angle lens:
         python calibrate_camera.py

  Press q to quit. It will not close on its own.
============================================================
EOF
