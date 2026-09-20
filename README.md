# Face subsystem — Real-Time Driver Monitoring System

Eye closure, blink classification, PERCLOS, gaze and head pose from **one**
neural network inference per frame. Everything downstream is arithmetic.

Runs identically on a laptop webcam (development) and a Raspberry Pi 5 with
`picamera2` (deployment). No dlib, no PyTorch, no TensorFlow.

---

## Quick start

```bash
pip install -r requirements.txt
python tests/test_core.py          # 41 tests, no camera needed
python run.py --calibrate          # calibrate, then run live
```

On the Raspberry Pi:

```bash
./install_pi.sh
```

---

## Read this before you install MediaPipe

MediaPipe's ARM64 packaging changed twice, and it breaks almost every tutorial
you will find. Verified from the PyPI release manifest and by opening the wheels:

| Version | ARM64 wheel | `mediapipe.solutions` |
|---|---|---|
| ≤ 0.10.18 (Nov 2024) | yes, `cp39`–`cp312` | present |
| 0.10.21 – 0.10.35 | **none at all** | present |
| ≥ 1.0.0 (Jul 2026) | yes, `py3-none` | **removed** |

So on any current install, this raises `AttributeError`:

```python
mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)   # dead
```

This project uses the Tasks API instead, which is the supported path and works
on both Bookworm (Python 3.11) and Trixie (Python 3.13):

```python
from mediapipe.tasks.python import vision
vision.FaceLandmarker.create_from_options(...)
```

---

## Measured performance

Fixed-image benchmark, `python bench.py --source image --resolutions`, so the
numbers isolate compute rather than camera behaviour.

**Development machine** (Intel, 32 cores, Python 3.14, mediapipe 1.0.1):

| Input resolution | FPS | Inference (mean) | CPU | RSS |
|---|---|---|---|---|
| 320×240 | 75.3 | 12.4 ms | 102% | 120 MB |
| 480×360 | 54.1 | 17.7 ms | 105% | 123 MB |
| 640×480 | 55.4 | 17.4 ms | 104% | 125 MB |
| 960×720 | 53.9 | 17.8 ms | 104% | 129 MB |

Two things worth putting in your report:

1. **Resolution above 480×360 buys nothing.** 54 / 55 / 54 FPS across three
   very different input sizes, because MediaPipe resizes internally to 192×192
   (detector) and 256×256 (mesh). Feeding it a larger frame is wasted work.
   This is why the pipeline requests a 640×480 `lores` stream and never
   upscales.
2. **CPU sits at ~104%, i.e. about one core.** The face pipeline does not need
   the AI HAT+. Hailo's own model zoo lists this landmark model at 0.07 GOPS —
   there is no compute problem to accelerate.

Re-run the same command on your Pi 5 and put both columns side by side. That
comparison is a genuine result; a quoted FPS from a blog post is not.

---

## What runs when

| Every frame | At 1 Hz |
|---|---|
| landmark inference | PERCLOS decision |
| EAR both eyes → median filter | state fusion |
| openness normalisation | baseline adaptation (0.1 Hz) |
| blink FSM | |
| PERCLOS ring-buffer update | |
| `solvePnP` → yaw/pitch/roll | |
| iris ratios → gaze deviation | |

A microsleep in progress escalates immediately rather than waiting for the next
1 Hz tick — see `FacePipeline._decide`.

---

## Design decisions you will be asked to defend

**Openness normalisation, not raw EAR.** P80 means "eyelid covers ≥80% of the
pupil". Raw EAR never reaches zero when the eye shuts, so dividing it by a fixed
constant is not P80, it is an arbitrary number. We calibrate both endpoints:

```
openness = (EAR − EAR_closed) / (EAR_open − EAR_closed)
P80      = openness < 0.20
```

**Hysteresis, not a single threshold.** Close below 0.20, reopen above 0.30. A
single threshold makes the eye state chatter whenever the signal rests near it.
`test_single_threshold_would_chatter` demonstrates 4 state changes vs 0.

**Durations in milliseconds, not frames.** The ubiquitous
`EYE_AR_CONSEC_FRAMES = 16` silently encodes a frame rate — 0.53 s at 30 fps but
3.2 s at 5 fps. On a loaded Pi that changes meaning underneath you.
`test_duration_is_measured_in_time_not_frames` pins this.

**Invalid frames leave both numerator and denominator.** Counting "no face" as
"eyes open" is the classic bug that makes PERCLOS under-report exactly when the
driver has slumped out of frame. Below 60% validity the system reports
`MONITORING_UNAVAILABLE` rather than a confidently wrong number.

**EAR is gated on head pose.** EAR is scale-invariant but *not* rotation
invariant: yaw foreshortens the horizontal denominator (a closed eye reads
open), pitch-down shrinks the vertical numerator (an open eye reads closed —
the classic false alarm when a driver glances at the gear stick). Past 30° yaw
the eye channel is marked invalid rather than trusted.
`test_ear_is_not_rotation_invariant` documents the effect.

**A sanity guard on implausible closures.** A real microsleep ends. An unbroken
60-second "closure" means the openness scale is wrong — almost always an
uncalibrated profile. The system reports `UNAVAILABLE` with the reason instead
of `DROWSY`. This was added after the pipeline confidently reported DROWSY on a
still photograph of an open-eyed face during testing.

**Alert at 1.0–1.5 s, not 2–3 s.** Normal blinks are 100–400 ms; alert total
blink duration is 265 ± 57 ms, sleep-deprived 586 ± 592 ms. At 100 km/h, 2 s of
closed eyes is 55 m of road travelled blind.

---

## Head pose and why camera calibration matters

Measured on one fixed frame, sweeping **only** the assumed focal length:

| focal | yaw | pitch | roll |
|---|---|---|---|
| 400 | −1.4 | **38.1** | −1.4 |
| 820 | −1.6 | **21.1** | −1.0 |
| 2000 | −1.7 | **9.4** | −0.7 |
| 3000 | −1.7 | **6.5** | −0.6 |

Pitch is almost entirely an artefact of the guess. Yaw and roll barely move.
Consequences, all implemented:

- yaw and roll are usable uncalibrated;
- **absolute** pitch is not;
- pitch **relative to the driver's calibrated neutral** still is, because a
  constant bias cancels in the subtraction — so the distraction gate uses
  `pitch − pitch₀`, never raw pitch.

Run `python calibrate_camera.py` to remove the guess entirely. On the 160°
IR module in your BOM this is mandatory, not optional: barrel distortion at
that field of view will corrupt `solvePnP` regardless of focal length.

Two independent head-pose estimates are computed and both shown in the HUD:
`solvePnP` on 22 rigid landmarks, and MediaPipe's own facial transformation
matrix. Agreement is a validation result worth reporting; disagreement in pitch
alone is the intrinsics problem above.

---

## Command reference

```bash
# live, with the debug HUD
python run.py

# calibrate this driver first (15 s, two phases)
python run.py --calibrate --profile youssef

# repeatable evaluation on a recording — a live webcam gives a different
# answer every run, which is useless for a results chapter
python run.py --video clip.mp4 --log results.csv

# headless throughput
python run.py --no-display --duration 60

# IR mode: replicates mono to 3 channels and applies CLAHE
python run.py --ir

# benchmark
python bench.py --source image --resolutions --json bench_pi5.json

# camera intrinsics
python calibrate_camera.py
```

Keys while running: `q` quit · `c` recalibrate · `m` full mesh ·
`i` show EAR landmark indices · `r` reset counters · `s` snapshot · `space` pause.

**Press `i` once on your own face.** The EAR landmark index sets are a community
convention, not an official Google specification. Verify them visually before
you rely on them.

---

## Layout

```
dms_face/
  landmark_ids.py  index sets + canonical 3D model (from Google's .obj)
  config.py        every tunable, with the reasoning for each value
  camera.py        Picamera2 / OpenCV / video-file sources, threaded
  landmarker.py    the one neural network
  _lean.py         lets mediapipe import without matplotlib (lean install)
  metrics.py       EAR, openness, iris ratios
  headpose.py      solvePnP, angle decomposition, flip handling
  blink.py         four-state FSM, closure classification
  perclos.py       P80, dual window, validity accounting
  calibration.py   driver profile, two-phase routine, baseline adaptation
  filters.py       median, EMA, hysteresis, debounce, time windows
  state.py         fusion into ALERT/DISTRACTED/WARNING/DROWSY/UNAVAILABLE
  alerts.py        GPIO buzzer / system beep / console
  hud.py           debug overlay
  pipeline.py      wires it together
run.py             main application
bench.py           performance measurement
calibrate_camera.py  chessboard intrinsics
install_pi.sh        lean Raspberry Pi install + self-test
requirements.txt     dev-machine install
requirements-lean.txt  what the Pi install actually needs
tests/test_core.py   41 unit tests
```

---

## Installing lean

`pip install mediapipe` is heavier than this project needs. mediapipe 1.0.1
declares seven dependencies (read from its PyPI metadata):

```
absl-py  certifi  numpy  flatbuffers  sounddevice  matplotlib  opencv-contrib-python
```

Three of them are dead weight here:

| Package | Why it is there | Why we skip it |
|---|---|---|
| `matplotlib` (+ pillow, fonttools, kiwisolver, contourpy, cycler, pyparsing, dateutil, packaging, six) | `drawing_utils.py` line 21 does an unconditional `import matplotlib.pyplot`, for one 3-D debug-plot helper | we never call it |
| `opencv-contrib-python` | provides `cv2` | a second, larger OpenCV that would collide with the one we choose |
| `sounddevice` | MediaPipe's audio tasks | imported inside a `try`; never reached |

`install_pi.sh` therefore installs `mediapipe` with `--no-deps`, then only what
the code imports (`requirements-lean.txt`), then **one** explicitly-chosen
OpenCV. `dms_face/_lean.py` supplies a stub for the single matplotlib import so
`import mediapipe` still succeeds; the stub raises a clear `AttributeError` if
anything actually uses it, and does nothing at all when matplotlib is installed.

**Measured** in clean virtual environments on a Windows machine (absolute sizes
differ on the Pi; the differences are what carry over):

| | old recipe | lean | change |
|---|---|---|---|
| Installed size | 323 MB | 246 MB | **−77 MB (−24%)** |
| Packages installed | 22 | 7 | −15 |
| RAM after loading the model | 131 MB | 89 MB | **−42 MB (−32%)** |
| Start-up to model ready | 0.66 s | 0.29 s | **2.3× faster** |
| Python modules loaded | 616 | 369 | −247 |
| Throughput (1200-frame clip) | 126 FPS | 126–129 FPS | none |

The memory and start-up gains are the more interesting result: in the old
install, `import mediapipe` loads matplotlib into RAM on **every run** for a
plotting function that is never used. Throughput is unchanged, so this is a free
saving, not a trade.

Other choices in the script, and the evidence behind each:

- **`--no-cache-dir`** — pip otherwise keeps a copy of every wheel in
  `~/.cache/pip`, roughly another 100 MB of SD card.
- **`opencv-python-headless`** — ~36–40 MB against ~50 MB for the GUI wheel on
  aarch64 (PyPI file sizes), and it drops the bundled Qt.
- **`apt --no-install-recommends`** — `python3-picamera2` recommends
  `python3-pyqt5` and `python3-opengl` (checked in the Raspberry Pi apt index,
  bookworm and trixie), which serve only its preview window. We never open one.
- **No system `python3-pip`** — the virtualenv brings its own.

### What cannot be trimmed

What remains is the actual work: the MediaPipe runtime (a single 111 MB
`libmediapipe.so` in the aarch64 wheel), OpenCV, and numpy. Going lower would
mean replacing MediaPipe's runtime with a bare TFLite interpreter and
re-implementing its detector and tracking glue by hand. That is a large,
risky rewrite for a saving I would not recommend chasing on a project
timeline.

### Do not install both OpenCV packages

`opencv-python`, `opencv-python-headless` and `opencv-contrib-python` all
install into the same `cv2/` directory. Installing two of them does not error;
whichever finished last silently wins. A plain `pip install mediapipe`
followed by `pip install opencv-python` (which the first version of this
project's script did) leaves exactly that state. If you have it:

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-contrib-python        # or opencv-python-headless on the lean install
```

### Getting the HUD window back

The lean install is headless. On a Pi with its own desktop:

```bash
WITH_GUI=1 ./install_pi.sh
```

or, in an existing environment, swap the one package:

```bash
pip uninstall -y opencv-python-headless && pip install opencv-python
```

---

## Does it work on every face?

No — and the system now measures how well it works for each driver instead of
assuming.

**What calibration does solve.** Openness is normalised by each driver's *own*
open and closed EAR, so people with naturally narrow eyes are not penalised by
a threshold picked for someone else. That absorbs most inter-person variation,
and it is why fixed thresholds (the published values range 0.20 / 0.25 / 0.30)
do not need to be guessed at.

**What it does not solve.** Normalisation rescales the signal, but it cannot
add information. What matters is the driver's open-to-closed range measured
against their landmark jitter:

```
SNR = (EAR_open − EAR_closed) / noise
```

Two drivers can both pass a naive `separation > 0.04` check and behave
completely differently. Calibration now reports this:

| Rating | Meaning |
|---|---|
| `good` | noise fits 4× inside the hysteresis dead band |
| `fair` | fits 2× — usable, expect occasional flicker |
| `poor` | eye-state results are unreliable for this driver |

The bands are derived from the hysteresis dead band (0.30 − 0.20 = 0.10), not
invented. A measured reference on a real adult face: separation 0.2414,
jitter 0.0035, **SNR 69.7 — good**.

**Who is most likely to score `fair` or `poor`:**

- Smaller palpebral aperture — epicanthic folds, monolid eyelid structure,
  ptosis, deep-set eyes, heavy lashes. All compress the range.
- Prescription glasses, which refract the eye region and shift the apparent
  landmark positions; strong IR reflection off the lens is worse still.
- Anyone far from the camera, or in dim or uneven lighting — both raise jitter
  without widening the range.
- Any face under infrared illumination, since MediaPipe's models were trained
  overwhelmingly on visible-light RGB.

A `poor` rating is a property of the measurement, not a threshold to tune
around. For those drivers the honest answer is that EAR geometry is the wrong
instrument, and an appearance-based eye-state classifier — a small CNN trained
on the MRL Eye dataset, which is infrared — would serve them better. That is a
sound justification for the AI HAT+ if you want one.

**For your ethics chapter:** this is a fairness issue, not just an engineering
one. A system that reports a single accuracy figure across all users conceals
that it serves some drivers worse than others. Recording per-driver signal
quality, and saying plainly which groups are most affected, is the defensible
position. Test on as many different faces as you can get, and report the SNR
distribution rather than one number.

## Known limitations

State these in your report; every one will otherwise be asked as a question.

- **Mirrored, polarised or IR-blocking sunglasses defeat the eye channel
  entirely.** The system degrades to head-pose-only and says so. No lightweight
  method solves this.
- **Beyond ~30° yaw the EAR measurement is deliberately discarded**, so eye
  monitoring coverage drops exactly when the driver looks away — which is why
  head pose runs as an independent channel.
- **MediaPipe is not trained on infrared imagery.** Practitioner reports confirm
  degraded but functional grayscale performance. Measure night performance on
  your own hardware; do not assume it.
- **PERCLOS lags by up to its 60 s window** and is weakest at early sleepiness.
  The blink-duration channel exists to cover that gap.
- **There is no ground truth for drowsiness.** You cannot ethically induce
  dangerous drowsiness in a subject. Validate the *components* — blink durations
  against hand-annotated video, head angles against measured physical rotations,
  PERCLOS against scripted eye-closure sequences — and say that is what you did.
- **Single camera, single viewpoint.** A hand at the face or a steering-wheel
  occlusion drops the face; the validity-ratio mechanism reports this rather
  than hiding it.
