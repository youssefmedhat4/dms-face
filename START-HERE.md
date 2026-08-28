# Start here

Face-based driver monitoring for a Raspberry Pi 5. Eye closure, blink
classification, PERCLOS, gaze and head pose from **one** neural network
inference per frame — everything after that is arithmetic.

## Install on the Pi

```bash
unzip dms-face-subsystem.zip
cd dms-face
chmod +x install_pi.sh
./install_pi.sh
```

The script installs everything, runs 34 unit tests, and does an end-to-end
check on a bundled clip **without needing the camera** — so if something is
wrong you find out before hardware is in the picture.

## Then run it

```bash
source .venv/bin/activate     # needed at the start of every session
python run.py
```

On the very first run it calibrates automatically: **10 seconds with your eyes
open** (sit as you would while driving), then **3 seconds with them closed**.
That step is not optional — a generic EAR threshold does not transfer between
people, and without a pose baseline the eye channel gets gated off entirely.

Press **q** to quit. It will not close by itself.

## Try these

| Do this | Expect |
|---|---|
| Blink normally | stays `ALERT`, blink counter rising |
| One slow 1-second blink | `WARNING` |
| Eyes shut ~2 seconds | `DROWSY` + buzzer |
| Turn your head ~40° | eye channel shows `GATED` — that is correct, not a fault |
| Look down at your lap | `DISTRACTED` |
| Cover the camera | `UNAVAILABLE: no face detected` |

PERCLOS reads `warming N/10s` for the first ten seconds. That is deliberate: it
is a ratio over a time window, and on frame one a single closed sample would
read as 100%.

## It does not work equally well on every face

Calibration normalises to each driver's own eyes, which handles most of the
variation between people. But it cannot manufacture signal. After calibrating,
the system reports a rating:

    signal quality=good (SNR 69.7)

`good` means the eye signal is clean. `fair` means usable with occasional
flicker. `poor` means blink and PERCLOS results are unreliable for that person,
and no threshold tweak will fix it -- the measurement itself is too close to
the noise.

Drivers most likely to score `fair` or `poor`: smaller eye openings (epicanthic
folds, monolid structure, ptosis, heavy lashes), glasses, poor or uneven
lighting, or sitting too far from the camera. Try better lighting and a closer
camera first -- both help more than any threshold change.

**Test it on several different people, not just yourself,** and note the rating
for each. That distribution belongs in the report.

## Two things to do before trusting the numbers

**1. Check the eye landmarks.** Press `i` while running. Six dots should sit on
each eyelid's corners and mid-points. Those indices are a community convention,
not an official Google specification — verify them once on your own face.

**2. Calibrate the camera.** Run `python calibrate_camera.py` with a printed
chessboard. Measured on one fixed frame, changing *only* the assumed focal
length moved pitch from 38.1° to 6.5° while yaw barely moved 0.3°. Pitch is
largely an artefact of that guess. On a wide-angle lens this is mandatory, not
advisory.

## If something looks wrong

Everything is logged per frame:

```bash
python run.py --log session.csv
```

That CSV has raw EAR, filtered EAR, openness, eye state, validity flags, all
three head angles, gaze deviation and both PERCLOS windows — enough to diagnose
a threshold problem without guessing.

Full documentation, measured benchmarks and the known limitations are in
`README.md`.
