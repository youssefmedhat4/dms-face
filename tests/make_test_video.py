#!/usr/bin/env python3
"""
Build the static test clip from the bundled portrait.

Used to verify an install WITHOUT a camera attached, which is the first thing
worth doing on a fresh Raspberry Pi:

    python tests/make_test_video.py
    python run.py --video tests/assets/static_test.mp4 --fast --no-display --no-calibrate

Expect: 100% face detection, state stays ALERT, zero blinks, zero alerts.

The clip is generated rather than shipped because it is 1200 copies of one
still frame -- 4 MB of redundancy in a package that is otherwise small.

Note the letterboxing below. Squashing a 4:5 portrait into a 4:3 frame
compresses the face vertically and drops EAR from 0.204 to 0.147, which is
enough to read as closed eyes. Aspect ratio must be preserved or the test
asset itself becomes the bug.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "assets", "portrait.jpg")
DST = os.path.join(HERE, "assets", "static_test.mp4")

WIDTH, HEIGHT, FPS, SECONDS = 640, 480, 30, 40


def main():
    img = cv2.imread(SRC)
    if img is None:
        sys.exit(f"missing source image: {SRC}")

    h0, w0 = img.shape[:2]
    scale = min(WIDTH / w0, HEIGHT / h0)
    resized = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))

    canvas = np.full((HEIGHT, WIDTH, 3), 32, dtype=np.uint8)
    y = (HEIGHT - resized.shape[0]) // 2
    x = (WIDTH - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized

    writer = cv2.VideoWriter(DST, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        sys.exit("could not open the video writer (missing codec?)")
    for _ in range(FPS * SECONDS):
        writer.write(canvas)
    writer.release()

    size_mb = os.path.getsize(DST) / 1e6
    print(f"wrote {DST}  ({FPS * SECONDS} frames, {SECONDS}s, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
