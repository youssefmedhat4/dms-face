#!/usr/bin/env python3
"""
Throughput and resource benchmark.

Produces the numbers your results chapter needs, measured on YOUR hardware
rather than quoted from someone else's blog post:

    inference latency (mean / median / p95 / p99)
    end-to-end frame latency
    sustained FPS
    CPU utilisation and RSS

    python bench.py                       # live camera, 30 s
    python bench.py --source image        # fixed image, isolates compute
    python bench.py --resolutions         # sweep input resolution
    python bench.py --threads 1 2 3 4     # sweep inference threads
"""

import argparse
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dms_face.calibration import DriverProfile
from dms_face.camera import open_source
from dms_face.config import Config
from dms_face.pipeline import FacePipeline

try:
    import psutil
except ImportError:
    psutil = None


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark the face pipeline")
    p.add_argument("--source", choices=["camera", "image", "video"], default="camera")
    p.add_argument("--image", default="tests/assets/portrait.jpg")
    p.add_argument("--video")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--resolutions", action="store_true", help="sweep input resolution")
    p.add_argument("--json", help="write results to this JSON file")
    return p.parse_args()


def describe_host():
    import platform

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        with open("/proc/device-tree/model", "rb") as f:
            info["board"] = f.read().decode(errors="ignore").strip("\x00")
    except OSError:
        pass
    try:
        import mediapipe

        info["mediapipe"] = mediapipe.__version__
    except Exception:
        pass
    info["opencv"] = cv2.__version__
    if psutil:
        info["cores"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    return info


def summarise(name, values):
    if not values:
        return f"{name:22s}  no samples"
    values = sorted(values)
    n = len(values)
    return (
        f"{name:22s}  mean {statistics.mean(values):6.2f}  "
        f"median {statistics.median(values):6.2f}  "
        f"p95 {values[int(n * 0.95) - 1]:6.2f}  "
        f"p99 {values[int(n * 0.99) - 1]:6.2f}  "
        f"max {values[-1]:6.2f}"
    )


def bench_once(frames_iter, width, height, duration, warmup, label=""):
    """frames_iter yields (frame_rgb, timestamp). Returns a metrics dict."""
    cfg = Config()
    cfg.camera.width, cfg.camera.height = width, height
    pipeline = FacePipeline(cfg, DriverProfile(), width=width, height=height)

    proc = psutil.Process() if psutil else None
    if proc:
        proc.cpu_percent(None)  # prime the counter

    inference_ms, total_ms = [], []
    faces = 0
    n = 0
    t_start = None

    try:
        for frame, ts in frames_iter:
            if frame is None:
                continue
            res = pipeline.process(frame, ts)
            n += 1
            if n <= warmup:  # discard: first frames include lazy init
                continue
            if t_start is None:
                t_start = time.monotonic()
            inference_ms.append(res.inference_ms)
            total_ms.append(res.total_ms)
            faces += int(res.face_present)
            if time.monotonic() - t_start >= duration:
                break
    finally:
        pipeline.close()

    wall = (time.monotonic() - t_start) if t_start else 0.0
    measured = len(inference_ms)

    result = {
        "label": label,
        "width": width,
        "height": height,
        "frames": measured,
        "wall_s": round(wall, 2),
        "fps": round(measured / wall, 2) if wall > 0 else 0.0,
        "detection_rate": round(faces / measured, 4) if measured else 0.0,
        "inference_ms_mean": round(statistics.mean(inference_ms), 2) if inference_ms else None,
        "inference_ms_p95": round(sorted(inference_ms)[int(measured * 0.95) - 1], 2)
        if measured > 1
        else None,
        "total_ms_mean": round(statistics.mean(total_ms), 2) if total_ms else None,
    }
    if proc:
        result["cpu_percent"] = round(proc.cpu_percent(None), 1)
        result["rss_mb"] = round(proc.memory_info().rss / 1e6, 1)

    result["_inference"] = inference_ms
    result["_total"] = total_ms
    return result


def image_frames(path, width, height):
    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"could not read {path}")
    bgr = cv2.resize(bgr, (width, height))
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    t = time.monotonic()
    i = 0
    while True:
        # Advance the timestamp on a synthetic 30 fps clock so the landmarker's
        # VIDEO-mode tracking behaves as it would on a live stream.
        yield rgb, t + i / 30.0
        i += 1


def camera_frames(cfg, video=None):
    source = open_source(cfg, video=video, realtime=False)
    print(f"[camera] {source.describe()}")
    with source:
        while True:
            frame, ts = source.read()
            yield frame, ts if ts is not None else time.monotonic()


def main():
    args = parse_args()

    print("=" * 74)
    print("Face pipeline benchmark")
    print("=" * 74)
    for k, v in describe_host().items():
        print(f"  {k:12s} {v}")
    if not psutil:
        print("  (install psutil for CPU/RSS figures: pip install psutil)")
    print()

    resolutions = [(320, 240), (480, 360), (640, 480), (960, 720)] if args.resolutions else [
        (args.width, args.height)
    ]

    results = []
    for w, h in resolutions:
        label = f"{w}x{h}"
        print(f"--- {label} " + "-" * (68 - len(label)))

        if args.source == "image":
            frames = image_frames(args.image, w, h)
            note = f"fixed image ({args.image})"
        else:
            cfg = Config()
            cfg.camera.width, cfg.camera.height = w, h
            cfg.camera.index = args.camera_index
            frames = camera_frames(cfg, video=args.video)
            note = "video file" if args.video else "live camera"

        print(f"  source: {note}   duration: {args.duration:.0f}s   warmup: {args.warmup} frames")
        r = bench_once(frames, w, h, args.duration, args.warmup, label)
        results.append(r)

        print(f"  {summarise('inference (ms)', r.pop('_inference'))}")
        print(f"  {summarise('total/frame (ms)', r.pop('_total'))}")
        print(f"  {'sustained FPS':22s}  {r['fps']:.2f}   "
              f"({r['frames']} frames in {r['wall_s']}s)")
        print(f"  {'face detected':22s}  {r['detection_rate']:.1%}")
        if "cpu_percent" in r:
            print(f"  {'CPU (this process)':22s}  {r['cpu_percent']:.0f}%   "
                  f"RSS {r['rss_mb']:.0f} MB")
        print()

    if args.json:
        import json

        with open(args.json, "w", encoding="utf8") as f:
            json.dump({"host": describe_host(), "results": results}, f, indent=2)
        print(f"[bench] written -> {args.json}")

    if len(results) > 1:
        print("=" * 74)
        print(f"{'resolution':>12} {'fps':>8} {'inference':>12} {'cpu%':>8}")
        for r in results:
            print(f"{r['label']:>12} {r['fps']:>8.1f} "
                  f"{r['inference_ms_mean'] or 0:>10.2f}ms "
                  f"{r.get('cpu_percent', 0):>7.0f}%")


if __name__ == "__main__":
    main()
