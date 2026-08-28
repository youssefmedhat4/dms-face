#!/usr/bin/env python3
"""
Real-Time DMS -- face subsystem.

    python run.py                     # webcam, live
    python run.py --calibrate         # run driver calibration first
    python run.py --video clip.mp4    # repeatable evaluation on a recording
    python run.py --no-display        # headless, for throughput measurement

Keys while running:
    q / Esc  quit          c  recalibrate       r  reset counters
    m  toggle full mesh    i  toggle EAR landmark indices
    s  save snapshot       space  pause
"""

import argparse
import csv
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dms_face import metrics as M
from dms_face.alerts import Alerter
from dms_face.calibration import CalibrationCollector, DriverProfile, build_profile
from dms_face.camera import open_source
from dms_face.config import Config
from dms_face.hud import Hud
from dms_face.pipeline import FacePipeline


def parse_args():
    p = argparse.ArgumentParser(description="Face-based driver monitoring subsystem")
    p.add_argument("--profile", default="default", help="driver profile name")
    p.add_argument("--calibrate", action="store_true", help="force recalibration")
    p.add_argument("--no-calibrate", action="store_true",
                   help="skip the automatic first-run calibration (expect a gated eye channel)")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--video", help="run on a video file instead of a camera")
    p.add_argument("--fast", action="store_true", help="with --video, decode as fast as possible")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--ir", action="store_true", help="IR mode: apply CLAHE to luminance")
    p.add_argument("--no-flip", action="store_true", help="disable the mirror view")
    p.add_argument("--no-display", action="store_true", help="headless")
    p.add_argument("--no-alerts", action="store_true")
    p.add_argument("--alert-backend", choices=["auto", "gpio", "console"], default="auto")
    p.add_argument("--camera-calib", help="camera_calibration.json from calibrate_camera.py")
    p.add_argument("--config", help="config JSON to load")
    p.add_argument("--log", help="write a per-frame CSV here (for your results chapter)")
    p.add_argument("--duration", type=float, help="stop after N seconds")
    return p.parse_args()


def build_config(args):
    cfg = Config.from_json(args.config) if args.config else Config()
    cfg.camera.index = args.camera_index
    cfg.camera.width = args.width
    cfg.camera.height = args.height
    cfg.camera.fps = args.fps
    cfg.camera.ir_mode = args.ir
    cfg.camera.flip_horizontal = not args.no_flip
    return cfg


# --------------------------------------------------------------------------
def run_calibration(source, pipeline, cfg, profile_name, display=True):
    """Two-phase routine. Returns a DriverProfile or None if it failed.

    Phase 1 captures the open-eye baseline, the on-road gaze centre and the
    driver's neutral head pose in one go -- the pose baseline matters as much
    as the EAR one, because it removes seat position and camera mounting angle
    from every later decision.
    """
    ccfg = cfg.calibration
    print("\n=== Driver calibration ===")

    phases = [
        ("open", ccfg.open_phase_s, "Look straight at the road. Blink normally."),
        ("closed", ccfg.closed_phase_s, "Now CLOSE your eyes and keep them closed."),
    ]
    results = {}

    for key, duration, prompt in phases:
        print(f"\n  {prompt}")

        # Pump the window through the countdown. A bare time.sleep() here stops
        # the GUI event loop, so Windows greys the window out and labels it
        # "Not Responding" -- indistinguishable from a crash to anyone watching.
        countdown_end = time.monotonic() + 3.0
        shown = None
        while True:
            remaining = countdown_end - time.monotonic()
            if remaining <= 0:
                break
            seconds = int(remaining) + 1
            if seconds != shown:  # only reprint when the digit changes
                print(f"    starting in {seconds}...", end="\r", flush=True)
                shown = seconds
            if display:
                frame, _ = source.read()
                if frame is not None:
                    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(img, (0, 0), (img.shape[1], 62), (26, 20, 16), -1)
                    cv2.putText(img, prompt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (235, 235, 230), 1, cv2.LINE_AA)
                    cv2.putText(img, f"starting in {int(remaining) + 1}...", (12, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 170, 60), 2, cv2.LINE_AA)
                    cv2.imshow("DMS", img)
                if cv2.waitKey(30) & 0xFF in (27, ord("q")):
                    return None
            else:
                time.sleep(0.05)
        print("    collecting...        ")

        collector = CalibrationCollector(ccfg.settle_s).start()
        last_ts = None
        while collector.elapsed < duration + ccfg.settle_s:
            frame, ts = source.read()
            if frame is None:
                continue
            # Same de-duplication as the main loop: the capture thread re-serves
            # the newest frame on every read, so without this the collector
            # would record the same EAR hundreds of times and report a sample
            # count far higher than the number of frames actually seen.
            if ts == last_ts:
                time.sleep(0.001)
                continue
            last_ts = ts
            res = pipeline.process(frame, ts)
            if res.face_present and res.ear_raw is not None:
                collector.push(
                    ear=res.ear_raw,
                    h=res.gaze_h,
                    v=res.gaze_v,
                    yaw=res.angles[0] if res.angles else None,
                    pitch=res.angles[1] if res.angles else None,
                    roll=res.angles[2] if res.angles else None,
                )
            if display:
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                remain = max(0.0, duration + ccfg.settle_s - collector.elapsed)
                banner = "SETTLE" if collector.settling else f"{prompt}  {remain:0.1f}s"
                cv2.rectangle(img, (0, 0), (img.shape[1], 34), (26, 20, 16), -1)
                cv2.putText(img, banner, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (235, 235, 230), 1, cv2.LINE_AA)
                cv2.putText(img, f"n={len(collector.ear)}", (img.shape[1] - 90, 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 170, 60), 1, cv2.LINE_AA)
                cv2.imshow("DMS", img)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    return None

        out = collector.result()
        print(f"    captured {out['n']} samples, median EAR = "
              f"{out['ear']:.4f}" if out["ear"] else "    no face detected!")
        if out["n"] < ccfg.min_samples:
            print(f"    WARNING: only {out['n']} samples (wanted {ccfg.min_samples})")
        results[key] = out

    try:
        profile = build_profile(profile_name, results["open"], results["closed"])
    except ValueError as exc:
        print(f"\n  !! {exc}")
        return None

    path = profile.save()
    print()
    print(f"  {profile.describe()}")
    advice = profile.quality_advice()
    if advice:
        print(f"  {advice}")
    print(f"  saved -> {path}")
    print()
    return profile


# --------------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = build_config(args)

    profile, loaded = DriverProfile.load_or_default(args.profile)
    if loaded:
        print(f"[profile] loaded: {profile.describe()}")
        advice = profile.quality_advice()
        if advice:
            print(f"[profile] {advice}")
    else:
        print(f"[profile] no saved profile '{args.profile}'.")
        if args.no_calibrate:
            print("          Running uncalibrated. Expect the eye channel to be gated off:")
            print("          generic thresholds do not transfer between people, and the")
            print("          uncalibrated head-pose bias usually exceeds the EAR validity gate.")
        else:
            print("          Calibrating automatically -- this is required for the eye")
            print("          channel to work at all. Use --no-calibrate to skip.")

    source = open_source(cfg, video=args.video, realtime=not args.fast)
    print(f"[camera] {source.describe()}")

    with source:
        w, h = source.size
        if (w, h) != (cfg.camera.width, cfg.camera.height):
            print(f"[camera] actual resolution {w}x{h} (requested "
                  f"{cfg.camera.width}x{cfg.camera.height})")

        pipeline = FacePipeline(cfg, profile, width=w, height=h, camera_calib=args.camera_calib)
        if not pipeline.headpose.calibrated:
            print("[headpose] no camera calibration supplied -- absolute pitch is unreliable.")
            print("           Gating uses pitch relative to your calibrated neutral, which is fine.")
            print("           Run `python calibrate_camera.py` to remove the approximation.")

        display = not args.no_display
        hud = Hud(w, h) if display else None
        alerter = Alerter(prefer=args.alert_backend, enabled=not args.no_alerts)
        print(f"[alerts] backend: {alerter.backend.name}")

        # Calibrate when explicitly asked, or on a first run with no stored
        # profile. Skipping it leaves generic thresholds AND a zero head-pose
        # baseline, which together gate the eye channel off entirely -- so the
        # uncalibrated first run would otherwise look broken rather than
        # uncalibrated.
        if args.calibrate or (not loaded and not args.no_calibrate):
            new_profile = run_calibration(source, pipeline, cfg, args.profile, display)
            if new_profile is not None:
                profile = new_profile
                pipeline = FacePipeline(cfg, profile, width=w, height=h,
                                        camera_calib=args.camera_calib)
            else:
                print("[profile] calibration did not complete -- continuing uncalibrated.")

        log_writer = None
        log_file = None
        if args.log:
            log_file = open(args.log, "w", newline="", encoding="utf8")
            log_writer = csv.writer(log_file)
            log_writer.writerow([
                "t", "face", "ear_raw", "ear_filt", "openness", "eye_state",
                "eye_valid", "yaw", "pitch", "roll", "gaze_dev",
                "perclos60", "perclos20", "validity", "state", "fps", "inference_ms",
            ])

        print()
        print("=" * 64)
        print("  RUNNING. The window stays open until you press q (or Esc).")
        print("  q=quit  c=recalibrate  m=mesh  i=indices  r=reset  space=pause")
        print("=" * 64)
        print()
        t_start = time.monotonic()
        paused = False
        last_state = None
        last_ts = None
        consecutive_errors = 0

        try:
            while True:
                if not paused:
                    frame, ts = source.read()
                    if frame is None:
                        if args.video:
                            print("[run] end of video")
                            break
                        continue

                    # The capture thread re-serves the newest frame on every
                    # read, so without this the main loop would burn CPU
                    # re-running inference on a frame it has already seen --
                    # and report an FPS several times the camera's actual rate.
                    if ts == last_ts:
                        time.sleep(0.001)
                        continue
                    last_ts = ts

                    # One malformed frame must not end the session. Without
                    # this, any exception anywhere in the pipeline unwinds
                    # straight past the loop to the `finally` block, which
                    # prints the summary and exits -- looking exactly like the
                    # program "just closed" for no reason.
                    try:
                        result = pipeline.process(frame, ts)
                        consecutive_errors = 0
                    except Exception as exc:
                        consecutive_errors += 1
                        print(f"  [frame error {consecutive_errors}] {type(exc).__name__}: {exc}")
                        if consecutive_errors >= 30:
                            print("  [run] 30 consecutive frame errors - stopping")
                            raise
                        continue
                    rep = result.report

                    if rep is not None and rep.should_alert:
                        alerter.alert(rep.state.severity)

                    if rep is not None and rep.state is not last_state:
                        elapsed = ts - t_start
                        print(f"  [{elapsed:7.1f}s] {rep.summary()}")
                        last_state = rep.state

                    if log_writer is not None:
                        log_writer.writerow([
                            f"{ts:.4f}", int(result.face_present),
                            _f(result.ear_raw), _f(result.ear_filtered), _f(result.openness),
                            result.eye_state.value, int(result.eye_channel_valid),
                            _f(result.angles[0] if result.angles else None),
                            _f(result.angles[1] if result.angles else None),
                            _f(result.angles[2] if result.angles else None),
                            _f(result.gaze_dev),
                            _f(rep.perclos if rep else None),
                            _f(rep.perclos_fast if rep else None),
                            _f(rep.validity if rep else None),
                            rep.state.value if rep else "",
                            f"{result.fps:.2f}", f"{result.inference_ms:.2f}",
                        ])

                    if display:
                        try:
                            cv2.imshow("DMS", hud.draw(frame, result, profile, cfg))
                        except Exception as exc:
                            print(f"  [hud error] {type(exc).__name__}: {exc}")

                if args.duration and (time.monotonic() - t_start) > args.duration:
                    break

                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    elif key == ord("m"):
                        hud.toggle_mesh()
                    elif key == ord("i"):
                        hud.toggle_indices()
                    elif key == ord(" "):
                        paused = not paused
                    elif key == ord("r"):
                        pipeline.perclos.reset()
                        pipeline.blink.reset()
                        print("  [reset] PERCLOS and blink counters cleared")
                    elif key == ord("s"):
                        name = f"snapshot_{int(time.time())}.png"
                        cv2.imwrite(name, hud.draw(frame, result, profile, cfg))
                        print(f"  [saved] {name}")
                    elif key == ord("c"):
                        new_profile = run_calibration(source, pipeline, cfg, args.profile, display)
                        if new_profile is not None:
                            profile = new_profile
                            pipeline = FacePipeline(cfg, profile, width=w, height=h,
                                                    camera_calib=args.camera_calib)
                elif not paused:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n[run] interrupted")
        finally:
            elapsed = time.monotonic() - t_start
            st = pipeline.frames_processed
            print(f"\n=== Session summary ===")
            print(f"  duration        {elapsed:.1f} s")
            print(f"  frames          {st}")
            if elapsed > 0:
                print(f"  mean FPS        {st / elapsed:.1f}")
            if st:
                print(f"  face detected   {pipeline.frames_with_face / st:.1%} of frames")
            print(f"  blinks          {len(pipeline.blink.events)}")
            print(f"  alerts raised   {alerter.count}")
            if log_file:
                log_file.close()
                print(f"  log written     {args.log}")

            pipeline.close()
            alerter.close()
            if display:
                cv2.destroyAllWindows()


def _f(v, nd=5):
    return "" if v is None else f"{v:.{nd}f}"


if __name__ == "__main__":
    main()
