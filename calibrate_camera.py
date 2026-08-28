#!/usr/bin/env python3
"""
One-off camera intrinsic calibration.

Why this is not optional on your hardware: measured on a single fixed frame,
sweeping only the ASSUMED focal length,

    focal    yaw    pitch    roll
      400   -1.4     38.1    -1.4
      820   -1.6     21.1    -1.0
     2000   -1.7      9.4    -0.7
     3000   -1.7      6.5    -0.6

Pitch is almost entirely an artefact of the guess; yaw and roll barely move.
On a 160-degree lens the distortion makes it worse still. Twenty minutes here
removes the guess and gives you a methodology section worth marks.

Usage:
    1. Print a chessboard  (https://github.com/opencv/opencv/blob/4.x/doc/pattern.png)
       Stick it on something rigid and flat. Cardboard is fine.
    2. python calibrate_camera.py
    3. Press SPACE to capture ~20 views: hold the board at different angles,
       distances and positions, covering the CORNERS of the frame -- distortion
       is strongest there and the fit needs to see it.
    4. Press c to compute, and it writes camera_calibration.json
    5. python run.py --camera-calib camera_calibration.json
"""

import argparse
import json
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Chessboard camera calibration")
    p.add_argument("--cols", type=int, default=9, help="inner corners per row")
    p.add_argument("--rows", type=int, default=6, help="inner corners per column")
    p.add_argument("--square-mm", type=float, default=25.0, help="square size in mm")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--min-views", type=int, default=12)
    p.add_argument("--out", default="camera_calibration.json")
    return p.parse_args()


def main():
    args = parse_args()
    pattern = (args.cols, args.rows)

    # Object points in real millimetres, so tvec comes out in mm too.
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2)
    objp *= args.square_mm

    obj_points, img_points = [], []

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    shape = None

    print(f"\nLooking for a {args.cols}x{args.rows} inner-corner chessboard.")
    print("SPACE = capture   c = compute   q = quit")
    print("Vary angle, distance and position. Include the frame corners.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shape = gray.shape[::-1]

            found, corners = cv2.findChessboardCorners(
                gray, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
            )

            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, pattern, corners, found)

            colour = (110, 190, 110) if found else (70, 70, 235)
            cv2.rectangle(display, (0, 0), (display.shape[1], 30), (26, 20, 16), -1)
            cv2.putText(display, f"views: {len(img_points)}/{args.min_views}   "
                        f"{'BOARD FOUND' if found else 'no board'}",
                        (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
            cv2.imshow("calibrate", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return
            if key == ord(" ") and found:
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(objp.copy())
                img_points.append(refined)
                print(f"  captured view {len(img_points)}")
            if key == ord("c"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(img_points) < args.min_views:
        raise SystemExit(
            f"only {len(img_points)} views captured; need at least {args.min_views} "
            "for a stable fit."
        )

    print("\ncomputing...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, shape, None, None)

    # Per-view reprojection error tells you whether the fit is trustworthy.
    errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        errors.append(cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected))
    mean_error = float(np.mean(errors))

    print(f"  RMS reprojection error   {rms:.4f}")
    print(f"  mean per-view error      {mean_error:.4f} px")
    if mean_error > 1.0:
        print("  WARNING: > 1 px is a poor fit. Recapture with more varied board poses.")
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"  distortion {np.round(dist.ravel(), 4).tolist()}")

    naive_f = float(shape[0])
    print(f"\n  For comparison, the uncalibrated guess would have used f = {naive_f:.0f} "
          f"({100 * (K[0,0] - naive_f) / naive_f:+.0f}% off).")

    with open(args.out, "w", encoding="utf8") as f:
        json.dump(
            {
                "camera_matrix": K.tolist(),
                "dist_coeffs": dist.ravel().tolist(),
                "image_size": list(shape),
                "rms_reprojection_error": float(rms),
                "mean_per_view_error_px": mean_error,
                "n_views": len(img_points),
                "pattern": {"cols": args.cols, "rows": args.rows, "square_mm": args.square_mm},
            },
            f,
            indent=2,
        )
    print(f"\nwritten -> {args.out}")
    print(f"use it:  python run.py --camera-calib {args.out}")


if __name__ == "__main__":
    main()
