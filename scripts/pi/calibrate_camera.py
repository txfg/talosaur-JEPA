#!/usr/bin/env python
"""Underwater camera calibration (checkerboard) -> intrinsics for guidance (port: calibrated).

Calibrate *in water, through the final housing port*: the refraction of a flat port is then
absorbed into the intrinsics. Needs OpenCV (sudo apt install python3-opencv).

  1) capture:  python scripts/pi/calibrate_camera.py capture --out calib_imgs --n 30
     (move a printed checkerboard around the whole field of view, at 0.5-2 m)
  2) solve:    python scripts/pi/calibrate_camera.py solve --images calib_imgs --board 9x6 --square 0.025
     -> writes camera_calibration.yaml; set guidance.camera: {port: calibrated, fx:, fy:, cx:, cy:}
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path


def capture(a) -> None:
    from picamera2 import Picamera2

    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (2304, 1296)}))
    cam.start()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        input(f"[{i + 1}/{a.n}] position the board, press Enter")
        cam.capture_file(str(Path(a.out) / f"calib_{i:03d}.jpg"))
        time.sleep(0.2)
    cam.stop()


def solve(a) -> None:
    import cv2
    import numpy as np
    import yaml

    cols, rows = (int(v) for v in a.board.split("x"))
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * a.square
    objpts, imgpts, size = [], [], None
    for f in sorted(glob.glob(str(Path(a.images) / "*.jpg"))):
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        size = g.shape[::-1]
        ok, corners = cv2.findChessboardCorners(g, (cols, rows))
        if ok:
            corners = cv2.cornerSubPix(
                g, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
            )
            objpts.append(objp)
            imgpts.append(corners)
    if len(objpts) < 8:
        raise SystemExit(f"only {len(objpts)} usable images; need >= 8")
    rms, K, dist, _, _ = cv2.calibrateCamera(objpts, imgpts, size, None, None)
    W, H = size
    out = {
        "rms_px": float(rms),
        "image_size": [W, H],
        "fx": float(K[0, 0] / W),
        "fy": float(K[1, 1] / H),
        "cx": float(K[0, 2] / W),
        "cy": float(K[1, 2] / H),
        "dist": dist.ravel().tolist(),
        "note": "fx/fy/cx/cy normalised by image width/height; use with guidance.camera.port=calibrated",
    }
    Path(a.out_yaml).write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"RMS reprojection error {rms:.2f} px -> {a.out_yaml}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--out", default="calib_imgs")
    c.add_argument("--n", type=int, default=30)
    s = sub.add_parser("solve")
    s.add_argument("--images", default="calib_imgs")
    s.add_argument("--board", default="9x6", help="inner corners, cols x rows")
    s.add_argument("--square", type=float, default=0.025, help="square size (m)")
    s.add_argument("--out-yaml", default="camera_calibration.yaml")
    a = ap.parse_args(argv)
    capture(a) if a.cmd == "capture" else solve(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
