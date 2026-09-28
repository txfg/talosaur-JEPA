"""Image position -> bearing / elevation for the Camera Module 3 Wide, in air or underwater.

* ``port="air"``: pinhole model from the in-air field of view (102 x 67 deg).
* ``port="flat"``: behind a flat port, rays refract at the window: sin(theta_w) = sin(theta_a) / n
  with n ~ 1.333, so the underwater field of view shrinks (~ 71 x 49 deg).
* ``port="dome"``: a correctly centred dome preserves the in-air field of view.
* ``port="calibrated"``: intrinsics (and OpenCV distortion ``dist`` = k1, k2, p1, p2[, k3])
  measured *underwater* with scripts/pi/calibrate_camera.py; they already include the port's
  effect and the wide lens's barrel distortion, so this is the accurate option once the housing
  is final. The nominal models above ignore lens distortion, which grows toward the image edges.

Angles: positive yaw = target to the right, positive pitch = target above the image centre.
Image coordinates are normalised: u = x / width, v = y / height, (0, 0) = top-left corner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CameraModel:
    hfov_deg: float = 102.0
    vfov_deg: float = 67.0
    port: str = "flat"  # air | flat | dome | calibrated
    n_water: float = 1.333
    fx: float | None = None  # calibrated focal lengths in units of image width / height
    fy: float | None = None
    cx: float = 0.5
    cy: float = 0.5
    dist: tuple[float, ...] = ()  # OpenCV k1, k2, p1, p2[, k3] (calibrated only)

    def __post_init__(self) -> None:
        if self.port not in ("air", "flat", "dome", "calibrated"):
            raise ValueError(f"unknown port {self.port!r}")
        if self.port == "calibrated" and not (self.fx and self.fy):
            raise ValueError("port=calibrated needs fx and fy (scripts/pi/calibrate_camera.py)")

    def _calibrated(self) -> bool:
        return self.port == "calibrated"

    def _distort(self, x: float, y: float) -> tuple[float, float]:
        k1, k2, p1, p2, k3 = (tuple(self.dist) + (0.0,) * 5)[:5]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        return (
            x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
        )

    def _undistort(self, xd: float, yd: float) -> tuple[float, float]:
        if not any(self.dist):
            return xd, yd
        x, y = xd, yd
        for _ in range(20):  # fixed-point iteration, as in OpenCV's undistortPoints
            ex, ey = self._distort(x, y)
            x, y = x + (xd - ex), y + (yd - ey)
        return x, y

    def _ray_air(self, u: float, v: float) -> tuple[float, float]:
        """Normalised image coords -> tangent-plane ray (x, y, 1) in air / behind the port."""
        if self._calibrated():
            return self._undistort((u - self.cx) / self.fx, (v - self.cy) / self.fy)
        tx = math.tan(math.radians(self.hfov_deg) / 2)
        ty = math.tan(math.radians(self.vfov_deg) / 2)
        return (u - 0.5) * 2 * tx, (v - 0.5) * 2 * ty

    def _pixel(self, x: float, y: float) -> tuple[float, float]:
        if self._calibrated():
            xd, yd = self._distort(x, y)
            return xd * self.fx + self.cx, yd * self.fy + self.cy
        tx = math.tan(math.radians(self.hfov_deg) / 2)
        ty = math.tan(math.radians(self.vfov_deg) / 2)
        return x / (2 * tx) + 0.5, y / (2 * ty) + 0.5

    def angles(self, u: float, v: float) -> tuple[float, float]:
        """Normalised image coords (0..1) -> (yaw_deg, pitch_deg) of the ray in the water."""
        x, y = self._ray_air(u, v)
        if self.port == "flat":
            r = math.hypot(x, y)
            if r > 1e-12:
                theta_a = math.atan(r)
                theta_w = math.asin(math.sin(theta_a) / self.n_water)
                s = math.tan(theta_w) / r
                x, y = x * s, y * s
        yaw = math.degrees(math.atan(x))
        pitch = math.degrees(math.atan2(-y, math.sqrt(1.0 + x * x)))
        return yaw, pitch

    def project(self, yaw_deg: float, pitch_deg: float) -> tuple[float, float]:
        """Inverse of :meth:`angles`: (yaw, pitch) in the water -> normalised image coords."""
        x = math.tan(math.radians(yaw_deg))
        y = -math.tan(math.radians(pitch_deg)) * math.sqrt(1.0 + x * x)
        if self.port == "flat":
            r = math.hypot(x, y)
            if r > 1e-12:
                theta_w = math.atan(r)
                theta_a = math.asin(min(1.0, math.sin(theta_w) * self.n_water))
                s = math.tan(theta_a) / r
                x, y = x * s, y * s
        return self._pixel(x, y)

    def effective_fov(self) -> tuple[float, float]:
        """Full horizontal / vertical field of view in the water (degrees)."""
        return (
            self.angles(1.0, self.cy)[0] - self.angles(0.0, self.cy)[0],
            self.angles(self.cx, 0.0)[1] - self.angles(self.cx, 1.0)[1],
        )
