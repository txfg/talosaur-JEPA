"""Frame sources for the onboard loop.

* :class:`Picamera2Source` - Camera Module 3 Wide on the Pi. The ISP produces two streams: a
  small "lores" stream at the model's input size (no CPU resize) and a "main" stream for video
  recording. The lores stream is YUV420 (works on every Pi); conversion to RGB at ~208x112 costs
  well under a millisecond.
* :class:`VideoFileSource` - recorded footage (PyAV or OpenCV), for replay and desktop tests.
* :class:`SyntheticSource` - procedural frames, no dependencies (tests, benchmarks without camera).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    t: float  # seconds (monotonic for live cameras, stream time for files)
    rgb: np.ndarray  # (H, W, 3) uint8 at (or near) the model input size
    index: int = 0
    full: np.ndarray | None = None  # full-resolution RGB when available (replay annotation)


def yuv420_to_rgb(yuv: np.ndarray, width: int, height: int) -> np.ndarray:
    """Planar I420 (as returned by picamera2 for YUV420 streams) -> RGB uint8 (BT.601 full range).

    ``yuv`` has shape (height * 3 // 2, stride) with stride >= width.
    """
    stride = yuv.shape[1]
    y = yuv[:height, :width].astype(np.float32)
    uv = yuv[height:].reshape(-1)
    half = (height // 2) * (stride // 2)
    u = uv[:half].reshape(height // 2, stride // 2)[:, : width // 2].astype(np.float32) - 128.0
    v = uv[half : 2 * half].reshape(height // 2, stride // 2)[:, : width // 2].astype(np.float32) - 128.0
    u = u.repeat(2, axis=0).repeat(2, axis=1)[:height, :width]
    v = v.repeat(2, axis=0).repeat(2, axis=1)[:height, :width]
    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


class Picamera2Source:
    """Camera Module 3 (Wide) via picamera2.

    ``hdr``: ``"off"``; ``"sensor"`` = the IMX708's on-sensor HDR (set through picamera2's
    IMX708 helper before the camera opens; limits the sensor to 2304x1296); ``"isp"`` / ``"night"``
    = the Pi 5 ISP's single-exposure HDR / night modes (libcamera ``HdrMode``). Whether any of them
    helps a lit animal against dark water is an open question for pool tests - default off.
    """

    def __init__(
        self,
        lores_size: tuple[int, int],
        main_size: tuple[int, int] = (1280, 720),
        fps: float = 15.0,
        controls: dict | None = None,
        hdr: str | bool = "off",
    ):
        from picamera2 import Picamera2  # apt: python3-picamera2

        hdr = {True: "sensor", False: "off", None: "off"}.get(hdr, hdr)
        if hdr not in ("off", "sensor", "isp", "night"):
            raise ValueError(f"hdr must be off | sensor | isp | night, not {hdr!r}")
        self.sensor_hdr = hdr == "sensor"
        if self.sensor_hdr:
            _set_imx708_hdr(True)
        self.cam = Picamera2()
        w, h = lores_size
        self.lw, self.lh = w - w % 2, h - h % 2
        # The lores stream shares the main stream's crop (full field of view) and is scaled by the
        # ISP; a slightly different aspect ratio (e.g. 208x112 vs 16:9) stretches it by a few %,
        # which the normalised camera model does not care about.
        cfg = self.cam.create_video_configuration(
            main={"size": tuple(main_size), "format": "YUV420"},
            lores={"size": (self.lw, self.lh), "format": "YUV420"},
            controls={"FrameRate": float(fps), **(controls or {})},
            buffer_count=4,
        )
        self.cam.configure(cfg)
        # libcamera may adjust the requested sizes; use what was actually configured
        actual = self.cam.camera_configuration().get("lores") or {}
        self.lw, self.lh = tuple(actual.get("size", (self.lw, self.lh)))
        if hdr in ("isp", "night"):
            from libcamera import controls as lc

            mode = lc.HdrModeEnum.SingleExposure if hdr == "isp" else lc.HdrModeEnum.Night
            self.cam.set_controls({"HdrMode": mode})
        self.cam.start()
        self.i = 0
        self.t0 = time.monotonic()

    def read(self) -> Frame | None:
        yuv = self.cam.capture_array("lores")
        self.i += 1
        return Frame(time.monotonic() - self.t0, yuv420_to_rgb(yuv, self.lw, self.lh), self.i)

    def close(self) -> None:
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass
        if self.sensor_hdr:  # the sensor keeps the setting after we exit; restore it
            try:
                _set_imx708_hdr(False)
            except Exception:
                pass


def _set_imx708_hdr(enable: bool) -> None:
    from picamera2.devices.imx708 import IMX708

    with IMX708() as sensor:
        sensor.set_sensor_hdr_mode(enable)


class VideoFileSource:
    def __init__(
        self,
        path: str | Path,
        size: tuple[int, int] | None = None,
        keep_full: bool = False,
        max_fps: float | None = None,
    ):
        """``size`` = (W, H) for the model input; ``max_fps`` drops frames to emulate a slower loop."""
        self.path, self.size, self.keep_full, self.max_fps = str(path), size, keep_full, max_fps
        self.i = 0
        self._last_t = -1e9
        try:
            import av

            self._c = av.open(self.path)
            self._stream = self._c.streams.video[0]
            self._stream.thread_type = "AUTO"
            self._it = self._c.decode(self._stream)
            self.kind = "av"
        except ImportError:
            import cv2

            self._cap = cv2.VideoCapture(self.path)
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.kind = "cv2"

    def _next_raw(self):
        if self.kind == "av":
            fr = next(self._it, None)
            if fr is None:
                return None, None
            t = float(fr.pts * self._stream.time_base) if fr.pts is not None else self.i / 30.0
            return t, fr.to_ndarray(format="rgb24")
        import cv2

        ok, bgr = self._cap.read()
        if not ok:
            return None, None
        return self.i / self._fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def read(self) -> Frame | None:
        while True:
            t, rgb = self._next_raw()
            if rgb is None:
                return None
            self.i += 1
            if self.max_fps and t - self._last_t < 1.0 / self.max_fps - 1e-6:
                continue
            self._last_t = t
            small = rgb
            if self.size is not None:
                from talosaur.onboard.runtime import resize_u8

                small = resize_u8(rgb, self.size[1], self.size[0])
            return Frame(t, small, self.i, rgb if self.keep_full else None)

    def close(self) -> None:
        if self.kind == "av":
            self._c.close()
        else:
            self._cap.release()


class SyntheticSource:
    """A drifting 'fish' over a gradient with noise; deterministic and dependency-free.

    The fish is visible for ``visible`` frames, then gone for ``hidden`` frames, and so on; each
    appearance takes the next colour from ``colors`` (so several colours = several animals)."""

    def __init__(
        self,
        size: tuple[int, int],
        n_frames: int = 300,
        fps: float = 10.0,
        seed: int = 0,
        colors=((200, 180, 90),),
        visible: int = 40,
        hidden: int = 40,
    ):
        self.w, self.h = size
        self.n, self.fps = n_frames, fps
        self.colors = [tuple(c) for c in colors]
        self.visible, self.hidden = visible, hidden
        self.rng = np.random.default_rng(seed)
        self.i = 0
        yy = np.linspace(0.9, 0.4, self.h, dtype=np.float32)[:, None, None]
        self.bg = (np.array([10, 60, 90], dtype=np.float32)[None, None, :] * yy).repeat(self.w, axis=1)

    def read(self) -> Frame | None:
        if self.i >= self.n:
            return None
        k = self.i
        self.i += 1
        img = self.bg.copy()
        cycle, phase = divmod(k, self.visible + self.hidden)
        if phase < self.visible:
            cx = self.w * (0.2 + 0.6 * (phase / self.visible))
            cy = self.h * (0.5 + 0.1 * np.sin(k / 7.0))
            yy, xx = np.mgrid[0 : self.h, 0 : self.w]
            m = ((xx - cx) / (self.w * 0.08)) ** 2 + ((yy - cy) / (self.h * 0.07)) ** 2 <= 1
            img[m] = self.colors[cycle % len(self.colors)]
        img += self.rng.normal(0, 3, img.shape).astype(np.float32)
        return Frame(k / self.fps, np.clip(img, 0, 255).astype(np.uint8), k)

    def close(self) -> None:
        pass
