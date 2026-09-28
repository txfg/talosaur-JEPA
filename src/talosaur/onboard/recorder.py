"""Recording control driven by the guidance state machine.

On the Pi 5 there is **no hardware H.264 encoder**: picamera2 encodes in software (libav/x264),
which the official docs put at roughly 30-40% CPU for 1080p30. Options, cheapest first:

* ``preroll_s: 0`` - the encoder only runs while recording (no approach footage before TRACK);
* ``preroll_s > 0`` - encode continuously into a circular buffer so the approach is kept (always
  paying the encoder CPU);
* record at 1280x720 instead of 1920x1080 (roughly half the encode cost);
* a separate action camera recording continuously (no Pi CPU at all).
The Pi benchmark measures model fps with and without recording, to choose between these.

Files are raw H.264 elementary streams (``.h264``); wrap them without re-encoding with
``ffmpeg -framerate 15 -i talosaur_X.h264 -c copy talosaur_X.mp4``.
"""

from __future__ import annotations

import time
from pathlib import Path

from talosaur.utils.log import get_logger

log = get_logger("recorder")


class NullRecorder:
    recording = False

    def start(self, t: float) -> None:
        self.recording = True

    def stop(self, t: float) -> None:
        self.recording = False

    def close(self) -> None:
        pass


class Picamera2Recorder:
    def __init__(
        self, cam, out_dir: str | Path, bitrate: int = 6_000_000, preroll_s: float = 5.0, fps: float = 15.0
    ):
        from picamera2.encoders import H264Encoder  # software (libav) on the Pi 5
        from picamera2.outputs import CircularOutput

        self.cam = cam
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # keyframe every second: the pre-roll buffer is written from its oldest keyframe, and the
        # encoder's rate must match the camera's for the bitrate budget to hold (picamera2 >= 0.3.x
        # accepts iperiod/framerate on both the libav and the V4L2 encoder)
        self.encoder = H264Encoder(bitrate=bitrate, iperiod=max(1, int(round(fps))), framerate=fps)
        self.preroll_s = preroll_s
        self.recording = False
        self.output = CircularOutput(buffersize=max(1, int(preroll_s * fps))) if preroll_s > 0 else None
        if self.output is not None:
            self.cam.start_encoder(self.encoder, self.output)  # continuous encode into the ring buffer

    def start(self, t: float) -> None:
        if self.recording:
            return
        path = self.out_dir / f"talosaur_{time.strftime('%Y%m%d_%H%M%S')}.h264"
        if self.output is not None:
            self.output.fileoutput = str(path)
            self.output.start()
        else:
            from picamera2.outputs import FileOutput

            self.cam.start_encoder(self.encoder, FileOutput(str(path)))
        self.recording = True
        log.info(f"recording -> {path}")

    def stop(self, t: float) -> None:
        if not self.recording:
            return
        if self.output is not None:
            self.output.stop()
        else:
            self.cam.stop_encoder()
        self.recording = False
        log.info("recording stopped")

    def close(self) -> None:
        self.stop(0.0)
        if self.output is not None:
            try:
                self.cam.stop_encoder()
            except Exception:
                pass
