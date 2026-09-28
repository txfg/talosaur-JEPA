"""Video recording on the Pi.

Two modes (``recording.mode`` in configs/onboard/pi5.yaml):

* ``continuous`` (default) - :class:`ContinuousRecorder` records the main stream for the whole run
  and never switches off. Files are cut into fixed-length segments (``segment_s``, split at a
  keyframe, no frames lost). The default container is MPEG-TS (``.ts``): a segment cut short by a
  power loss or crash still plays up to that point. The segment index and the encounter log say
  which file and offset shows each animal.
* ``events`` - :class:`Picamera2Recorder` records only around encounters, with a pre-roll ring
  buffer (``preroll_s``). Saves storage and, with ``preroll_s: 0``, encoder CPU.

On the Pi 5 there is **no hardware H.264 encoder**: picamera2's H264Encoder is libav (x264) in
software, so encoding competes with inference for the four cores. Raspberry Pi rate 1080p30 at
about 30-40% CPU; ``encoder_threads`` caps the encoder's threads (x264 otherwise picks 6). Continuous
mode costs the same CPU as events mode with a pre-roll (the encoder runs all the time either way);
the Pi benchmark measures model fps with and without an encoder running.

Never losing footage (continuous mode):
- a write error (e.g. a full disk) closes picamera2's PyavOutput for good and it then drops every
  later frame, silently; the recorder registers an ``error_callback`` and switches to a fresh
  segment, retrying every few seconds;
- every closed segment is fsync'ed: Linux can keep tens of seconds of written data in RAM, which a
  power cut would lose;
- run the app under systemd with ``Restart=always`` (docs/PI5.md section 4), so a crash costs
  seconds, not the rest of the dive.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

from talosaur.utils.log import get_logger

log = get_logger("recorder")

CONTAINER_EXT = {"mpegts": ".ts", "matroska": ".mkv", "mp4": ".mp4"}


def _overlaps(t0: float, t1: float | None, a: float, b: float) -> bool:
    return t0 <= b and (t1 is None or t1 >= a)


def _fsync(path: Path) -> None:
    """Push a closed file to storage (a power cut would otherwise lose what the kernel still holds)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as e:
        log.warning(f"fsync {path} failed: {e}")


def _encoder(bitrate: int, fps: float, threads: int | None):
    from picamera2.encoders import H264Encoder  # software (libav) on the Pi 5

    # keyframe every second: segment switches (and the events-mode pre-roll) start on a keyframe,
    # and the encoder's rate must match the camera's for the bitrate budget to hold
    enc = H264Encoder(bitrate=bitrate, iperiod=max(1, int(round(fps))), framerate=fps)
    if threads:
        enc.threads = int(threads)  # leave the other cores to the model
    return enc


class NullRecorder:
    """No camera (video file / synthetic sources): keeps the interface, records nothing."""

    def __init__(self, continuous: bool = False):
        self.continuous = continuous
        self.recording = continuous
        self.path = None  # the file being (or last) written
        self.protect_since: float | None = None

    def start(self, t: float) -> None:
        self.recording = True

    def stop(self, t: float) -> None:
        self.recording = self.continuous

    def tick(self, t: float) -> None:
        pass

    def files_between(self, a: float, b: float) -> list[dict]:
        return []

    def close(self, t: float | None = None) -> None:
        pass


class Picamera2Recorder:
    """``events`` mode: files only around encounters (start/stop from the state machine)."""

    continuous = False

    def __init__(
        self,
        cam,
        out_dir: str | Path,
        bitrate: int = 6_000_000,
        preroll_s: float = 5.0,
        fps: float = 15.0,
        encoder_threads: int | None = 2,
    ):
        from picamera2.outputs import CircularOutput

        self.cam = cam
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = _encoder(bitrate, fps, encoder_threads)  # the pre-roll starts on a keyframe
        self.preroll_s = preroll_s
        self.recording = False
        self.path: Path | None = None
        self.protect_since: float | None = None
        self.files: list[dict] = []  # {"path", "t0" (file start, pre-roll included), "t1"}
        self.seq = 0
        self.output = CircularOutput(buffersize=max(1, int(preroll_s * fps))) if preroll_s > 0 else None
        if self.output is not None:
            self.cam.start_encoder(self.encoder, self.output)  # continuous encode into the ring buffer

    def start(self, t: float) -> None:
        if self.recording:
            return
        # the sequence number keeps a roll-over (stop + start in the same second) from overwriting
        self.seq += 1
        path = self.out_dir / f"talosaur_{time.strftime('%Y%m%d_%H%M%S')}_{self.seq:03d}.h264"
        self.path = path
        if self.output is not None:
            self.output.fileoutput = str(path)
            self.output.start()
        else:
            from picamera2.outputs import FileOutput

            self.cam.start_encoder(self.encoder, FileOutput(str(path)))
        self.files.append({"path": path, "t0": t - self.preroll_s, "t1": None})
        self.recording = True
        log.info(f"recording -> {path}")

    def stop(self, t: float) -> None:
        if not self.recording:
            return
        if self.output is not None:
            self.output.stop()
        else:
            self.cam.stop_encoder()
        self.files[-1]["t1"] = t
        self.recording = False
        log.info("recording stopped")

    def tick(self, t: float) -> None:
        pass

    def files_between(self, a: float, b: float) -> list[dict]:
        return [
            {"file": f["path"].name, "offset_s": round(max(0.0, a - f["t0"]), 2)}
            for f in self.files
            if _overlaps(f["t0"], f["t1"], a, b)
        ]

    def close(self, t: float | None = None) -> None:
        self.stop(t if t is not None else 0.0)
        if self.output is not None:
            try:
                self.cam.stop_encoder()
            except Exception:
                pass


class ContinuousRecorder:
    """``continuous`` mode: the whole run, in segments; never switched off by guidance.

    Disk guard, checked every ``disk_check_s``: when free space drops below ``min_free_mb``,
    ``low_disk: delete_empty`` deletes the oldest finished segments that contain **no** animal
    (segments overlapping an encounter are kept, as is anything since the current encounter
    started); ``low_disk: stop`` stops instead. Either way, recording stops below
    ``min_free_mb / 4`` so the Pi's own filesystem never fills up.
    """

    continuous = True

    def __init__(
        self,
        cam,
        out_dir: str | Path,
        bitrate: int = 6_000_000,
        fps: float = 15.0,
        segment_s: float = 300.0,
        fmt: str = "mpegts",
        min_free_mb: float = 2000.0,
        low_disk: str = "delete_empty",
        disk_check_s: float = 10.0,
        t0: float = 0.0,
        encoder_threads: int | None = 2,
        retry_s: float = 5.0,
    ):
        from picamera2.outputs import PyavOutput, SplittableOutput

        if low_disk not in ("delete_empty", "stop"):
            raise ValueError(f"low_disk must be delete_empty or stop, not {low_disk!r}")
        self._pyav = PyavOutput
        self.cam = cam
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fmt, self.ext = fmt, CONTAINER_EXT.get(fmt, f".{fmt}")
        self.segment_s, self.min_free_mb, self.low_disk = segment_s, min_free_mb, low_disk
        self.disk_check_s = disk_check_s
        self.bitrate = bitrate
        self.session = time.strftime("%Y%m%d_%H%M%S")
        self.index_path = self.out_dir / f"talosaur_{self.session}_segments.jsonl"
        self.seq = 0
        self.segments: list[dict] = []  # {"path", "t0", "t1", "wall0", "keep", "deleted"}
        self.protect_since: float | None = None  # the current encounter's start: never delete after it
        self.lock = threading.Lock()
        self._rotation: threading.Thread | None = None
        self._last_disk_check = t0
        self.write_failed = False  # set by the output's error callback: switch to a fresh segment
        self.errors = 0
        self.retry_s = retry_s
        self._last_retry = -float("inf")
        self.encoder = _encoder(bitrate, fps, encoder_threads)
        path = self._next_path()
        self.split = SplittableOutput(self._output(path))
        self._open(path, t0)
        self.cam.start_encoder(self.encoder, self.split)
        self.recording = True
        free = self.free_mb()
        hours = free * 8e6 / bitrate / 3600 if bitrate else float("inf")  # MB -> bits -> s -> h
        log.info(
            f"continuous recording -> {self.out_dir} ({free:.0f} MB free, ~{hours:.1f} h at {bitrate / 1e6:.1f} Mbit/s)"
        )

    # ------------------------------------------------------------------ segments

    def _output(self, path: Path):
        out = self._pyav(str(path), format=self.fmt)
        out.error_callback = self._on_error  # otherwise a write error silently ends the recording
        return out

    def _on_error(self, e: Exception) -> None:
        """Called from the encoder's thread when writing a frame failed; the file is closed."""
        self.errors += 1
        self.write_failed = True
        log.error(f"writing {self.path} failed ({e}); switching to a new segment")

    def _next_path(self) -> Path:
        self.seq += 1
        return self.out_dir / f"talosaur_{self.session}_{self.seq:05d}{self.ext}"

    def _open(self, path: Path, t: float) -> None:
        self.segments.append(
            {"path": path, "t0": t, "t1": None, "wall0": time.time(), "keep": False, "deleted": False}
        )

    def _log_segment(self, seg: dict, event: str) -> None:  # also fsync'ed: the index must survive too
        rec = {
            "event": event,
            "file": seg["path"].name,
            "t0": round(seg["t0"], 3),
            "t1": None if seg["t1"] is None else round(seg["t1"], 3),
            "wall_start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(seg["wall0"])),
            "animal": seg["keep"],
        }
        with open(self.index_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @property
    def path(self) -> Path | None:
        return self.segments[-1]["path"] if self.segments else None

    def _rotate(self, t_req: float) -> None:
        m_req = time.monotonic()
        path = self._next_path()
        failed = self.write_failed
        try:
            self.split.split_output(self._output(path))  # waits for a keyframe (<= 1 s)
        except Exception as e:  # keep recording into the current file rather than stop
            log.error(f"segment switch failed ({e}); continuing in {self.path}")
            return
        if failed:
            self.write_failed = False
            log.warning(f"recording again, in {path.name}")
        t_switch = t_req + (time.monotonic() - m_req)
        with self.lock:
            prev = self.segments[-1]
            prev["t1"] = t_switch
            self._open(path, t_switch)
        _fsync(prev["path"])
        self._log_segment(prev, "closed_after_error" if failed else "closed")

    def tick(self, t: float) -> None:
        """Call once per frame with the app's clock: starts segment switches and disk checks."""
        if not self.recording:
            return
        busy = self._rotation is not None and self._rotation.is_alive()
        due = t - self.segments[-1]["t0"] >= self.segment_s
        retry = self.write_failed and t - self._last_retry >= self.retry_s
        if retry:
            self._last_retry = t
            self.check_disk(t)  # a full disk is the usual cause: make room first
        if not busy and self.recording and (due or retry):
            self._rotation = threading.Thread(target=self._rotate, args=(t,), daemon=True)
            self._rotation.start()
        if t - self._last_disk_check >= self.disk_check_s:
            self._last_disk_check = t
            self.check_disk(t)

    def wait_rotation(self, timeout: float = 5.0) -> None:
        if self._rotation is not None:
            self._rotation.join(timeout)

    # ------------------------------------------------------------------ disk

    def free_mb(self) -> float:
        try:
            return shutil.disk_usage(self.out_dir).free / 1e6
        except OSError:
            return float("inf")

    def check_disk(self, t: float) -> None:
        free = self.free_mb()
        if free >= self.min_free_mb:
            return
        if self.low_disk == "delete_empty":
            with self.lock:
                candidates = [
                    s
                    for s in self.segments
                    if s["t1"] is not None
                    and not s["keep"]
                    and not s["deleted"]
                    and (self.protect_since is None or s["t1"] < self.protect_since)
                ]
            for seg in candidates:  # oldest first
                try:
                    seg["path"].unlink(missing_ok=True)
                except OSError as e:
                    log.error(f"could not delete {seg['path']}: {e}")
                    continue
                seg["deleted"] = True
                self._log_segment(seg, "deleted_low_disk")
                log.warning(f"low disk ({free:.0f} MB free): deleted {seg['path'].name} (no animals in it)")
                free = self.free_mb()
                if free >= self.min_free_mb:
                    return
            if free >= self.min_free_mb / 4:
                log.warning(f"low disk: {free:.0f} MB free and only animal footage left; still recording")
                return
        log.error(f"disk nearly full ({free:.0f} MB free): stopping the recording to protect the system")
        self.stop_recording(t)

    # ------------------------------------------------------------------ interface

    def files_between(self, a: float, b: float) -> list[dict]:
        """Segments overlapping [a, b] with the offset of ``a`` in each; marks them as animal footage."""
        out = []
        with self.lock:
            for s in self.segments:
                if _overlaps(s["t0"], s["t1"], a, b) and not s["deleted"]:
                    s["keep"] = True
                    out.append({"file": s["path"].name, "offset_s": round(max(0.0, a - s["t0"]), 2)})
        return out

    def start(self, t: float) -> None:  # the state machine's clip events do not apply here
        pass

    def stop(self, t: float) -> None:
        pass

    def stop_recording(self, t: float) -> None:
        if not self.recording:
            return
        self.wait_rotation()
        try:
            self.cam.stop_encoder()
        except Exception as e:
            log.error(f"stopping the encoder failed: {e}")
        with self.lock:
            last = self.segments[-1]
            last["t1"] = t
        _fsync(last["path"])
        self._log_segment(last, "closed")
        self.recording = False

    def close(self, t: float | None = None) -> None:
        self.stop_recording(t if t is not None else self.segments[-1]["t0"])
