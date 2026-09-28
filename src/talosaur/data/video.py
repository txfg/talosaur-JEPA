"""Frame extraction from underwater video with PyAV (bundled FFmpeg; no system ffmpeg needed).

* samples at a fixed rate (default 1 fps) and drops near-identical consecutive frames
  (stationary ROV / fixed cameras), so static stretches cost little and dynamic ones keep
  all sampled frames;
* interlaced sources (NOAA ROV ProRes is 1080i) are de-interlaced by keeping one field,
  which is exact bob de-interlacing once frames are downscaled for training anyway;
* burned-in HUD/text overlays are detected from a handful of frames spread over the video
  (static, high-edge pixels in the top/bottom bands) and cropped away;
* frames are saved as JPEG with their short side resized (default 288 px).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from talosaur.data.dedup import DedupConfig, is_near_duplicate, phash64, thumbnail


@dataclass
class ExtractConfig:
    fps: float = 1.0
    short_side: int = 288
    jpeg_quality: int = 90
    deinterlace: str = "auto"  # auto | on | off
    overlay: str = "auto"  # auto | off
    dedup: bool = True
    max_hamming: int = 5
    max_thumb_dist: float = 1.0
    start_sec: float = 0.0
    end_sec: float | None = None
    max_frames: int | None = None
    threads: int = 0  # 0 = FFmpeg decides


@dataclass
class OverlayInfo:
    top: float = 0.0  # fraction of height to crop from the top
    bottom: float = 0.0
    detected: bool = False
    score_top: float = 0.0
    score_bottom: float = 0.0

    def crop_rows(self, h: int) -> tuple[int, int]:
        return int(math.ceil(self.top * h)), h - int(math.ceil(self.bottom * h))


def probe(path: str | Path) -> dict:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        rate = float(s.average_rate) if s.average_rate else (float(s.guessed_rate) if s.guessed_rate else 0.0)
        dur = float(s.duration * s.time_base) if s.duration else (c.duration / 1e6 if c.duration else None)
        return {
            "width": s.codec_context.width,
            "height": s.codec_context.height,
            "fps": rate,
            "duration": dur,
            "frames": s.frames or None,
            "codec": s.codec_context.name,
        }


def _gray_small(arr: np.ndarray, width: int = 160) -> np.ndarray:
    from talosaur.data.synthetic import resize_bilinear

    g = arr.astype(np.float32).mean(axis=-1) / 255.0
    h, w = g.shape
    return resize_bilinear(g, max(2, int(round(h * width / w))), width)


def detect_overlay(frames: list[np.ndarray], band: float = 0.18, edge_thr: float = 0.12) -> OverlayInfo:
    """Find static text/HUD bands at the top or bottom of the frame.

    Pixels that are (a) nearly constant across frames sampled over the whole video and (b) high
    local-contrast edges are overlay candidates. We only look at the top/bottom ``band`` rows,
    because a fixed camera's static *scene* must not be mistaken for an overlay, and HUDs on ROV
    video almost always live in those bands.
    """
    if len(frames) < 4:
        return OverlayInfo()
    g = np.stack([_gray_small(f) for f in frames])  # (T, h, w)
    tstd = g.std(axis=0)
    mean = g.mean(axis=0)
    gy, gx = np.gradient(mean)
    edges = np.hypot(gx, gy) > edge_thr
    static_edges = edges & (tstd < 0.02)
    h = g.shape[1]
    b = max(1, int(round(band * h)))

    def row_scores(rows):
        return static_edges[rows].mean(axis=1)

    info = OverlayInfo()
    top_rows = row_scores(slice(0, b))
    bot_rows = row_scores(slice(h - b, h))
    info.score_top, info.score_bottom = float(top_rows.mean()), float(bot_rows.mean())
    active_top = np.flatnonzero(top_rows > 0.08)
    active_bot = np.flatnonzero(bot_rows > 0.08)
    if len(active_top):
        info.top = float((active_top.max() + 2) / h)
        info.detected = True
    if len(active_bot):
        info.bottom = float((b - active_bot.min() + 1) / h)
        info.detected = True
    return info


def _to_rgb_array(frame, deinterlace: bool) -> np.ndarray:
    arr = frame.to_ndarray(format="rgb24")
    if deinterlace:
        arr = arr[0::2]  # keep one field; aspect is restored at resize time
    return arr


def _resize_save(arr: np.ndarray, out_path: Path, short_side: int, quality: int, aspect_fix: float) -> tuple[int, int]:
    from PIL import Image

    from talosaur.data.imageio import save_jpeg

    h, w = arr.shape[:2]
    true_h = h * aspect_fix
    scale = min(1.0, short_side / min(w, true_h))
    nw, nh = max(1, round(w * scale)), max(1, round(true_h * scale))
    im = Image.fromarray(arr)
    if (nw, nh) != (w, h):
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    save_jpeg(im, out_path, quality)
    return nw, nh


def extract_frames(video_path: str | Path, out_dir: str | Path, video_id: str, cfg: ExtractConfig | None = None) -> dict:
    """Extract frames; returns ``{"frames": [...], "overlay": {...}, "stats": {...}}``.

    Each frame dict has ``path`` (relative to ``out_dir``), ``t_sec``, ``width``, ``height``,
    ``phash``. The overlay crop (if any) is already applied to the saved frames.
    """
    import av

    cfg = cfg or ExtractConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(video_path)
    duration = info["duration"]

    # 1) overlay detection from frames spread across the video
    overlay = OverlayInfo()
    if cfg.overlay == "auto":
        samples = _sample_spread(video_path, n=16, duration=duration)
        overlay = detect_overlay(samples)

    # 2) sequential decode with timestamp gating
    dcfg = DedupConfig(max_hamming=cfg.max_hamming, max_thumb_dist=cfg.max_thumb_dist)
    frames_out: list[dict] = []
    n_decoded = n_sampled = n_dups = 0
    last_hash = last_thumb = None
    next_t = cfg.start_sec
    step = 1.0 / cfg.fps
    interlaced_seen = False
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if cfg.threads:
            stream.codec_context.thread_count = cfg.threads
        if cfg.start_sec > 0:
            container.seek(int(cfg.start_sec / stream.time_base), stream=stream, any_frame=False, backward=True)
        for frame in container.decode(stream):
            n_decoded += 1
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if cfg.end_sec is not None and t > cfg.end_sec:
                break
            if t + 1e-6 < next_t:
                continue
            next_t = max(next_t + step, t + step * 0.5)
            n_sampled += 1
            interlaced = bool(getattr(frame, "interlaced_frame", False))
            interlaced_seen |= interlaced
            deint = cfg.deinterlace == "on" or (cfg.deinterlace == "auto" and interlaced)
            arr = _to_rgb_array(frame, deint)
            if overlay.detected:
                r0, r1 = overlay.crop_rows(arr.shape[0])
                arr = arr[r0:r1]
            h_small = phash64(arr)
            th = thumbnail(arr)
            if cfg.dedup and last_hash is not None and is_near_duplicate(h_small, th, last_hash, last_thumb, dcfg):
                n_dups += 1
                continue
            last_hash, last_thumb = h_small, th
            name = f"{video_id}_t{t:09.2f}.jpg".replace("/", "_")
            w, h = _resize_save(arr, out_dir / name, cfg.short_side, cfg.jpeg_quality, 2.0 if deint else 1.0)
            frames_out.append({"path": name, "t_sec": round(t, 3), "width": w, "height": h, "phash": int(h_small)})
            if cfg.max_frames and len(frames_out) >= cfg.max_frames:
                break
    return {
        "frames": frames_out,
        "overlay": {
            "detected": overlay.detected,
            "top": overlay.top,
            "bottom": overlay.bottom,
            "score_top": overlay.score_top,
            "score_bottom": overlay.score_bottom,
        },
        "stats": {
            "decoded": n_decoded,
            "sampled": n_sampled,
            "dropped_near_duplicates": n_dups,
            "kept": len(frames_out),
            "interlaced": interlaced_seen,
            **{k: info[k] for k in ("width", "height", "fps", "duration", "codec")},
        },
    }


def _sample_spread(video_path: str | Path, n: int, duration: float | None) -> list[np.ndarray]:
    """Decode ~n frames spread over the video (seeking; falls back to a sequential scan)."""
    import av

    out: list[np.ndarray] = []
    with av.open(str(video_path)) as c:
        s = c.streams.video[0]
        if duration and duration > 2.0:
            for i in range(n):
                t = duration * (i + 0.5) / n
                try:
                    c.seek(int(t / s.time_base), stream=s, any_frame=False, backward=True)
                    for fr in c.decode(s):
                        out.append(fr.to_ndarray(format="rgb24"))
                        break
                except Exception:  # some containers can't seek; fall back to a sequential scan
                    out = []
                    break
            if len(out) >= min(n, 4):
                return out
    with av.open(str(video_path)) as c:
        s = c.streams.video[0]
        total = s.frames or 0
        stride = max(1, total // n) if total else 5
        for i, fr in enumerate(c.decode(s)):
            if i % stride == 0:
                out.append(fr.to_ndarray(format="rgb24"))
            if len(out) >= n:
                break
    return out
