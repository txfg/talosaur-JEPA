"""Procedural "underwater-like" scenes with exact ground truth, for tests, CI and debug runs.

They are not meant to be realistic, only to exercise every code path with known answers:
boxes, masks, frame labels, clear / murky / dark conditions, videos with moving animals,
drifting particles, burned-in text overlays and stretches of empty open water.

numpy only (Pillow / PyAV are needed just for the ``write_*`` helpers).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WATER_COLORS: dict[str, tuple[float, float, float]] = {
    "ocean": (0.03, 0.22, 0.38),
    "coastal": (0.06, 0.32, 0.30),
    "lake": (0.20, 0.28, 0.10),
}
CONDITIONS = ("clear", "murky", "dark")


@dataclass
class Scene:
    image: np.ndarray  # (H, W, 3) uint8
    boxes: np.ndarray  # (N, 4) float32, normalised x0, y0, x1, y1
    mask: np.ndarray  # (H, W) bool, animal pixels
    condition: str
    water: str
    labels: list[str] = field(default_factory=list)

    @property
    def has_animal(self) -> bool:
        return len(self.boxes) > 0


# --------------------------------------------------------------------------- primitives


def resize_bilinear(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize of an (H, W) or (H, W, C) float array (align-corners=False)."""
    a = np.asarray(a, dtype=np.float32)
    sh, sw = a.shape[:2]
    ys = (np.arange(h, dtype=np.float32) + 0.5) * sh / h - 0.5
    xs = (np.arange(w, dtype=np.float32) + 0.5) * sw / w - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, sh - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, sw - 1)
    y1 = np.clip(y0 + 1, 0, sh - 1)
    x1 = np.clip(x0 + 1, 0, sw - 1)
    wy = np.clip(ys - y0, 0, 1)
    wx = np.clip(xs - x0, 0, 1)
    if a.ndim == 3:
        wy = wy[:, None, None]
        wx = wx[None, :, None]
    else:
        wy = wy[:, None]
        wx = wx[None, :]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def box_blur(img: np.ndarray, k: int) -> np.ndarray:
    """Separable box blur with odd kernel size ``k`` (edge-replicated), float in / float out."""
    if k <= 1:
        return img
    r = k // 2
    out = img.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis, dtype=np.float64)
        hi = np.take(c, np.arange(k, c.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, c.shape[axis] - k), axis=axis)
        out = ((hi - lo) / k).astype(np.float32)
    return out


def smooth_noise(rng: np.random.Generator, h: int, w: int, cells: int = 6) -> np.ndarray:
    """Low-frequency noise in [0, 1]."""
    n = rng.random((cells, max(2, int(cells * w / max(h, 1)))), dtype=np.float32)
    return resize_bilinear(n, h, w)


# --------------------------------------------------------------------------- scene parts


def _background(rng: np.random.Generator, h: int, w: int, water: str, openwater: bool) -> np.ndarray:
    base = np.array(WATER_COLORS[water], dtype=np.float32)
    grad = np.linspace(1.25, 0.65, h, dtype=np.float32)[:, None, None]  # brighter towards surface
    img = base[None, None, :] * grad * (0.85 + 0.3 * smooth_noise(rng, h, w, 4)[..., None])
    if not openwater:
        horizon = int(h * rng.uniform(0.45, 0.8))
        floor_col = np.array([0.30, 0.28, 0.20], dtype=np.float32) * rng.uniform(0.6, 1.2)
        tex = 0.55 + 0.45 * smooth_noise(rng, h - horizon, w, 12)
        tex += 0.15 * (rng.random((h - horizon, w), dtype=np.float32) - 0.5)
        ramp = np.linspace(0.3, 1.0, h - horizon, dtype=np.float32)[:, None]
        floor = floor_col[None, None, :] * (tex * ramp)[..., None]
        img[horizon:] = 0.4 * img[horizon:] + 0.6 * floor
    return np.clip(img, 0, 1)


def _draw_fish(
    img: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    cx: float,
    cy: float,
    length: float,
    angle: float,
    color: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Rasterise an ellipse body + triangular tail. Returns the pixel bbox or None if off-screen."""
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ca, sa = math.cos(angle), math.sin(angle)
    u = (xx - cx) * ca + (yy - cy) * sa  # along body
    v = -(xx - cx) * sa + (yy - cy) * ca  # across body
    a, b = length / 2.0, length / 5.0
    body = (u / a) ** 2 + (v / b) ** 2 <= 1.0
    tu = u + a  # tail starts at the back of the body
    tail = (tu <= 0) & (tu >= -0.45 * length) & (np.abs(v) <= (-tu) * 0.9)
    shape = body | tail
    if not shape.any():
        return None
    stripes = 0.75 + 0.25 * np.sign(np.sin(u * (6.0 / max(length, 1.0)) * math.pi))
    shade = (0.7 + 0.3 * (1 - np.clip(np.abs(v) / b, 0, 1)))[..., None]
    fish = color[None, None, :] * shade * stripes[..., None]
    img[shape] = fish[shape]
    mask |= shape
    ys, xs = np.nonzero(shape)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _particles(img: np.ndarray, rng: np.random.Generator, n: int, brightness: float) -> None:
    h, w = img.shape[:2]
    if n <= 0:
        return
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    val = brightness * rng.uniform(0.4, 1.0, n).astype(np.float32)
    img[ys, xs] = np.clip(img[ys, xs] + val[:, None], 0, 1)


def _apply_condition(img: np.ndarray, rng: np.random.Generator, condition: str, water: str) -> np.ndarray:
    col = np.array(WATER_COLORS[water], dtype=np.float32) * 1.4
    if condition == "clear":
        t = rng.uniform(0.0, 0.15)
        out = img * (1 - t) + col * t
    elif condition == "murky":
        t = rng.uniform(0.5, 0.75)
        out = box_blur(img, int(rng.choice([3, 5, 7]))) * (1 - t) + col * t
    elif condition == "dark":
        gain = rng.uniform(0.04, 0.14)
        out = img * gain
        out = out + rng.normal(0.0, rng.uniform(0.006, 0.018), out.shape).astype(np.float32)
    else:  # pragma: no cover - guarded by make_scene
        raise ValueError(condition)
    return np.clip(out, 0, 1)


# --------------------------------------------------------------------------- public API


def make_scene(
    rng: np.random.Generator,
    h: int = 256,
    w: int = 256,
    condition: str | None = None,
    n_animals: int | None = None,
    water: str | None = None,
    openwater: bool | None = None,
) -> Scene:
    """One synthetic still with 0..3 'fish' and exact boxes/mask."""
    condition = condition or str(rng.choice(CONDITIONS))
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    water = water or str(rng.choice(list(WATER_COLORS)))
    openwater = bool(rng.random() < 0.35) if openwater is None else openwater
    n_animals = int(rng.integers(0, 4)) if n_animals is None else n_animals

    img = _background(rng, h, w, water, openwater)
    mask = np.zeros((h, w), dtype=bool)
    boxes = []
    for _ in range(n_animals):
        for _attempt in range(10):
            length = rng.uniform(0.12, 0.45) * min(h, w)
            cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
            color = rng.uniform(0.35, 1.0, 3).astype(np.float32)
            bb = _draw_fish(img, mask, rng, cx, cy, length, rng.uniform(-0.6, 0.6), color)
            if bb is not None:
                boxes.append([bb[0] / w, bb[1] / h, bb[2] / w, bb[3] / h])
                break
    _particles(img, rng, int(rng.integers(0, 40)), 0.5)
    img = _apply_condition(img, rng, condition, water)
    return Scene(
        image=(img * 255.0 + 0.5).astype(np.uint8),
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        mask=mask,
        condition=condition,
        water=water,
        labels=["fish"] * len(boxes),
    )


def make_video_frames(
    rng: np.random.Generator,
    n_frames: int = 60,
    h: int = 144,
    w: int = 256,
    condition: str = "clear",
    water: str = "ocean",
    animal: bool = True,
    empty_span: tuple[float, float] | None = None,
    overlay_text: bool = False,
    static_span: tuple[float, float] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """A short clip: static background, one fish swimming across, drifting particles.

    ``empty_span=(a, b)`` removes the animal for that fraction of the clip (open-water stretch);
    ``static_span=(a, b)`` freezes the frame content (near-duplicate frames for dedup tests);
    ``overlay_text`` burns a static high-contrast "HUD" band into the top rows.
    Returns frames (T, H, W, 3) uint8 and per-frame (N, 4) normalised boxes.
    """
    bg = _background(rng, h, w, water, openwater=False)
    color = rng.uniform(0.4, 1.0, 3).astype(np.float32)
    length = rng.uniform(0.2, 0.35) * min(h, w) * 1.6
    x0, x1 = -0.1 * w, 1.1 * w
    y = rng.uniform(0.35, 0.65) * h
    n_part = 60
    px = rng.uniform(0, w, n_part)
    py = rng.uniform(0, h, n_part)
    vx = rng.normal(0.4, 0.3, n_part)
    vy = rng.normal(0.8, 0.3, n_part)
    hud = None
    if overlay_text:
        hud = np.zeros((h, w), dtype=bool)
        band = max(6, h // 12)
        blocks = rng.random((band // 3, w // 4)) < 0.45
        hud[: (band // 3) * 3, : (w // 4) * 4] = np.kron(blocks, np.ones((3, 4), dtype=bool))

    frames, boxes = [], []
    frozen: tuple[np.ndarray, np.ndarray] | None = None
    for t in range(n_frames):
        f = t / max(n_frames - 1, 1)
        if static_span and static_span[0] <= f <= static_span[1] and frozen is not None:
            frames.append(frozen[0].copy())
            boxes.append(frozen[1].copy())
            continue
        img = bg.copy()
        mask = np.zeros((h, w), dtype=bool)
        bx = []
        visible = animal and not (empty_span and empty_span[0] <= f <= empty_span[1])
        if visible:
            cx = x0 + (x1 - x0) * f
            bb = _draw_fish(img, mask, rng, cx, y + 4 * math.sin(6 * f), length, 0.0, color)
            if bb is not None:
                bx.append([max(bb[0], 0) / w, max(bb[1], 0) / h, min(bb[2], w) / w, min(bb[3], h) / h])
        pxi = ((px + vx * t) % w).astype(int)
        pyi = ((py + vy * t) % h).astype(int)
        img[pyi, pxi] = np.clip(img[pyi, pxi] + 0.35, 0, 1)
        img = _apply_condition(img, np.random.default_rng(1234), condition, water)  # fixed per clip
        if hud is not None:
            img[hud] = 1.0
        u8 = (img * 255.0 + 0.5).astype(np.uint8)
        b = np.asarray(bx, dtype=np.float32).reshape(-1, 4)
        frames.append(u8)
        boxes.append(b)
        frozen = (u8, b)
    return np.stack(frames), boxes


def write_image_dataset(
    root: str | Path,
    n: int,
    seed: int = 0,
    size: tuple[int, int] = (256, 256),
    p_animal: float = 0.6,
    source: str = "synthetic",
) -> Path:
    """Write ``n`` JPEG scenes + masks and a COCO-style ``annotations.json``. Needs Pillow.

    Each image carries its condition and a ``group`` (5 images per group) so grouped splits
    can be tested.
    """
    from PIL import Image

    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    images, annotations = [], []
    ann_id = 1
    for i in range(n):
        k = int(rng.integers(1, 4)) if rng.random() < p_animal else 0
        sc = make_scene(rng, size[0], size[1], condition=CONDITIONS[i % 3], n_animals=k)
        name = f"{source}_{i:05d}.jpg"
        Image.fromarray(sc.image).save(root / "images" / name, quality=92)
        Image.fromarray(sc.mask.astype(np.uint8) * 255).save(root / "masks" / name.replace(".jpg", ".png"))
        images.append(
            {
                "id": i + 1,
                "file_name": f"images/{name}",
                "mask_file": f"masks/{name.replace('.jpg', '.png')}",
                "width": size[1],
                "height": size[0],
                "condition": sc.condition,
                "group": f"{source}_g{i // 5:04d}",
                "has_animal": sc.has_animal,
            }
        )
        for bx in sc.boxes:
            x0, y0, x1, y1 = (bx * np.array([size[1], size[0], size[1], size[0]])).tolist()
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i + 1,
                    "category_id": 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    coco = {
        "info": {"description": "Talosaur synthetic test set", "version": "1"},
        "licenses": [{"id": 1, "name": "CC0-1.0"}],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "fish", "supercategory": "animal"}],
    }
    path = root / "annotations.json"
    path.write_text(json.dumps(coco))
    return path


def write_video(path: str | Path, frames: np.ndarray, fps: int = 10, codec: str = "libx264") -> Path:
    """Encode (T, H, W, 3) uint8 frames to a video file with PyAV."""
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames.shape
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        if codec == "libx264":
            stream.options = {"crf": "18", "preset": "veryfast"}
        for i in range(t):
            frame = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
