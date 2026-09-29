"""A rendered camera for the digital twin: the simulated world as the Pi's camera would see it.

Per frame, in linear light, then auto-exposure, noise and gamma:

* **water**: ambient light from the surface, falling with depth (``kd_per_m``) and with the time of
  day, brighter toward the top of the frame;
* **lamp**: a beam along the camera axis. Light reaching an animal falls with distance squared and
  with the water's attenuation both ways; the lamp also lights the water in front of the camera
  (backscatter veil);
* **animals**: real animals cut out of held-out FathomNet test images (never the training split),
  one taxonomic group per simulated species. Each is placed by the guidance camera model (flat-port
  refraction included) at its true bearing, elevation and angular size, lit by the lamp and the
  ambient light, and faded toward the water colour with distance;
* **marine snow**: particles fixed in the water around the vehicle, bright near the lamp;
* **bioluminescent flashes**: brief blue-green blobs from animals, lamp or not;
* **camera**: auto-exposure with a gain limit (dark water stays dark), gain noise, gamma 2.2.

The sprites are derived from FathomNet images, which must not be redistributed: the sprite bank
and anything rendered with it (frames, videos) stay local.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from talosaur.guidance.camera_model import CameraModel

# simulated species (talosaur.sim.world.default_species) -> sprite group(s)
SPECIES_GROUPS = {
    "lanternfish": ("fish",),
    "shrimp": ("shrimp",),
    "medusa": ("medusa", "ctenophore"),
    "siphonophore": ("siphonophore",),
    "squid": ("squid",),
}
FISH_TAXA = {"Actinopterygii", "Actinopteri", "Teleostei", "Chondrichthyes", "Elasmobranchii", "Myxini"}


def taxon_group(ancestors: set[str]) -> str | None:
    """Sprite group of a WoRMS concept from its ancestor names (midwater animals only)."""
    if ancestors & FISH_TAXA:
        return "fish"
    if "Cephalopoda" in ancestors:
        return "squid"
    if "Siphonophorae" in ancestors:
        return "siphonophore"
    if "Ctenophora" in ancestors:
        return "ctenophore"
    if ancestors & {"Scyphozoa", "Cubozoa", "Hydrozoa"}:
        return "medusa"
    if ancestors & {"Euphausiacea", "Mysida", "Lophogastrida"} or (
        "Decapoda" in ancestors and not ancestors & {"Brachyura", "Anomura", "Astacidea", "Achelata"}
    ):
        return "shrimp"
    return None


def concept_groups(concepts, cache_path: str | Path, workers: int = 5) -> dict[str, str | None]:
    """Sprite group per FathomNet concept via WoRMS ancestors (cached as JSON)."""
    from concurrent.futures import ThreadPoolExecutor

    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = sorted({c for c in concepts if c and c not in cache})
    if todo:
        from fathomnet.api import worms

        def look(c):
            try:
                return c, taxon_group(set(worms.get_ancestors_names(c) or []) | {c})
            except Exception:
                return c, None

        with ThreadPoolExecutor(workers) as ex:
            cache.update(dict(ex.map(look, todo)))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, sort_keys=True))
    return {c: cache.get(c) for c in concepts}


# --------------------------------------------------------------------------- sprites


def cut_out(crop: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """RGBA sprite from an RGB crop whose animal box is ``box`` (x0, y0, x1, y1) inside it.

    The background colour is the median of the margin around the box; alpha is the distance from
    it, scaled by the margin's own spread, feathered, and kept inside the box. None if the animal
    barely differs from its background.
    """
    h, w = crop.shape[:2]
    x0, y0, x1, y1 = box
    rgb = crop.astype(np.float32)
    ring = np.ones((h, w), bool)
    ring[y0:y1, x0:x1] = False
    if ring.sum() < 16:
        return None
    bg = np.median(rgb[ring], axis=0)
    dist = np.linalg.norm(rgb - bg, axis=2)
    noise = float(np.median(dist[ring])) + 1.0
    inside = dist[y0:y1, x0:x1]
    hi = max(float(np.quantile(inside, 0.9)), 3.0 * noise + 8.0)
    a = np.clip((dist - 2.5 * noise) / max(1e-3, hi - 2.5 * noise), 0.0, 1.0)
    k = np.array([0.25, 0.5, 0.25], np.float32)  # 3x3 smoothing
    a = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, a)
    a = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, a)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max(1.0, (x1 - x0) / 2), max(1.0, (y1 - y0) / 2)
    ell = np.clip(1.25 - np.hypot((xx - cx) / rx, (yy - cy) / ry), 0.0, 1.0) / 0.25  # soft edge
    a = a * np.clip(ell, 0.0, 1.0)
    a[~((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1))] = 0.0
    if a[y0:y1, x0:x1].mean() < 0.08:
        return None
    out = np.dstack([crop[y0:y1, x0:x1], (a[y0:y1, x0:x1] * 255).astype(np.uint8)])
    return out


@dataclass
class SpriteBank:
    """RGBA uint8 sprites per group, with the image each came from."""

    sprites: dict[str, list[np.ndarray]] = field(default_factory=dict)
    sources: dict[str, list[str]] = field(default_factory=dict)

    def pick(self, group_options, key: int) -> tuple[np.ndarray, str] | None:
        for g in group_options:
            s = self.sprites.get(g)
            if s:
                i = key % len(s)
                return s[i], g
        return None

    def save(self, path: str | Path) -> None:
        arrs, meta = {}, []
        for g, ss in self.sprites.items():
            for s, src in zip(ss, self.sources[g]):
                arrs[f"s{len(meta)}"] = s
                meta.append([g, src])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrs)

    @classmethod
    def load(cls, path: str | Path) -> SpriteBank:
        z = np.load(path)
        bank = cls()
        for i, (g, src) in enumerate(json.loads(str(z["meta"]))):
            bank.sprites.setdefault(g, []).append(z[f"s{i}"])
            bank.sources.setdefault(g, []).append(src)
        return bank

    @classmethod
    def build(
        cls,
        index_path: str | Path,
        root: str | Path,
        groups_cache: str | Path,
        splits=("test", "val"),
        min_px: int = 24,
        max_side: int = 160,
        per_group: int = 300,
        seed: int = 0,
    ) -> SpriteBank:
        """Cut animals of the midwater groups out of FathomNet ``splits`` images."""
        from PIL import Image

        from talosaur.data.imageio import load_rgb
        from talosaur.data.index import read_table

        df = read_table(index_path)
        df = df[(df.source == "fathomnet") & df.split.isin(splits) & df.boxes.map(len).gt(0)]
        concepts = {c for labels in df.box_labels for c in labels}
        groups = concept_groups(concepts, groups_cache)
        cands: dict[str, list] = {}
        for r in df.itertuples():
            for b, lab, an in zip(r.boxes, r.box_labels, r.box_is_animal):
                g = groups.get(lab)
                if not an or g is None:
                    continue
                if max((b[2] - b[0]) * r.width, (b[3] - b[1]) * r.height) >= min_px:
                    cands.setdefault(g, []).append((r.path, r.image_id, b, r.width, r.height))
        rng = np.random.default_rng(seed)
        bank = cls()
        for g, items in cands.items():
            for j in rng.permutation(len(items)):
                if len(bank.sprites.get(g, [])) >= per_group:
                    break
                path, iid, b, w, h = items[j]
                x0, y0, x1, y1 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
                pad = 0.2 * max(x1 - x0, y1 - y0) + 4
                X0, Y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
                X1, Y1 = int(min(w, x1 + pad)), int(min(h, y1 + pad))
                img = load_rgb(Path(root) / path)
                crop = img[Y0:Y1, X0:X1]
                s = cut_out(
                    crop, (int(x0 - X0), int(y0 - Y0), int(math.ceil(x1 - X0)), int(math.ceil(y1 - Y0)))
                )
                if s is None:
                    continue
                if max(s.shape[:2]) > max_side:
                    f = max_side / max(s.shape[:2])
                    size = (max(1, round(s.shape[1] * f)), max(1, round(s.shape[0] * f)))
                    s = np.asarray(Image.fromarray(s).resize(size, Image.Resampling.LANCZOS))
                bank.sprites.setdefault(g, []).append(s)
                bank.sources.setdefault(g, []).append(iid)
        return bank


# --------------------------------------------------------------------------- renderer

LAMP_COLOURS = {"white": (1.0, 1.0, 1.0), "red": (1.0, 0.08, 0.03)}  # far-red: the sensor's red channel
AMBIENT_COLOUR = np.array([0.10, 0.45, 0.75], np.float32)  # downwelling light at depth is blue-green
FLASH_COLOUR = np.array([0.15, 0.85, 1.0], np.float32)


@dataclass
class RenderConfig:
    hw: tuple[int, int] = (224, 416)  # rendered frame (the model gets it downsampled 2x: 112 x 208)
    model_hw: tuple[int, int] = (112, 208)
    max_range_m: float = 12.0
    kd_per_m: float = 0.045  # diffuse attenuation of daylight with depth (clear ocean) [E]
    c_per_m: float = 0.15  # beam attenuation along a line of sight [E]
    night_light: float = 2e-6  # surface light at night (moon, stars) relative to day
    twilight_h: float = 0.75  # daylight fades over this long around sunset / sunrise
    lamp: str = "red"  # red (far-red plan, docs/TWILIGHT_ZONE.md 7.6) | white
    lamp_power: float = (
        0.8  # linear light on a target 1 m ahead at full level (daylight at the surface = 1) [E]
    )
    lamp_beam_deg: float = 35.0
    backscatter: float = 0.015
    snow_per_m3: float = 3.0
    snow_box_m: float = 6.0
    flash_rate: dict[str, float] = field(
        default_factory=lambda: {
            "lanternfish": 0.02,
            "shrimp": 0.04,
            "medusa": 0.03,
            "siphonophore": 0.01,
            "squid": 0.01,
        }
    )
    flash_s: float = 0.4
    flash_radiance: float = 2e-3
    ae_target: float = 0.18  # auto-exposure: linear mean it aims for ...
    ae_max_gain: float = 25.0  # ... up to this gain (surface daylight = 1): dark water stays dark
    ae_tau_s: float = 0.5
    read_noise: float = 0.002
    gain_noise: float = 0.006  # after the ISP's denoising
    min_contrast: float = 0.04  # an animal counts as visible (ground truth) above this contrast (noise-free)
    seed: int = 0


def surface_light(hour: float, sunrise: float, sunset: float, cfg: RenderConfig) -> float:
    """Daylight at the surface, 1 at midday, ``night_light`` at night, log-smooth in twilight."""
    tw = cfg.twilight_h

    def ramp(x):  # 0 -> 1 over [-tw, tw]
        return float(np.clip((x + tw) / (2 * tw), 0.0, 1.0))

    day = min(ramp(hour - sunrise), ramp(sunset - hour))
    return float(math.exp(math.log(cfg.night_light) * (1.0 - day)))


class Renderer:
    def __init__(self, bank: SpriteBank, camera: CameraModel | None = None, cfg: RenderConfig | None = None):
        self.bank = bank
        self.cam = camera or CameraModel()
        self.cfg = cfg or RenderConfig()
        self.hfov, self.vfov = self.cam.effective_fov()
        self.rng = np.random.default_rng(self.cfg.seed)
        H, W = self.cfg.hw
        self._ys, self._xs = np.mgrid[0:H, 0:W].astype(np.float32)
        self._ys /= H
        self._xs /= W
        n = int(self.cfg.snow_per_m3 * self.cfg.snow_box_m**3)
        self._snow = None if n == 0 else self.rng.uniform(-0.5, 0.5, size=(n, 3)) * self.cfg.snow_box_m
        self._flash_until: dict[int, float] = {}
        self._gain = self.cfg.ae_max_gain
        self._t = None

    # --- geometry
    def _view(self, d: np.ndarray, fwd: np.ndarray, rgt: np.ndarray):
        """World offsets (N, 3) -> forward distance, yaw and pitch (deg)."""
        f = d @ fwd
        r = d @ rgt
        up = -d[:, 2]
        yaw = np.degrees(np.arctan2(r, f))
        pitch = np.degrees(np.arctan2(up, np.hypot(f, r)))
        return f, yaw, pitch

    def _in_fov(self, f, yaw, pitch, margin=0.0):
        return (f > 0.05) & (np.abs(yaw) < self.hfov / 2 + margin) & (np.abs(pitch) < self.vfov / 2 + margin)

    def _lamp_at(self, dist, off_axis_deg, level):
        c = self.cfg
        beam = np.exp(-((off_axis_deg / c.lamp_beam_deg) ** 2))
        return level * c.lamp_power * beam * np.exp(-c.c_per_m * dist) / np.maximum(dist, 0.3) ** 2

    # --- frame
    def render(self, vehicle, world, light: float, t: float):
        """-> (rgb (H, W, 3) uint8, rgb for the model (h, w, 3) uint8, truth list, luma) where truth is
        [(animal, u, v, distance_m, contrast, (x0, y0, x1, y1) normalised, species)] for visible animals."""
        c = self.cfg
        H, W = c.hw
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        light = float(np.clip(light or 0.0, 0.0, 1.0))
        lamp_col = np.array(LAMP_COLOURS[c.lamp], np.float32)
        wc = world.cfg
        depth = float(vehicle.pos[2])
        ambient = surface_light(world.hour(), wc.sunrise_h, wc.sunset_h, c) * math.exp(-c.kd_per_m * depth)

        # water: ambient, brighter toward the surface; the lamp's backscatter veil near the lamp
        grad = (1.4 - 0.8 * self._ys)[..., None]
        img = ambient * grad * AMBIENT_COLOUR
        veil = np.exp(-(((self._xs - 0.5) / 0.55) ** 2) - (((self._ys - 1.1) / 0.75) ** 2))
        img = img + (light * c.lamp_power * c.backscatter * veil)[..., None] * lamp_col
        bg = img.copy()

        fwd, rgt = vehicle.forward(), vehicle.right()
        truth = []
        # marine snow: particles fixed in the water, kept in a box around the vehicle
        if self._snow is not None:
            box = c.snow_box_m
            rel = (self._snow - vehicle.pos + box / 2) % box - box / 2
            self._snow = vehicle.pos + rel
            f, yaw, pitch = self._view(rel, fwd, rgt)
            m = self._in_fov(f, yaw, pitch)
            if m.any():
                dist = np.linalg.norm(rel[m], axis=1)
                us, vs = self._project(yaw[m], pitch[m])
                lum = (0.6 * self._lamp_at(dist, np.degrees(np.arccos(np.clip(f[m] / dist, -1, 1))), light))[
                    :, None
                ] * lamp_col + 0.3 * ambient * AMBIENT_COLOUR
                lum = lum * np.exp(-c.c_per_m * dist)[:, None] * np.clip(0.4 / dist, 0.05, 1.0)[:, None]
                xi = np.clip((us * W).astype(int), 0, W - 1)
                yi = np.clip((vs * H).astype(int), 0, H - 1)
                np.add.at(img, (yi, xi), lum)

        # animals, far to near
        idx = world.near(vehicle.pos, c.max_range_m)
        if len(idx):
            d = world.offsets(idx, vehicle.pos)
            dist = np.linalg.norm(d, axis=1)
            f, yaw, pitch = self._view(d, fwd, rgt)
            ang = np.degrees(2 * np.arctan(world.size[idx] / (2 * np.maximum(dist, 1e-3))))
            m = (dist < c.max_range_m) & self._in_fov(f, yaw, pitch, margin=0.5 * ang)
            names = [s.name for s in world.cfg.species]
            for k in np.argsort(-dist[m]):
                i = int(idx[m][k])
                dd, yw, pt, a = float(dist[m][k]), float(yaw[m][k]), float(pitch[m][k]), float(ang[m][k])
                sp = names[int(world.sp[i])]
                self._maybe_flash(i, sp, dt, t)
                pick = self.bank.pick(SPECIES_GROUPS.get(sp, ()), key=i * 2654435761 % (1 << 31))
                if pick is None:
                    continue
                sprite, _ = pick
                off_axis = math.degrees(math.acos(max(-1.0, min(1.0, float(f[m][k]) / dd))))
                lit = self._lamp_at(dd, off_axis, light) * lamp_col + ambient * AMBIENT_COLOUR
                box = self._paste(img, bg, sprite, yw, pt, a, dd, lit, flip=(i % 2 == 1))
                if box is not None:
                    truth.append([i, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, dd, box, sp])
            for i, until in list(self._flash_until.items()):  # flashes (drawn after the bodies)
                if until < t:
                    del self._flash_until[i]
                    continue
                j = np.flatnonzero(idx[m] == i)
                if len(j):
                    self._blob(img, float(yaw[m][j[0]]), float(pitch[m][j[0]]), float(dist[m][j[0]]))

        # camera: auto-exposure (limited gain), noise, gamma
        mean = float(img.mean()) + 1e-12
        want = float(np.clip(c.ae_target / mean, 1.0, c.ae_max_gain))
        a = 1.0 if dt == 0 else 1.0 - math.exp(-dt / c.ae_tau_s)
        self._gain = math.exp(math.log(self._gain) + a * (math.log(want) - math.log(self._gain)))
        x = img * self._gain
        clean = np.clip(x, 0.0, 1.0) ** (1 / 2.2) * 255  # before noise: what counts as visible
        sigma = c.read_noise + c.gain_noise * math.sqrt(self._gain / c.ae_max_gain)
        x = x + self.rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
        rgb = (np.clip(x, 0.0, 1.0) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
        bg8 = (np.clip(bg * self._gain, 0.0, 1.0) ** (1 / 2.2) * 255).astype(np.float32)

        vis = []
        for i, u, v, dd, box, sp in truth:
            x0, y0, x1, y1 = (
                int(box[0] * W),
                int(box[1] * H),
                int(math.ceil(box[2] * W)),
                int(math.ceil(box[3] * H)),
            )
            patch = clean[y0:y1, x0:x1]
            contrast = float(np.abs(patch - bg8[y0:y1, x0:x1]).mean() / 255) if patch.size else 0.0
            if contrast >= c.min_contrast:
                vis.append((i, u, v, dd, contrast, tuple(box), sp))
        mh, mw = c.model_hw
        small = rgb.reshape(mh, H // mh, mw, W // mw, 3).mean(axis=(1, 3)).astype(np.uint8)
        return rgb, small, vis, float(small.mean()) / 255.0

    def _project(self, yaw, pitch):
        uv = np.array(
            [self.cam.project(float(a), float(b)) for a, b in zip(np.atleast_1d(yaw), np.atleast_1d(pitch))]
        )
        return uv[:, 0], uv[:, 1]

    def _paste(self, img, bg, sprite, yaw, pitch, ang, dist, lit, flip=False):
        from PIL import Image

        c = self.cfg
        H, W = c.hw
        u, v = self.cam.project(yaw, pitch)
        ul, _ = self.cam.project(yaw - ang / 2, pitch)
        ur, _ = self.cam.project(yaw + ang / 2, pitch)
        long_px = max(2.0, abs(ur - ul) * W)
        sh, sw = sprite.shape[:2]
        f = long_px / max(sh, sw)
        w, h = max(1, round(sw * f)), max(1, round(sh * f))
        if w > 2 * W or h > 2 * H:
            return None
        s = Image.fromarray(sprite[:, ::-1] if flip else sprite).resize((w, h), Image.Resampling.BILINEAR)
        s = np.asarray(s, np.float32) / 255.0
        x0, y0 = int(round(u * W - w / 2)), int(round(v * H - h / 2))
        X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if X1 <= X0 or Y1 <= Y0:
            return None
        s = s[Y0 - y0 : Y1 - y0, X0 - x0 : X1 - x0]
        rgb, alpha = s[..., :3], s[..., 3:4]
        lum = (
            rgb.reshape(-1, 3)[alpha.reshape(-1) > 0.5].mean(axis=1)
            if (alpha > 0.5).any()
            else rgb.mean(axis=2)
        )
        refl = (
            np.clip(rgb / max(1e-3, float(np.quantile(lum, 0.95))), 0.0, 1.5) * 0.6
        )  # albedo ~0.6 at its brightest
        T = math.exp(-c.c_per_m * dist)  # contrast lost along the line of sight
        radiance = refl * lit
        region = img[Y0:Y1, X0:X1]
        img[Y0:Y1, X0:X1] = region * (1 - alpha * T) + alpha * T * radiance  # fades into the water behind
        return (X0 / W, Y0 / H, X1 / W, Y1 / H)

    def _maybe_flash(self, i, species, dt, t):
        rate = self.cfg.flash_rate.get(species, 0.0)
        if rate and i not in self._flash_until and self.rng.random() < rate * dt:
            self._flash_until[i] = t + self.cfg.flash_s

    def _blob(self, img, yaw, pitch, dist):
        c = self.cfg
        H, W = c.hw
        u, v = self.cam.project(yaw, pitch)
        s = 0.008 + 0.01 / max(dist, 0.5)
        g = np.exp(-(((self._xs - u) / s) ** 2 + ((self._ys - v) / (s * W / H)) ** 2))
        img += (c.flash_radiance * math.exp(-c.c_per_m * dist) * g)[..., None] * FLASH_COLOUR
