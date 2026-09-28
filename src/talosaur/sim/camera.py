"""A simulated camera + model: animals in view -> the outputs the onboard model would give.

Each animal inside the field of view and the visibility range is projected with the guidance
camera model (so flat-port refraction is included). Its detection confidence falls with range and
with apparent size (small, distant animals are missed), and it is painted into the patch heatmap
and patch tokens (its appearance vector). A few false specks (marine snow) appear at random.

Light: visibility grows with the vehicle's light level (``visibility_dark_m`` -> ``visibility_m``);
the world makes light-shy species flee further when the lights are on - the trade-off the light
policy has to make.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from talosaur.guidance.camera_model import CameraModel


@dataclass
class SimCameraConfig:
    grid: tuple[int, int] = (7, 13)  # 112x208 input
    visibility_m: float = 5.0  # nothing beyond this with the lights fully on ...
    visibility_dark_m: float = 2.0  # ... or with them off (ambient light / bioluminescence only)
    size50_deg: float = 1.5  # detection probability 0.5 at this apparent size
    speck_rate: float = 0.02  # false specks per frame
    token_noise: float = 0.05
    heat_noise: float = 0.3  # logit noise


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class SimCamera:
    def __init__(self, cfg: SimCameraConfig | None = None, camera: CameraModel | None = None, seed: int = 0):
        self.cfg = cfg or SimCameraConfig()
        self.cam = camera or CameraModel()
        self.hfov, self.vfov = self.cam.effective_fov()
        self.rng = np.random.default_rng(seed)
        gh, gw = self.cfg.grid
        self._ys, self._xs = np.mgrid[0:gh, 0:gw]

    def render(self, vehicle, world, light: float = 0.0):
        """-> (frame_logit (1,), heat_logit (h, w), embedding (D,), tokens (h, w, D), visible)
        where ``visible`` lists (animal index, u, v, distance, confidence)."""
        c, rng = self.cfg, self.rng
        gh, gw = c.grid
        vis = c.visibility_dark_m + (c.visibility_m - c.visibility_dark_m) * float(np.clip(light, 0, 1))
        range50 = 0.6 * vis
        idx = world.near(vehicle.pos, vis)
        D = world.appearance.shape[1]
        water = np.zeros(D)
        water[0] = 1.0  # the "empty water" token direction
        prob = np.full((gh, gw), 0.02)
        tokens = np.tile(water, (gh, gw, 1)) + rng.normal(0, c.token_noise, size=(gh, gw, D))
        visible = []
        if len(idx):
            d = world.offsets(idx, vehicle.pos)
            dist = np.linalg.norm(d, axis=1)
            keep = dist < vis
            f, r = vehicle.forward(), vehicle.right()
            for i, di, dd in zip(idx[keep], d[keep], dist[keep]):
                fwd, rgt, up = float(di @ f), float(di @ r), -float(di[2])
                if fwd <= 0.05:
                    continue
                yaw = math.degrees(math.atan2(rgt, fwd))
                pitch = math.degrees(math.atan2(up, math.hypot(fwd, rgt)))
                if abs(yaw) > self.hfov / 2 or abs(pitch) > self.vfov / 2:
                    continue
                u, v = self.cam.project(yaw, pitch)
                ang = math.degrees(2 * math.atan(world.size[i] / (2 * dd)))
                conf = 1 / (1 + math.exp((dd - range50) / 0.4)) / (1 + math.exp((c.size50_deg - ang) / 0.4))
                if rng.random() > conf:  # missed this frame
                    continue
                visible.append((int(i), u, v, float(dd), float(conf)))
                radius = max(0.5, 0.5 * ang / self.hfov * gw)  # in cells
                cells = ((self._xs + 0.5 - u * gw) ** 2 + (self._ys + 0.5 - v * gh) ** 2) <= radius**2
                if not cells.any():
                    cells[min(gh - 1, max(0, int(v * gh))), min(gw - 1, max(0, int(u * gw)))] = True
                prob[cells] = np.maximum(prob[cells], 0.95 * conf)
                tokens[cells] = (
                    world.appearance[i] * 2.0
                    + water
                    + rng.normal(0, c.token_noise, size=(int(cells.sum()), D))
                )
        if rng.random() < c.speck_rate:  # marine snow
            prob[rng.integers(gh), rng.integers(gw)] = 0.7
        heat = _logit(prob) + rng.normal(0, c.heat_noise, size=prob.shape)
        frame = np.array([_logit(max([v[4] for v in visible], default=0.02))], np.float32)
        return frame, heat.astype(np.float32), tokens.mean(axis=(0, 1)), tokens.astype(np.float32), visible
