"""The simulated water column: animals in horizontal patches and vertical layers.

Coordinates: x east, y north (metres, a square domain that wraps around), z depth (metres,
positive down). Migrating species follow a layer whose depth changes with the time of day (diel
vertical migration); the others keep their own depth range.

Scale matters: a camera sees a few cubic metres, so realistic encounter rates need realistic
densities (``layer_density`` animals per cubic metre in the layer). To keep millions of animals
cheap, positions are analytic (start + drift x t) and only animals near the vehicle are simulated
in detail (reactions, fleeing). They are found with a two-level spatial index: every ``local_s`` the
animals within a box around the vehicle, and every ``index_s`` a 3-D grid of just those.
Animals in a patch (a school, a swarm) move together, so patches persist while the vehicle
searches them. All numbers are placeholders to be replaced by the literature (docs/SEARCH.md) and
your own dives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Species:
    name: str
    share: float  # fraction of the animals
    size_m: float  # body (or colony) length
    speed_mps: float  # cruising speed
    avoid_prob: float  # chance of fleeing once the vehicle is within avoid_range_m
    avoid_range_m: float
    flee_mps: float
    migrates: bool  # follows the migrating layer
    depth_m: tuple[float, float] = (150.0, 60.0)  # non-migrants: mean, sd
    light_avoid_m: float = 0.0  # extra avoidance range with the vehicle's lights fully on


def default_species() -> list[Species]:
    return [
        Species("lanternfish", 0.40, 0.06, 0.10, 0.5, 2.0, 0.6, True, light_avoid_m=20.0),
        Species("shrimp", 0.25, 0.04, 0.05, 0.6, 1.5, 0.3, True, light_avoid_m=15.0),
        Species("medusa", 0.15, 0.10, 0.02, 0.0, 0.0, 0.0, False, (150.0, 80.0)),
        Species("siphonophore", 0.10, 0.50, 0.01, 0.0, 0.0, 0.0, False, (200.0, 100.0)),
        Species("squid", 0.10, 0.20, 0.30, 0.7, 3.0, 1.0, True, light_avoid_m=20.0),
    ]


@dataclass
class WorldConfig:
    size_m: float = 1000.0
    layer_density: float = 0.002  # animals per m^3 averaged over the layer (patches included)
    patch_share: float = 0.5  # fraction of animals in patches
    patch_radius_m: float = 8.0
    animals_per_patch: float = 40.0
    day_layer_m: float = 500.0  # migrating layer centre by day ...
    night_layer_m: float = 70.0  # ... and by night
    layer_sd_m: float = 25.0
    sunrise_h: float = 6.5
    sunset_h: float = 18.5
    migration_h: float = 1.5  # hours the layer takes to move between its day and night depths
    start_hour: float = 20.0  # local time at t = 0
    token_dim: int = 16
    same_species_sim: float = 0.9  # typical appearance similarity of two animals of one species
    patch_jitter: float = 0.1  # animals in a patch: the patch's velocity + this fraction of their own
    index_s: float = 5.0  # 3-D grid index of the animals near the vehicle: rebuild period ...
    cell_m: float = 10.0  # ... and cell size
    local_s: float = 30.0  # which animals are near the vehicle: chosen again this often
    species: list[Species] = field(default_factory=default_species)
    seed: int = 0


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class World:
    def __init__(self, cfg: WorldConfig | None = None):
        self.cfg = c = cfg or WorldConfig()
        rng = self.rng = np.random.default_rng(c.seed)
        L = c.size_m
        n = int(c.layer_density * L * L * (4 * c.layer_sd_m))  # the layer holds ~4 sd of depth
        n_patch = int(n * c.patch_share)
        n_centers = max(1, int(n_patch / c.animals_per_patch))
        centers = rng.uniform(0, L, size=(n_centers, 2))
        which = rng.integers(n_centers, size=n_patch)
        xy_p = centers[which] + rng.normal(0, c.patch_radius_m / 2, size=(n_patch, 2))
        xy_b = rng.uniform(0, L, size=(n - n_patch, 2))
        self.xy0 = np.concatenate([xy_p, xy_b]) % L
        self.patch = np.concatenate([which, np.full(n - n_patch, -1)])
        shares = np.array([s.share for s in c.species], float)
        # animals in one patch are mostly one species (schools, swarms)
        patch_sp = rng.choice(len(c.species), size=n_centers, p=shares / shares.sum())
        sp_b = rng.choice(len(c.species), size=n - n_patch, p=shares / shares.sum())
        self.sp = np.concatenate([patch_sp[which], sp_b])
        spec = c.species
        self.migrant = np.array([spec[s].migrates for s in self.sp])
        self.dz = rng.normal(0, c.layer_sd_m, size=n)
        z_mean = np.array([s.depth_m[0] for s in spec])[self.sp]
        z_sd = np.array([s.depth_m[1] for s in spec])[self.sp]
        self.z_fixed = np.clip(rng.normal(z_mean, z_sd), 2.0, None)
        cruise = np.array([s.speed_mps for s in spec])
        speed = cruise[self.sp] * rng.uniform(0.2, 1.0, size=n)
        ang = rng.uniform(0, 2 * math.pi, size=n)
        self.v = np.stack([np.sin(ang), np.cos(ang)], axis=1) * speed[:, None]
        # a school or swarm moves together: the patch's velocity plus a little of each animal's own
        p_speed = cruise[patch_sp] * rng.uniform(0.2, 1.0, size=n_centers)
        p_ang = rng.uniform(0, 2 * math.pi, size=n_centers)
        p_v = np.stack([np.sin(p_ang), np.cos(p_ang)], axis=1) * p_speed[:, None]
        self.v[:n_patch] = p_v[which] + c.patch_jitter * self.v[:n_patch]
        self.size = np.array([spec[s].size_m for s in self.sp])
        self.avoid_prob = np.array([spec[s].avoid_prob for s in self.sp])
        self.avoid_range = np.array([spec[s].avoid_range_m for s in self.sp])
        self.light_avoid = np.array([spec[s].light_avoid_m for s in self.sp])
        self.flee_mps = np.array([spec[s].flee_mps for s in self.sp])
        # appearance: a unit prototype per species plus individual variation
        d = c.token_dim
        protos = rng.normal(size=(len(spec), d))
        protos /= np.linalg.norm(protos, axis=1, keepdims=True)
        spread = math.sqrt(max(1e-6, 1.0 / c.same_species_sim - 1.0) / d)
        app = protos[self.sp] + rng.normal(0, spread, size=(n, d))
        self.appearance = app / np.linalg.norm(app, axis=1, keepdims=True)
        # sparse state for animals that met the vehicle
        self.offset = np.zeros((n, 3))  # displacement from fleeing
        self.flee_until = np.full(n, -1.0)
        self.flee_dir = np.zeros((n, 3))
        self.reacted = np.zeros(n, bool)
        self._fleeing = np.zeros(0, np.int64)
        self.t = 0.0
        # nothing moves faster than this (drift + fleeing + the migrating layer): index margins
        migrate = abs(c.day_layer_m - c.night_layer_m) / max(c.migration_h * 3600.0, 1.0) * 1.5
        drift = float(np.linalg.norm(self.v, axis=1).max()) if n else 0.0
        self.max_speed = drift + float(self.flee_mps.max(initial=0.0)) + migrate
        self._local = np.zeros(0, np.int64)
        self._local_p: np.ndarray | None = None
        self._local_r = 0.0
        self._local_half = 0.0
        self._cells: dict[int, np.ndarray] = {}
        self.invalidate_index()

    @property
    def n(self) -> int:
        return len(self.xy0)

    def hour(self, t: float | None = None) -> float:
        return (self.cfg.start_hour + (self.t if t is None else t) / 3600.0) % 24.0

    def layer_depth(self, t: float | None = None) -> float:
        """Centre of the migrating layer: deep by day, shallow by night, moving at dusk and dawn."""
        c, h = self.cfg, self.hour(t)
        m = c.migration_h
        if c.sunrise_h - m / 2 <= h < c.sunrise_h + m / 2:  # dawn: descending
            f = 1.0 - _smoothstep((h - (c.sunrise_h - m / 2)) / m)
        elif c.sunset_h - m / 2 <= h < c.sunset_h + m / 2:  # dusk: ascending
            f = _smoothstep((h - (c.sunset_h - m / 2)) / m)
        else:
            f = 0.0 if c.sunrise_h + m / 2 <= h < c.sunset_h - m / 2 else 1.0  # 1 = night
        return c.day_layer_m + (c.night_layer_m - c.day_layer_m) * f

    def positions(self, idx: np.ndarray) -> np.ndarray:
        """Current positions of the animals ``idx`` (analytic drift + fleeing offsets)."""
        L = self.cfg.size_m
        p = np.empty((len(idx), 3))
        p[:, :2] = (self.xy0[idx] + self.v[idx] * self.t + self.offset[idx, :2]) % L
        z = np.where(self.migrant[idx], self.layer_depth() + self.dz[idx], self.z_fixed[idx])
        p[:, 2] = np.clip(z + self.offset[idx, 2], 1.0, None)
        return p

    _NZ = 1 << 20  # depth cells per column in the index key

    def invalidate_index(self) -> None:
        """Force a full rebuild at the next query (after moving animals by hand, e.g. in tests)."""
        self._index_t = self._local_t = -math.inf

    def _wrap(self, d: np.ndarray) -> np.ndarray:
        L = self.cfg.size_m
        return (d + L / 2) % L - L / 2

    def _refresh(self, p: np.ndarray, radius: float) -> None:
        c = self.cfg
        L = c.size_m
        moved = math.inf if self._local_p is None else float(np.abs(self._wrap(p[:2] - self._local_p)).max())
        if self.t - self._local_t >= c.local_s or moved > self._local_half / 3 or radius > self._local_r:
            # every animal that can come within `radius` of a query point up to a third of the box
            # away, before the next choice (all animals: O(n), but only every local_s)
            self._local_r = max(radius, self._local_r)
            self._local_half = 1.5 * (self._local_r + self.max_speed * (c.local_s + c.index_s))
            d = self.v * self.t  # in place from here: this touches every animal
            d += self.xy0
            d += self.offset[:, :2]
            d -= p[:2] - L / 2
            np.mod(d, L, out=d)
            d -= L / 2
            np.abs(d, out=d)
            self._local = np.nonzero(d.max(axis=1) < self._local_half)[0]
            self._local_t, self._local_p = self.t, np.array(p[:2], float)
            self._index_t = -math.inf
        if self.t - self._index_t >= c.index_s:  # 3-D grid of the local animals
            idx = self._local
            cells = np.floor(self.positions(idx) / c.cell_m).astype(np.int64)
            n_xy = max(1, int(round(L / c.cell_m)))
            key = ((cells[:, 0] % n_xy) * n_xy + cells[:, 1] % n_xy) * self._NZ + np.clip(
                cells[:, 2], 0, self._NZ - 1
            )
            order = np.argsort(key, kind="stable")
            keys, starts = np.unique(key[order], return_index=True)
            ends = np.r_[starts[1:], len(order)]
            self._cells = {int(k): idx[order[s:e]] for k, s, e in zip(keys, starts, ends)}
            self._index_t = self.t

    def near(self, p: np.ndarray, radius: float) -> np.ndarray:
        """Indices of animals possibly within ``radius`` of ``p`` (a superset; check distances)."""
        c = self.cfg
        self._refresh(np.asarray(p, float), radius)
        margin = radius + self.max_speed * (self.t - self._index_t)
        n_xy = max(1, int(round(c.size_m / c.cell_m)))
        cx, cy, cz = (int(math.floor(v / c.cell_m)) for v in p)
        k = int(math.ceil(margin / c.cell_m))
        out = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                base = (((cx + i) % n_xy) * n_xy + (cy + j) % n_xy) * self._NZ
                for m in range(max(0, cz - k), cz + k + 1):
                    o = self._cells.get(base + m)
                    if o is not None:
                        out.append(o)
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def offsets(self, idx: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Positions of ``idx`` relative to ``p`` (the shortest way round the wrapped domain)."""
        d = self.positions(idx) - p
        L = self.cfg.size_m
        d[:, :2] = (d[:, :2] + L / 2) % L - L / 2
        return d

    def step(self, dt: float, vehicle_pos: np.ndarray | None = None, light: float = 0.0) -> None:
        self.t += dt
        if len(self._fleeing):
            self._fleeing = self._fleeing[self.flee_until[self._fleeing] > self.t]
            f = self._fleeing
            self.offset[f] += self.flee_dir[f] * self.flee_mps[f, None] * dt
        if vehicle_pos is None:
            return
        idx = self.near(vehicle_pos, 3.5 + float(self.light_avoid.max()) * light)
        if not len(idx):
            return
        d = self.offsets(idx, vehicle_pos)
        dist = np.linalg.norm(d, axis=1)
        rng_avoid = self.avoid_range[idx] + self.light_avoid[idx] * light
        close = (dist < rng_avoid) & ~self.reacted[idx]
        react = close & (
            self.rng.random(len(idx)) < np.maximum(self.avoid_prob[idx], (self.light_avoid[idx] > 0) * light)
        )
        self.reacted[idx[close]] = True
        self.reacted[idx[dist > 2 * np.maximum(rng_avoid, 0.5)]] = False  # decides again next approach
        r = idx[react]
        self.flee_until[r] = self.t + 5.0
        self.flee_dir[r] = d[react] / np.maximum(dist[react, None], 1e-6)
        if len(r):
            self._fleeing = np.union1d(self._fleeing, r)
