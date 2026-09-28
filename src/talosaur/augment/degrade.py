"""Underwater image degradation, batched on the GPU, with per-image random parameters.

Loosely follows the (revised) underwater image-formation model: the direct signal is
attenuated per colour channel with range, backscatter (veiling light) is added, and the
camera adds its own low-light behaviour. Everything runs in (approximately) linear light:

  I = J * exp(-beta * z) * L  +  B_inf * (1 - exp(-beta_B * z)) * A  +  particles
  then exposure, white-balance error, blur, Poisson-Gaussian sensor noise, 8-bit quantisation,
  and JPEG-like block compression.

* ``z``: a smooth random range field (not a depth map; we have none);
* ``beta``: per-channel attenuation from water-type presets (ocean / coastal / lake-CDOM);
* ``L``: optional artificial light (spot cones with fall-off; "deep" mode = no ambient light);
* ``A``: ambient level of the veiling light;
* particles: marine snow / backscatter specks, brighter where lit, optionally motion-streaked.

``variant="eval"`` uses different presets, particle shape and noise model so the synthetic
robustness sweep (Underwater-C) is not a copy of the training augmentation.

Modes for JEPA (see :func:`make_views`): ``none`` | ``shared`` | ``context_only`` | ``independent``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

OPS = (
    "attenuation",
    "backscatter",
    "light",
    "low_light",
    "white_balance",
    "particles",
    "blur",
    "noise",
    "compression",
)

# per-channel attenuation ranges [R, G, B] in 1/m and veiling-light colours (linear RGB)
WATER_TRAIN = {
    "ocean": {
        "beta_lo": (0.30, 0.04, 0.02),
        "beta_hi": (0.60, 0.12, 0.08),
        "veil": (0.02, 0.18, 0.35),
        "p": 0.35,
    },
    "coastal": {
        "beta_lo": (0.40, 0.10, 0.15),
        "beta_hi": (0.90, 0.30, 0.40),
        "veil": (0.04, 0.24, 0.22),
        "p": 0.35,
    },
    "lake": {
        "beta_lo": (0.50, 0.30, 0.80),
        "beta_hi": (1.20, 0.70, 2.00),
        "veil": (0.16, 0.20, 0.06),
        "p": 0.30,
    },
}
WATER_EVAL = {  # held-out presets for evaluation sweeps
    "open_ocean_deep": {
        "beta_lo": (0.45, 0.05, 0.03),
        "beta_hi": (0.55, 0.07, 0.04),
        "veil": (0.01, 0.10, 0.25),
        "p": 0.5,
    },
    "turbid_river": {
        "beta_lo": (0.9, 0.8, 1.5),
        "beta_hi": (1.4, 1.1, 2.5),
        "veil": (0.20, 0.17, 0.08),
        "p": 0.5,
    },
}


@dataclass
class DegradeConfig:
    p: float = 0.8  # probability that an image is degraded at all
    severity: tuple[float, float] = (0.2, 1.0)
    op_p: dict[str, float] = field(
        default_factory=lambda: {
            "attenuation": 0.9,
            "backscatter": 0.8,
            "light": 0.35,
            "low_light": 0.4,
            "white_balance": 0.3,
            "particles": 0.5,
            "blur": 0.4,
            "noise": 0.6,
            "compression": 0.3,
        }
    )
    skip_low_light_below: float = 0.08  # don't darken images that are already this dark
    variant: str = "train"  # train | eval
    ref_size: int = 224  # kernel/particle sizes are defined at this resolution


def _u(gen, n, lo, hi, device):
    return lo + (hi - lo) * torch.rand(n, generator=gen, device=device)


class UnderwaterDegradation(torch.nn.Module):
    def __init__(self, cfg: DegradeConfig | None = None):
        super().__init__()
        self.cfg = cfg or DegradeConfig()
        self.water = WATER_TRAIN if self.cfg.variant == "train" else WATER_EVAL

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
        ops: dict[str, bool] | None = None,
        severity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, 3, H, W) float in [0, 1] (sRGB). Returns the degraded batch (same shape/dtype).

        ``ops`` forces which operations are active (all images), ``severity`` (B,) fixes severity;
        both are for controlled evaluation sweeps. By default everything is sampled.
        """
        cfg = self.cfg
        B, _, H, W = x.shape
        dev = x.device
        gen = generator
        dtype = x.dtype
        x = x.float()
        s = severity.to(dev).float() if severity is not None else _u(gen, B, *cfg.severity, dev)
        apply = (
            torch.rand(B, generator=gen, device=dev) < cfg.p
            if ops is None
            else torch.ones(B, dtype=torch.bool, device=dev)
        )

        def active(name):
            if ops is not None:
                return torch.full((B,), bool(ops.get(name, False)), device=dev)
            return (torch.rand(B, generator=gen, device=dev) < cfg.op_p.get(name, 0.0)) & apply

        lin = x.clamp(0, 1) ** 2.2
        lum_in = (0.2126 * x[:, 0] + 0.7152 * x[:, 1] + 0.0722 * x[:, 2]).mean(dim=(1, 2))
        scale = max(H, W) / cfg.ref_size

        # --- range field z (metres), smooth + vertical gradient
        z0 = 0.5 + 7.5 * s * torch.rand(B, generator=gen, device=dev)
        low = torch.rand(B, 1, 4, 4, generator=gen, device=dev)
        field_ = F.interpolate(low, size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)
        yy = torch.linspace(-0.5, 0.5, H, device=dev).view(1, 1, H, 1)
        grad = _u(gen, B, -0.6, 0.6, dev).view(B, 1, 1, 1)
        z = z0.view(B, 1, 1, 1) * (1.0 + 0.6 * s.view(B, 1, 1, 1) * (field_ - 0.5) + grad * yy)
        z = z.clamp_min(0.1)

        # --- water type per image
        names = list(self.water)
        probs = torch.tensor([self.water[n]["p"] for n in names], device=dev)
        wt = torch.multinomial(probs, B, replacement=True, generator=gen)
        lo = torch.tensor([self.water[n]["beta_lo"] for n in names], device=dev)[wt]
        hi = torch.tensor([self.water[n]["beta_hi"] for n in names], device=dev)[wt]
        beta = lo + (hi - lo) * torch.rand(B, 3, generator=gen, device=dev)
        veil = torch.tensor([self.water[n]["veil"] for n in names], device=dev)[wt]
        veil = veil * _u(gen, B * 3, 0.7, 1.3, dev).view(B, 3)
        beta = beta.view(B, 3, 1, 1)
        veil = veil.view(B, 3, 1, 1)

        out = lin.clone()
        a_att = active("attenuation").view(B, 1, 1, 1)
        out = torch.where(a_att, out * torch.exp(-beta * z), out)

        # --- artificial light cones (optional) and ambient level
        a_light = active("light").view(B, 1, 1, 1)
        light = self._light_field(gen, B, H, W, s, dev)
        deep = (torch.rand(B, generator=gen, device=dev) < 0.5).view(B, 1, 1, 1) & a_light
        ambient = torch.where(
            deep, _u(gen, B, 0.0, 0.08, dev).view(B, 1, 1, 1), _u(gen, B, 0.4, 1.0, dev).view(B, 1, 1, 1)
        )
        out = torch.where(a_light, out * light, out)

        a_bs = active("backscatter").view(B, 1, 1, 1)
        beta_b = beta * _u(gen, B * 3, 0.8, 1.2, dev).view(B, 3, 1, 1)
        veil_term = (
            veil * (1.0 - torch.exp(-beta_b * z)) * torch.where(a_light, 0.35 * light + ambient, ambient)
        )
        out = torch.where(a_bs, out + veil_term * (0.6 + 0.8 * s.view(B, 1, 1, 1)), out)

        # --- particles / marine snow (lit by the lamps when present)
        a_part = active("particles").view(B, 1, 1, 1)
        if a_part.any():
            parts = self._particles(gen, B, H, W, s, scale, dev)
            illum = torch.where(a_light, light, torch.ones_like(light) * (0.4 + 0.6 * ambient))
            out = torch.where(
                a_part, out + parts * illum * torch.tensor([0.9, 1.0, 1.0], device=dev).view(1, 3, 1, 1), out
            )

        # --- exposure (low light) and white-balance error
        a_ll = (active("low_light") & (lum_in > cfg.skip_low_light_below)).view(B, 1, 1, 1)
        expo = torch.exp(_u(gen, B, math.log(0.03), math.log(0.5), dev))
        expo = 1.0 - s * (1.0 - expo)
        out = torch.where(a_ll, out * expo.view(B, 1, 1, 1), out)
        a_wb = active("white_balance").view(B, 1, 1, 1)
        gains = torch.exp(0.15 * s.view(B, 1) * torch.randn(B, 3, generator=gen, device=dev)).view(B, 3, 1, 1)
        out = torch.where(a_wb, out * gains, out)

        # --- blur (defocus / motion / forward scatter), per-image kernels
        a_blur = active("blur")
        if a_blur.any():
            out = torch.where(a_blur.view(B, 1, 1, 1), self._blur(out, gen, s, scale), out)

        # --- sensor noise (Poisson-Gaussian), fewer photons when darker. After demosaicing and
        # ISP denoising real cameras show mostly luminance noise, so chroma noise is weaker.
        a_noise = active("noise").view(B, 1, 1, 1)
        photons = torch.exp(_u(gen, B, math.log(150.0), math.log(3000.0), dev)) * (1.3 - s)
        photons = photons.view(B, 1, 1, 1)
        read = (0.002 + 0.010 * s * torch.rand(B, generator=gen, device=dev)).view(B, 1, 1, 1)
        sig = out.clamp_min(0)
        shape1 = (B, 1, H, W)
        if self.cfg.variant == "train":
            n_lum = torch.randn(shape1, generator=gen, device=dev)
            n_chr = torch.randn(sig.shape, generator=gen, device=dev)
            shot = torch.sqrt(sig / photons) * (0.85 * n_lum + 0.45 * n_chr)
        else:  # eval variant: plain Gaussian noise, fully per-channel
            shot = (0.5 * read + 0.02 * s.view(B, 1, 1, 1)) * torch.randn(
                sig.shape, generator=gen, device=dev
            )
        noisy = sig + shot + read * torch.randn(shape1, generator=gen, device=dev)
        out = torch.where(a_noise, noisy, out)

        # --- back to sRGB, 8-bit, optional JPEG-like compression
        y = out.clamp(0, 1) ** (1 / 2.2)
        y = torch.round(y * 255.0) / 255.0
        a_jpg = active("compression")
        if a_jpg.any():
            q = _u(gen, B, 15.0, 70.0, dev) * (1.3 - 0.6 * s)
            y = torch.where(a_jpg.view(B, 1, 1, 1), jpeg_like(y, q.clamp(5, 95)), y)
        y = torch.where(apply.view(B, 1, 1, 1), y, x)
        return y.to(dtype)

    # ------------------------------------------------------------------ parts
    def _light_field(self, gen, B, H, W, s, dev):
        yy = torch.linspace(0, 1, H, device=dev).view(1, H, 1)
        xx = torch.linspace(0, 1, W, device=dev).view(1, 1, W)
        L = torch.zeros(B, H, W, device=dev)
        n_spots = 1 + (torch.rand(B, generator=gen, device=dev) < 0.4).long()
        for k in range(2):
            cx = _u(gen, B, 0.25, 0.75, dev).view(B, 1, 1)
            cy = _u(gen, B, 0.15, 0.65, dev).view(B, 1, 1)
            r = _u(gen, B, 0.25, 0.7, dev).view(B, 1, 1)
            spot = torch.exp(-(((xx - cx) ** 2 + ((yy - cy) * H / W) ** 2) / (r**2)))
            L = L + spot * (k < n_spots).view(B, 1, 1).float()
        floor = (0.02 + 0.25 * (1 - s)).view(B, 1, 1)
        L = floor + (1 - floor) * L.clamp(0, 1)
        return L.unsqueeze(1)

    def _particles(self, gen, B, H, W, s, scale, dev):
        density = (0.0003 + 0.004 * s * torch.rand(B, generator=gen, device=dev)) / max(scale, 1e-6) ** 2
        nmax = int(min(4000, max(8, float(density.max()) * H * W)))
        keep = torch.rand(B, nmax, generator=gen, device=dev) < (density * H * W / nmax).view(B, 1)
        ys = torch.randint(0, H, (B, nmax), generator=gen, device=dev)
        xs = torch.randint(0, W, (B, nmax), generator=gen, device=dev)
        amp = _u(gen, B * nmax, 0.2, 1.0, dev).view(B, nmax) ** 2 * keep
        pts = torch.zeros(B, H * W, device=dev)
        pts.scatter_add_(1, ys * W + xs, amp)
        pts = pts.view(B, 1, H, W)
        if self.cfg.variant == "train":
            sigma = _u(gen, B, 0.5, 2.0, dev) * scale
            elong = 1.0 + 3.0 * (torch.rand(B, generator=gen, device=dev) < 0.3).float() * torch.rand(
                B, generator=gen, device=dev
            )
            ang = _u(gen, B, 0.0, math.pi, dev)
            k = gaussian_kernels(sigma, elong, ang, size=_odd(int(8 * scale) + 1))
            return grouped_conv(pts, k) * 6.0
        # eval variant: hard defocus disks with brighter rims ("bokeh")
        radius = _u(gen, B, 1.0, 3.0, dev) * scale
        k = disk_kernels(radius, size=_odd(int(8 * scale) + 1), rim=0.5)
        return grouped_conv(pts, k) * 4.0

    def _blur(self, x, gen, s, scale):
        B = x.shape[0]
        dev = x.device
        size = _odd(int(11 * scale))
        kind = torch.randint(0, 3, (B,), generator=gen, device=dev)
        sig = (0.4 + 2.0 * s * torch.rand(B, generator=gen, device=dev)) * scale
        k_gauss = gaussian_kernels(sig, torch.ones(B, device=dev), torch.zeros(B, device=dev), size)
        k_disk = disk_kernels((0.5 + 3.0 * s * torch.rand(B, generator=gen, device=dev)) * scale, size)
        length = (1.0 + 7.0 * s * torch.rand(B, generator=gen, device=dev)) * scale
        k_motion = gaussian_kernels(
            length / 2.5, torch.full((B,), 6.0, device=dev), _u(gen, B, 0, math.pi, dev), size
        )
        k = torch.where(
            (kind == 0).view(B, 1, 1, 1), k_gauss, torch.where((kind == 1).view(B, 1, 1, 1), k_disk, k_motion)
        )
        return grouped_conv(x, k)


# ---------------------------------------------------------------------- kernels / helpers


def _odd(n: int) -> int:
    n = max(3, n)
    return n if n % 2 else n + 1


def gaussian_kernels(
    sigma: torch.Tensor, elong: torch.Tensor, angle: torch.Tensor, size: int
) -> torch.Tensor:
    """(B, 1, size, size) normalised anisotropic Gaussians (elong >= 1 stretches along ``angle``)."""
    B = sigma.shape[0]
    r = torch.arange(size, device=sigma.device, dtype=torch.float32) - size // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    c, s_ = torch.cos(angle).view(B, 1, 1), torch.sin(angle).view(B, 1, 1)
    u = xx * c + yy * s_
    v = -xx * s_ + yy * c
    su = (sigma * elong).view(B, 1, 1).clamp_min(0.3)
    sv = sigma.view(B, 1, 1).clamp_min(0.3)
    k = torch.exp(-0.5 * ((u / su) ** 2 + (v / sv) ** 2))
    return (k / k.sum(dim=(1, 2), keepdim=True)).unsqueeze(1)


def disk_kernels(radius: torch.Tensor, size: int, rim: float = 0.0) -> torch.Tensor:
    B = radius.shape[0]
    r = torch.arange(size, device=radius.device, dtype=torch.float32) - size // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    d = torch.sqrt(xx**2 + yy**2).unsqueeze(0)
    rad = radius.view(B, 1, 1).clamp_min(0.5)
    k = torch.sigmoid((rad - d) * 3.0)
    if rim:
        k = k * (1.0 + rim * torch.exp(-((d - rad) ** 2) / 0.5))
    return (k / k.sum(dim=(1, 2), keepdim=True)).unsqueeze(1)


def grouped_conv(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Convolve each image (all channels) with its own kernel. x (B, C, H, W), k (B, 1, s, s)."""
    B, C, H, W = x.shape
    ks = k.shape[-1]
    weight = k.repeat_interleave(C, dim=0)  # (B*C, 1, s, s)
    y = F.conv2d(F.pad(x.reshape(1, B * C, H, W), (ks // 2,) * 4, mode="reflect"), weight, groups=B * C)
    return y.view(B, C, H, W)


_JPEG_Y = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)
_JPEG_C = torch.full((8, 8), 99.0)
_JPEG_C[:4, :4] = torch.tensor(
    [[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]], dtype=torch.float32
)


def _dct8(device) -> torch.Tensor:
    k = torch.arange(8, device=device, dtype=torch.float32).view(8, 1)
    n = torch.arange(8, device=device, dtype=torch.float32).view(1, 8)
    m = torch.cos(math.pi * (2 * n + 1) * k / 16) * math.sqrt(2 / 8)
    m[0] /= math.sqrt(2)
    return m


def jpeg_like(x: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """8x8 DCT quantisation in YCbCr with standard tables scaled by per-image ``quality`` (1..100),
    plus 2x chroma subsampling. x (B, 3, H, W) in [0, 1]."""
    B, _, H, W = x.shape
    dev = x.device
    ph, pw = (-H) % 8, (-W) % 8
    xp = F.pad(x, (0, pw, 0, ph), mode="replicate") * 255.0
    r, g, b = xp[:, 0], xp[:, 1], xp[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    # chroma subsampling
    cb = F.interpolate(F.avg_pool2d(cb.unsqueeze(1), 2), scale_factor=2, mode="nearest").squeeze(1)
    cr = F.interpolate(F.avg_pool2d(cr.unsqueeze(1), 2), scale_factor=2, mode="nearest").squeeze(1)
    q = quality.view(B, 1, 1, 1, 1)
    scale = torch.where(q < 50, 5000.0 / q, 200.0 - 2.0 * q) / 100.0
    D = _dct8(dev)
    outs = []
    for ch, table in ((y, _JPEG_Y), (cb, _JPEG_C), (cr, _JPEG_C)):
        Hp, Wp = ch.shape[-2:]
        blocks = ch.view(B, Hp // 8, 8, Wp // 8, 8).permute(0, 1, 3, 2, 4) - 128.0
        coef = D @ blocks @ D.T
        qt = (table.to(dev).view(1, 1, 1, 8, 8) * scale).clamp_min(1.0)
        coef = torch.round(coef / qt) * qt
        rec = D.T @ coef @ D + 128.0
        outs.append(rec.permute(0, 1, 3, 2, 4).reshape(B, Hp, Wp))
    y, cb, cr = outs
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    out = torch.stack([r, g, b], dim=1)[:, :, :H, :W] / 255.0
    return out.clamp(0, 1)


def make_views(
    x: torch.Tensor,
    mode: str,
    degrade: UnderwaterDegradation | None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(context input, target input) for the four JEPA degradation modes."""
    if mode == "none" or degrade is None:
        return x, x
    if mode == "shared":
        d = degrade(x, generator)
        return d, d
    if mode == "context_only":
        return degrade(x, generator), x
    if mode == "independent":
        return degrade(x, generator), degrade(x, generator)
    raise ValueError(f"unknown degrade mode {mode!r}")
