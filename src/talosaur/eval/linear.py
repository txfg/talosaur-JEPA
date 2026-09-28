"""Linear (logistic-regression) probes in torch, full-batch L-BFGS on CPU or GPU.

The fitted probe is an ``nn.Linear`` with input standardisation folded in, so it can be exported
directly as a deployable head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProbeResult:
    linear: nn.Linear  # standardisation folded in: logits = linear(raw_features)
    weight_decay: float
    val_loss: float


def fit_logreg(
    X: torch.Tensor,
    y: torch.Tensor,
    weight_decay: float = 1e-4,
    max_iter: int = 100,
    balanced: bool = True,
    sample_weight: torch.Tensor | None = None,
) -> nn.Linear:
    """Binary logistic regression. X (N, D) float, y (N,) in {0, 1}. Returns nn.Linear(D, 1)."""
    X = X.float()
    y = y.float().to(X.device)
    mu = X.mean(0)
    sd = X.std(0).clamp_min(1e-6)
    Z = (X - mu) / sd
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    if sample_weight is None:
        sample_weight = torch.ones_like(y)
    if balanced:
        pos = y.sum().clamp_min(1)
        neg = (1 - y).sum().clamp_min(1)
        sample_weight = sample_weight * torch.where(y > 0.5, 0.5 * len(y) / pos, 0.5 * len(y) / neg)
    sample_weight = sample_weight / sample_weight.mean()
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = Z @ w + b
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * sample_weight).mean()
        loss = loss + 0.5 * weight_decay * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    lin = nn.Linear(X.shape[1], 1)
    with torch.no_grad():
        lin.weight.copy_((w / sd).unsqueeze(0))
        lin.bias.copy_(b - (w * mu / sd).sum())
    return lin.to(X.device)


def fit_logreg_cv(Xtr, ytr, Xval, yval, grid=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1), **kw) -> ProbeResult:
    """Pick weight decay on a validation set by log-loss."""
    best = None
    for wd in grid:
        lin = fit_logreg(Xtr, ytr, weight_decay=wd, **kw)
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(
                lin(Xval.float()).squeeze(-1), yval.float().to(Xval.device)
            ).item()
        if best is None or vl < best.val_loss:
            best = ProbeResult(lin, wd, vl)
    return best
