"""Periodic linear probes during pretraining: is the *representation* getting better, not just
the loss falling?

Frame probe: logistic regression on mean-pooled patch tokens -> AUROC / AP / TPR@5%FPR (+ per slice).
Patch probe: logistic regression on individual patch tokens -> patch AUROC and the median
centroid error in degrees (heatmap soft-centroid vs. box centroid), i.e. the steering signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from talosaur.eval.linear import fit_logreg
from talosaur.eval.metrics import angular_error_deg, average_precision, roc_auc, soft_centroid, tpr_at_fpr

SLICES = ("clear", "murky", "dark", "other")


@dataclass
class ProbeData:
    images: torch.Tensor  # (N, 3, H, W) uint8
    frame_label: torch.Tensor  # (N,) -1/0/1
    coverage: torch.Tensor  # (N, h, w)
    patch_valid: torch.Tensor  # (N,) bool
    centroid: torch.Tensor  # (N, 3)
    slice: torch.Tensor  # (N,)

    @classmethod
    def from_items(cls, items: list[dict]) -> ProbeData:
        return cls(
            *(
                torch.stack([it[k] for it in items])
                for k in ("image", "frame_label", "coverage", "patch_valid", "centroid", "slice")
            )
        )

    def __len__(self) -> int:
        return self.images.shape[0]


class ProbeHook:
    def __init__(
        self,
        train: ProbeData,
        val: ProbeData,
        mean,
        std,
        batch_size: int = 128,
        pos_thr: float = 0.3,
        max_patches: int = 200_000,
        seed: int = 0,
    ):
        self.train, self.val = train, val
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.batch_size, self.pos_thr, self.max_patches, self.seed = batch_size, pos_thr, max_patches, seed

    # ------------------------------------------------------------------ constructors
    @classmethod
    def synthetic(cls, n_train: int, n_val: int, size, mean, std, patch: int = 16, **kw) -> ProbeHook:
        from talosaur.data.torch_datasets import synthetic_labeled_frames

        tr = ProbeData.from_items(synthetic_labeled_frames(n_train, size, patch, seed=1))
        va = ProbeData.from_items(synthetic_labeled_frames(n_val, size, patch, seed=2))
        return cls(tr, va, mean, std, **kw)

    @classmethod
    def from_index(
        cls,
        index_path,
        root,
        size,
        mean,
        std,
        n_train: int = 4000,
        n_val: int = 2000,
        patch: int = 16,
        workers: int = 8,
        seed: int = 0,
        **kw,
    ) -> ProbeHook:
        from torch.utils.data import DataLoader

        from talosaur.data.index import read_table
        from talosaur.data.torch_datasets import LabeledFrames

        df = read_table(index_path)
        labeled = df[
            (df["frame_label"] != -1)
            | df["boxes"].map(len).gt(0)
            | df["mask_path"].map(lambda v: isinstance(v, str) and len(v) > 0)
        ]
        out = []
        for split, n in (("train", n_train), ("val", n_val)):
            sub = labeled[labeled["split"] == split]
            if len(sub) > n:
                sub = sub.sample(n, random_state=seed)
            ds = LabeledFrames(sub, root, size, patch)
            items = [b for b in DataLoader(ds, batch_size=None, num_workers=workers)]
            out.append(ProbeData.from_items(items))
        return cls(out[0], out[1], mean, std, **kw)

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def _tokens(self, encoder, data: ProbeData, device, amp_dtype) -> torch.Tensor:
        was_training = encoder.training
        encoder.eval()
        outs = []
        for i in range(0, len(data), self.batch_size):
            x = data.images[i : i + self.batch_size].to(device, non_blocking=True).float() / 255.0
            x = (x - self.mean.to(device)) / self.std.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                t, _ = encoder.forward_grid(x)
            outs.append(t.float().cpu())
        encoder.train(was_training)
        return torch.cat(outs)  # (N, h, w, D)

    # ------------------------------------------------------------------ run
    def run(self, encoder, device, amp_dtype=None, prefix: str = "probe/") -> dict[str, float]:
        rng = np.random.default_rng(self.seed)
        ttr = self._tokens(encoder, self.train, device, amp_dtype)
        tva = self._tokens(encoder, self.val, device, amp_dtype)
        out: dict[str, float] = {}

        # frame probe
        ytr, yva = self.train.frame_label, self.val.frame_label
        mtr, mva = ytr >= 0, yva >= 0
        if mtr.sum() > 10 and len(torch.unique(ytr[mtr])) == 2 and mva.sum() > 0:
            lin = fit_logreg(ttr.mean(dim=(1, 2))[mtr].to(device), ytr[mtr].to(device), weight_decay=1e-3)
            with torch.no_grad():
                s = lin(tva.mean(dim=(1, 2))[mva].to(device)).squeeze(-1).cpu().numpy()
            y = yva[mva].numpy()
            out[f"{prefix}frame_auroc"] = roc_auc(y, s)
            out[f"{prefix}frame_ap"] = average_precision(y, s)
            out[f"{prefix}frame_tpr_at_5fpr"] = tpr_at_fpr(y, s, 0.05)
            sl = self.val.slice[mva].numpy()
            for k, name in enumerate(SLICES[:3]):
                m = sl == k
                if m.sum() > 5 and len(np.unique(y[m])) == 2:
                    out[f"{prefix}frame_auroc_{name}"] = roc_auc(y[m], s[m])

        # patch probe
        cov_tr = self.train.coverage
        pos = cov_tr >= self.pos_thr
        neg = (cov_tr <= 0.0) & self.train.patch_valid.view(-1, 1, 1)
        D = ttr.shape[-1]
        pi = torch.nonzero(pos.reshape(-1)).squeeze(1)
        ni = torch.nonzero(neg.reshape(-1)).squeeze(1)
        if len(pi) > 20 and len(ni) > 20:
            half = self.max_patches // 2
            pi = pi[torch.from_numpy(rng.permutation(len(pi))[:half])]
            ni = ni[torch.from_numpy(rng.permutation(len(ni))[:half])]
            flat = ttr.reshape(-1, D)
            X = torch.cat([flat[pi], flat[ni]]).to(device)
            y = torch.cat([torch.ones(len(pi)), torch.zeros(len(ni))]).to(device)
            lin = fit_logreg(X, y, weight_decay=1e-3)
            with torch.no_grad():
                logit = lin(tva.reshape(-1, D).to(device)).squeeze(-1).cpu().view(tva.shape[:3])
            prob = torch.sigmoid(logit).numpy()
            cv = self.val.coverage.numpy()
            valid = self.val.patch_valid.numpy()
            lab = np.where(cv >= self.pos_thr, 1, np.where(cv <= 0.0, 0, -1))
            lab[~valid] = np.where(lab[~valid] == 1, 1, -1)
            m = lab >= 0
            if m.any() and (lab[m] == 1).any() and (lab[m] == 0).any():
                out[f"{prefix}patch_auroc"] = roc_auc(lab[m], prob[m])
                out[f"{prefix}patch_ap"] = average_precision(lab[m], prob[m])
            errs = []
            for i in range(len(self.val)):
                c = self.val.centroid[i]
                if c[0] < 0:
                    continue
                pc = soft_centroid(prob[i])
                if pc is not None:
                    errs.append(angular_error_deg(pc, c.tolist()))
            if errs:
                out[f"{prefix}centroid_err_deg_median"] = float(np.median(errs))
                out[f"{prefix}centroid_found_frac"] = len(errs) / max(
                    1, int((self.val.centroid[:, 0] >= 0).sum())
                )
        return out
