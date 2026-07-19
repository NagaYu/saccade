"""Forward patch-embedding predictor (self-supervised, online).

Claim demonstrated: **省エネ** (a good prediction lets the gate trust the cache and
skip encoding) and **品質保持** (the predicted embedding *is* the reconstructed
value used for un-encoded patches, so its accuracy bounds the fidelity we keep).

A tiny GRU with weights shared across all patches maintains one hidden state per
patch. Each step it receives, for every patch, the motion-compensated cached
embedding plus the local + global ego-motion (pseudo-IMU) features, and predicts
that patch's embedding for the *next* frame. It is trained purely self-supervised:
whenever a patch is actually (re)encoded we observe its true embedding and take one
online SGD step on ||prediction - truth||. No labels, no offline dataset.

The prediction is intentionally a *residual* on top of the motion-warped cache
(head initialised near zero), so the network only has to learn the part of the
change that ego-motion does not already explain. That keeps it small and stable,
and makes learning visibly reduce error over a predictable sequence.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class GRUPatchPredictor(nn.Module):
    """Shared-weight GRU predicting next-frame patch embeddings from history + pseudo-IMU.

    Claim: 省エネ / 品質保持. Small enough (hidden=128) that its FLOPs live in the
    engine's fixed overhead, yet expressive enough to anticipate predictable motion
    so the gate can safely reuse tokens.
    """

    def __init__(self, embed_dim: int, n_patches: int, hidden: int = 128,
                 motion_dim: int = 6, lr: float = 5e-3, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed_dim = embed_dim
        self.n_patches = n_patches
        self.hidden = hidden
        # input per patch: warped cached embedding (D) + per-patch flow (2) + global ego-motion (motion_dim)
        self.in_dim = embed_dim + 2 + motion_dim
        self.cell = nn.GRUCell(self.in_dim, hidden)
        self.head = nn.Linear(hidden, embed_dim)
        nn.init.zeros_(self.head.weight)          # start as identity-on-warped-cache (residual = 0)
        nn.init.zeros_(self.head.bias)
        self.opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        self._h: Optional[torch.Tensor] = None    # (P, hidden) hidden state
        self._last_input: Optional[torch.Tensor] = None
        self._last_warped: Optional[torch.Tensor] = None
        self._last_h_prev: Optional[torch.Tensor] = None

    def reset(self):
        """Clear temporal state (new stream)."""
        self._h = None
        self._last_input = None
        self._last_warped = None
        self._last_h_prev = None

    def _build_input(self, warped: torch.Tensor, patch_flow: torch.Tensor,
                     global_feat: torch.Tensor) -> torch.Tensor:
        gfeat = global_feat.reshape(1, -1).expand(self.n_patches, -1)
        return torch.cat([warped, patch_flow, gfeat], dim=1)

    def predict(self, warped: torch.Tensor, patch_flow: np.ndarray,
                global_feat: np.ndarray) -> torch.Tensor:
        """Predict next-frame embeddings for all patches. Claim: 品質保持.

        ``warped`` is the (P, D) motion-compensated cache (from :mod:`saccade.kv_remap`).
        The prediction is ``warped + head(GRU(...))`` and is L2-normalised, matching
        the backbone's output convention so it can be substituted for un-encoded patches.
        Caches the input/warped tensors so the subsequent :meth:`observe` can train
        on exactly the state that produced this prediction.
        """
        flow = torch.as_tensor(patch_flow.reshape(self.n_patches, 2), dtype=torch.float32)
        gfeat = torch.as_tensor(global_feat, dtype=torch.float32)
        x = self._build_input(warped, flow, gfeat)
        h_prev = self._h if self._h is not None else torch.zeros(self.n_patches, self.hidden)
        with torch.no_grad():
            h = self.cell(x, h_prev)
            residual = self.head(h)
            pred = warped + residual
            pred = pred / (pred.norm(dim=-1, keepdim=True) + 1e-6)
        # Stash the state that PRODUCED this prediction so observe() can retrain on the
        # identical forward pass (h_prev is the hidden BEFORE this step, not after).
        self._last_h_prev = h_prev.detach()
        self._h = h
        self._last_input = x
        self._last_warped = warped.detach()
        return pred

    def observe(self, encoded_indices: torch.Tensor, true_embeddings: torch.Tensor) -> float:
        """Self-supervised online update on the patches we actually encoded.

        Claim: 省エネ / 品質保持. Trains the predictor with zero labels using the fresh
        embeddings the gate forced us to compute anyway, so prediction quality (and
        thus the amount we can safely skip next time) improves online. Returns the
        training loss on the observed subset. No-op if nothing was encoded.
        """
        if self._last_input is None or encoded_indices.numel() == 0:
            return 0.0
        idx = encoded_indices.long()
        x = self._last_input[idx]
        warped = self._last_warped[idx]
        h_prev = self._last_h_prev[idx]      # the pre-step hidden that produced the prediction
        h = self.cell(x, h_prev)
        pred = warped + self.head(h)
        pred = pred / (pred.norm(dim=-1, keepdim=True) + 1e-6)
        target = true_embeddings.detach()
        loss = ((pred - target) ** 2).sum(dim=-1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach())
