"""Surprisal gate: decide which patches are worth re-encoding.

Claim demonstrated: **省エネ** (encode only the surprising minority) with a
**品質保持** safety valve (task-aware correction protects output-critical patches).

Chicken-and-egg note: the literal quantity we care about is the *embedding*
surprisal ``||actual_embedding - predicted_embedding||`` — but the actual embedding
only exists after we pay to encode the patch. So the operational decision uses a
**cheap pixel-space proxy**: the motion-compensated appearance residual (how much a
patch changed *after* removing the ego-motion that :mod:`saccade.kv_remap` explains).
This proxy is a lower bound on genuine change and needs no encoder. Where we *do*
encode, the engine still computes the true embedding surprisal and uses it to (a)
train the predictor and (b) update per-patch importance — closing the loop.

This is exactly why temporal-similarity fails on steady motion: it compares
same-position pixels, so a smooth camera pan makes every patch "change" and nothing
can be skipped. The Saccade gate compares *motion-compensated* pixels, so a static
world under ego-motion produces a near-zero residual and is skipped wholesale.

Task-aware version: an important patch (recently high surprisal — our stand-in for
"recent attention x error", since we have no LLM attention on CPU) gets a *lower*
effective threshold, so change there is caught earlier. ``τ_eff = τ / (1 + λ·w)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


def patch_residual(gray_now: np.ndarray, gray_ref: np.ndarray, grid: Tuple[int, int]) -> np.ndarray:
    """Per-patch mean-abs pixel difference in [0, 1] between two frames.

    Claim: 省エネ. The single cheap signal driving the encode decision. Passing a
    *motion-compensated* ``gray_ref`` (Saccade) vs a *same-position* previous frame
    (TemporalSim) is the only change that separates the two conditions here — which
    is what makes the benchmark's central comparison fair and interpretable.
    """
    gh, gw = grid
    h, w = gray_now.shape
    diff = np.abs(gray_now.astype(np.float32) - gray_ref.astype(np.float32)) / 255.0
    ph, pw = h / gh, w / gw
    res = np.zeros(gh * gw, dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            cell = diff[int(i * ph):int((i + 1) * ph), int(j * pw):int((j + 1) * pw)]
            res[i * gw + j] = float(cell.mean())
    return res


class ImportanceTracker:
    """Per-patch importance ~ EMA of recent surprisal ("attention x error" proxy).

    Claim: 品質保持. Patches that keep surprising the model are where output-relevant
    events happen; tracking them lets the task-aware gate lower their threshold so we
    do not skip a semantically important change to save a few FLOPs.
    """

    def __init__(self, n_patches: int, momentum: float = 0.85):
        self.w = np.zeros(n_patches, dtype=np.float32)
        self.momentum = momentum

    def update(self, indices: np.ndarray, surprisal: np.ndarray):
        """Blend freshly-observed embedding surprisal into the importance EMA."""
        if indices.size == 0:
            self.w *= self.momentum
            return
        decayed = self.w * self.momentum
        s = np.zeros_like(self.w)
        s[indices] = surprisal
        self.w = decayed + (1 - self.momentum) * s

    def normalized(self) -> np.ndarray:
        """Importance in [0, 1] (max-normalised, robust to all-zero)."""
        m = float(self.w.max())
        if m <= 1e-8:
            return np.zeros_like(self.w)
        return self.w / m


@dataclass
class GateDecision:
    """Which patches to encode this frame, and why."""

    encoded_mask: np.ndarray     # (P,) bool — True => re-encode with the backbone
    residual: np.ndarray         # (P,) proxy surprisal used for the decision
    tau_eff: np.ndarray          # (P,) per-patch effective threshold applied

    @property
    def n_encoded(self) -> int:
        return int(self.encoded_mask.sum())


class SurprisalGate:
    """Threshold the (task-weighted) motion-compensated residual to pick encode set.

    Claim: 省エネ + 品質保持. Higher τ => fewer encodes (energy down); the task-aware
    term prevents that from silently dropping important patches.
    """

    def __init__(self, tau: float = 0.06, task_aware: bool = True, task_lambda: float = 1.5):
        self.tau = tau
        self.task_aware = task_aware
        self.task_lambda = task_lambda

    def decide(self, residual: np.ndarray, importance: Optional[np.ndarray] = None,
               force_mask: Optional[np.ndarray] = None) -> GateDecision:
        """Return the encode decision.

        Claim: 省エネ. ``residual`` is the cheap proxy surprisal (P,). ``importance``
        (P, in [0,1]) lowers τ where it matters (task-aware). ``force_mask`` (P, bool)
        marks patches that *must* be encoded regardless of residual — e.g. patches the
        motion warp could not fill (newly revealed at the frame edge), guaranteeing we
        never reuse a token with no valid source (品質保持).
        """
        p = residual.shape[0]
        tau_eff = np.full(p, self.tau, dtype=np.float32)
        if self.task_aware and importance is not None:
            tau_eff = self.tau / (1.0 + self.task_lambda * importance.astype(np.float32))
        mask = residual > tau_eff
        if force_mask is not None:
            mask = mask | force_mask.astype(bool)
        return GateDecision(encoded_mask=mask, residual=residual, tau_eff=tau_eff)


def embedding_surprisal(predicted: torch.Tensor, actual: torch.Tensor) -> np.ndarray:
    """True per-patch surprisal ||actual - predicted|| for the encoded patches.

    Claim: 品質保持. Computed only where we already paid to encode; used to train the
    predictor and to feed importance — it is the ground-truth the cheap proxy stands
    in for, so reporting it lets the benchmark audit how good the proxy is.
    """
    return (actual - predicted).norm(dim=-1).detach().cpu().numpy()
