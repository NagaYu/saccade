"""Shared dataclasses: engine configuration and per-frame results.

These are deliberately plain data containers so every core module speaks the same
vocabulary (patches encoded, joules spent, fidelity) and the benchmark can compare
the three conditions (Full / TemporalSim / Saccade) frame-by-frame on equal terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EngineConfig:
    """Configuration for :class:`saccade.engine.SaccadeEngine`.

    Groups the knobs for all four cores. The three benchmark conditions are all
    produced by this one engine with different flags, which keeps the comparison
    apples-to-apples (only the mechanism under test differs).
    """

    # --- geometry (must match the backbone) ---
    grid_h: int = 14
    grid_w: int = 14
    img_size: int = 224

    # --- surprisal gate ---
    tau: float = 0.06                 # base threshold on motion-compensated residual (normalised 0..1)
    task_aware: bool = True           # scale threshold by per-patch importance (attention x error)
    task_lambda: float = 1.5          # strength of the task-aware correction

    # --- motion compensation ---
    motion_compensation: bool = True  # remap cached tokens along ego-motion before gating
    static_motion_thresh: float = 0.01  # below this ego-motion (patch units) skip warp (no motion => nothing to compensate)
    use_predictor: bool = True        # refine warped cache with the learned predictor

    # --- energy budget controller ---
    budget_watts: Optional[float] = None   # J/s; None disables the controller (open loop)
    fps: float = 10.0
    reserve_seconds: float = 0.5      # token-bucket capacity E0 = reserve_seconds * budget
    allow_frame_skip: bool = True     # controller may drop whole frames when starved

    # --- bootstrap / online learning ---
    warmup_frames: int = 6            # frames fully encoded at the start to seed cache + predictor
    online_lr: float = 1e-3           # predictor online SGD step size
    explore_frac: float = 0.01        # fraction of otherwise-skipped patches encoded per frame to
                                      # give the predictor UNBIASED feedback (see predictor trust gate).
                                      # Costs real energy, so it is counted in n_encoded.

    # --- baseline selection ---
    mode: str = "saccade"             # "full" | "temporalsim" | "saccade"
    temporalsim_tau: float = 0.06     # per-patch same-position frame-diff threshold for baseline B

    def n_patches(self) -> int:
        return self.grid_h * self.grid_w


@dataclass
class FrameResult:
    """Everything measured for a single processed frame — the benchmark's atom.

    ``encoded_fraction`` and ``joules`` back the 省エネ claim; ``fidelity`` (cosine
    similarity of the reconstructed patch-token map vs the Full ground truth) backs
    the 品質保持 claim; ``cum_joules``/``budget_ok`` back the 予算保証 claim.
    """

    index: int
    n_encoded: int
    n_patches: int
    processed: bool                    # False => whole frame skipped (pure reuse, anytime output)
    frame_flops: float
    frame_joules: float
    cum_joules: float
    tau: float
    fidelity: float = float("nan")     # cosine(reconstructed, full ground truth) in [-1, 1]
    surprisal_mean: float = float("nan")
    budget_ok: bool = True             # invariant cum_joules <= B*t + E0 held after this frame
    # optional per-patch mask of which patches were (re)encoded (for figures/GIF)
    encoded_mask: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def encoded_fraction(self) -> float:
        return self.n_encoded / max(1, self.n_patches)


@dataclass
class RunSummary:
    """Aggregate metrics for one condition over a whole stream (README table row)."""

    label: str
    n_frames: int
    total_flops: float
    total_joules: float
    mean_encoded_fraction: float
    mean_fidelity: float
    mean_power_watts: float
    runtime_hours_10kJ: float          # estimated continuous runtime under a 10 kJ envelope
    budget_violations: int = 0

    def as_row(self) -> dict:
        return {
            "label": self.label,
            "frames": self.n_frames,
            "GFLOPs": round(self.total_flops / 1e9, 3),
            "Joules": round(self.total_joules, 4),
            "encoded_%": round(100 * self.mean_encoded_fraction, 1),
            "fidelity": round(self.mean_fidelity, 4),
            "mean_W": round(self.mean_power_watts, 5),
            "hrs@10kJ": round(self.runtime_hours_10kJ, 1),
            "budget_viol": self.budget_violations,
        }
