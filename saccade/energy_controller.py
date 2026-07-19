"""Energy budget controller: closed-loop τ / frame-skip control with a hard cap.

Claim demonstrated: **予算保証** (a hard, provable energy ceiling with an *anytime*
degraded output) plus **省エネ** utility maximisation (spend right up to the budget,
not below it).

Two cooperating layers:

1. Soft loop — adapts the gate threshold τ each frame with a proportional rule so
   the running power tracks the budget B [J/s]. Overspending raises τ (encode
   fewer patches); underspending lowers it (encode more, use the budget you paid
   for). This maximises task utility *within* the budget.

2. Hard cap — a token bucket makes the guarantee provable regardless of what the
   soft loop or the gate do. The bucket holds at most ``E0 = reserve_seconds * B``
   Joules and refills ``B/fps`` Joules per frame. A frame may spend at most what is
   in the bucket. Therefore, for every frame index ``t``:

       cumulative_energy(t)  ≤  B · elapsed(t)  +  E0

   If the bucket cannot even cover a frame's fixed overhead, the frame is *dropped*
   (zero cost) and the engine emits the reused cache — a valid, degraded, anytime
   output. This drop path is what guarantees the ceiling can never be crossed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .flops import FlopModel


@dataclass
class AffordPlan:
    """Controller's verdict for one frame."""

    processed: bool        # False => safety frame-drop (0 cost, reuse-only output)
    max_encode: int        # max patches affordable this frame (0..P)
    available_joules: float


class EnergyBudgetController:
    """Token-bucket energy governor with proportional τ adaptation.

    Claim: 予算保証. The bucket invariant is the guarantee; :meth:`afford` and
    :meth:`commit` are the only places energy is granted or spent, so the ceiling
    holds by construction. ``budget_watts=None`` runs open-loop (baselines / no cap).
    """

    def __init__(self, flop_model: FlopModel, budget_watts: Optional[float], fps: float,
                 reserve_seconds: float = 0.5, allow_frame_skip: bool = True,
                 k_tau: float = 0.4, tau_bounds: Tuple[float, float] = (0.005, 0.5),
                 tau_init: float = 0.06):
        self.fm = flop_model
        self.budget_watts = budget_watts
        self.fps = fps
        self.dt = 1.0 / fps
        self.reserve_seconds = reserve_seconds
        self.allow_frame_skip = allow_frame_skip
        self.k_tau = k_tau
        self.tau_lo, self.tau_hi = tau_bounds
        self.tau = float(min(max(tau_init, self.tau_lo), self.tau_hi))

        self.per_patch_j = flop_model.joules(flop_model.per_patch_flops)
        self.overhead_j = flop_model.joules(flop_model.fixed_overhead_flops)
        self.e_target = (budget_watts * self.dt) if budget_watts is not None else math.inf
        self.E0 = (reserve_seconds * budget_watts) if budget_watts is not None else math.inf
        self.bucket = self.E0                       # start with a full reserve
        self._ema_energy = self.e_target if budget_watts is not None else 0.0
        self._ema_m = 0.8

        self.frame_idx = -1
        self.cum_joules = 0.0

    # -- open-loop? ----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.budget_watts is not None

    def current_tau(self) -> float:
        """The soft-adapted threshold the engine hands to the gate. Claim: 省エネ."""
        return self.tau

    # -- per-frame protocol: begin_frame -> afford -> commit -----------------
    def begin_frame(self) -> float:
        """Refill the bucket for this frame's time slice (capped at E0). Claim: 予算保証."""
        self.frame_idx += 1
        if not self.enabled:
            return math.inf
        self.bucket = min(self.E0, self.bucket + self.budget_watts * self.dt)
        return self.bucket

    def afford(self, n_desired: int, n_forced: int = 0) -> AffordPlan:
        """How many of the gate's desired patches can we pay for without breaking the cap?

        Claim: 予算保証. Encodes are granted only while the bucket covers them. Forced
        patches (invalid-warp / newly revealed) are prioritised; if even the frame's
        fixed overhead is unaffordable, the frame is dropped (anytime degraded output).
        """
        if not self.enabled:
            return AffordPlan(processed=True, max_encode=n_desired, available_joules=math.inf)

        if self.bucket < self.overhead_j:
            # cannot even afford to *look* at the frame -> safety drop (reuse cache)
            return AffordPlan(processed=False, max_encode=0, available_joules=self.bucket)

        spendable = self.bucket - self.overhead_j
        max_by_energy = int(spendable // self.per_patch_j) if self.per_patch_j > 0 else n_desired
        max_encode = min(n_desired, max_by_energy)
        # forced patches take priority but still cannot exceed the hard cap
        max_encode = max(max_encode, min(n_forced, max_by_energy))
        return AffordPlan(processed=True, max_encode=max_encode, available_joules=self.bucket)

    def commit(self, n_encoded: int, processed: bool) -> float:
        """Deduct the frame's energy, update the τ loop, return the frame's Joules.

        Claim: 予算保証 + 省エネ. Deduction keeps the bucket ≥ 0 (ceiling preserved);
        the proportional τ update steers average power toward the budget.
        """
        frame_j = self.fm.frame_joules(n_encoded, processed=processed)
        self.cum_joules += frame_j
        if self.enabled:
            self.bucket -= frame_j
            if self.bucket < 0:      # only possible via float error; clamp defensively
                self.bucket = 0.0
            # proportional τ adaptation on smoothed energy error
            self._ema_energy = self._ema_m * self._ema_energy + (1 - self._ema_m) * frame_j
            ratio = self._ema_energy / self.e_target if self.e_target > 0 else 1.0
            self.tau *= math.exp(self.k_tau * (ratio - 1.0))
            self.tau = float(min(max(self.tau, self.tau_lo), self.tau_hi))
        return frame_j

    # -- guarantee check (used by the engine + tests) ------------------------
    def elapsed(self) -> float:
        return (self.frame_idx + 1) * self.dt

    def invariant_ok(self, eps: float = 1e-9) -> bool:
        """True iff cumulative energy still satisfies cum ≤ B·elapsed + E0. Claim: 予算保証."""
        if not self.enabled:
            return True
        ceiling = self.budget_watts * self.elapsed() + self.E0
        return self.cum_joules <= ceiling + eps

    def should_proactively_skip(self, motion_mag: float, motion_thresh: float = 5e-3) -> bool:
        """Utility heuristic: bank credit by skipping near-static frames when behind budget.

        Claim: 省エネ. Distinct from the safety drop — this proactively conserves budget
        during predictable/quiet stretches so it is available for genuine events later.
        """
        if not (self.enabled and self.allow_frame_skip):
            return False
        behind = self.bucket < 0.5 * self.E0
        return behind and motion_mag < motion_thresh
