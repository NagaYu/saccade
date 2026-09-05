"""SaccadeEngine: the always-on loop that ties the four cores together.

Claim demonstrated: all three at once — **省エネ** (encode only surprising patches),
**品質保持** (reconstructed patch-token map stays close to the Full encoder), and
**予算保証** (a hard energy ceiling with an anytime degraded output).

One engine produces all three benchmark conditions so the comparison is fair — the
*only* thing that differs is the mechanism under test:

* ``mode="full"``        — re-encode every patch every frame (ground truth / upper cost).
* ``mode="temporalsim"`` — per-patch **same-position** frame-difference gate; reuse the
                           co-located cached token otherwise. No motion compensation,
                           no predictor, no budget control. The "existing method".
* ``mode="saccade"``     — motion-compensated residual gate + forward predictor +
                           token-caching reuse + closed-loop energy budget controller.

Per Saccade frame: refill budget → (safety-drop if starved) → optical-flow pseudo-IMU
→ motion-compensate the cache → predict → gate on the compensated residual → cap the
encode set to the energy budget → encode that subset → splice fresh+predicted tokens →
self-train the predictor on what we just observed → commit energy.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import torch

from .backbone import VisionBackbone
from .energy_controller import EnergyBudgetController
from .gate import ImportanceTracker, SurprisalGate, embedding_surprisal, patch_residual
from .kv_remap import remap_gray, remap_grid
from .motion import estimate_motion, to_gray
from .predictor import GRUPatchPredictor
from .types import EngineConfig, FrameResult, RunSummary


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def _cap_selection(cand_idx: np.ndarray, residual: np.ndarray, n_afford: int,
                   forced_mask: np.ndarray) -> np.ndarray:
    """Keep at most ``n_afford`` candidates, prioritising forced patches then residual.

    Claim: 予算保証. This is where the gate's wish list is truncated to what the energy
    budget allows, so encoded count never exceeds the affordable count.
    """
    if len(cand_idx) <= n_afford:
        return cand_idx
    score = residual[cand_idx].astype(np.float64).copy()
    score[forced_mask[cand_idx]] = np.inf     # forced (invalid-warp) patches first
    order = np.argsort(-score)
    return cand_idx[order[:n_afford]]


class SaccadeEngine:
    """Stateful always-on engine. Feed it frames; it returns per-frame measurements.

    Claim: 省エネ + 品質保持 + 予算保証 (integration of all four cores).
    """

    def __init__(self, backbone: VisionBackbone, config: EngineConfig,
                 measure_fidelity: bool = True, collect_masks: bool = False, seed: int = 0,
                 train_predictor_on_all: bool = False):
        # train_predictor_on_all: offline-only. Supervise the predictor on every patch
        # (requires measure_fidelity so ground truth exists) to remove the gate's
        # selection bias when producing a publishable checkpoint. Never used in the
        # deployment path, where only encoded patches have ground truth.
        self.train_predictor_on_all = train_predictor_on_all
        self.bb = backbone
        self.cfg = config
        self.grid = backbone.grid
        self.P = backbone.n_patches
        self.D = backbone.embed_dim
        self.fm = backbone.flop_model
        self.measure_fidelity = measure_fidelity
        self.collect_masks = collect_masks

        self.gate = SurprisalGate(tau=config.tau, task_aware=config.task_aware,
                                  task_lambda=config.task_lambda)
        self.importance = ImportanceTracker(self.P)
        self.predictor = GRUPatchPredictor(self.D, self.P, lr=config.online_lr, seed=seed)
        self.controller = EnergyBudgetController(
            self.fm, config.budget_watts, config.fps,
            reserve_seconds=config.reserve_seconds,
            allow_frame_skip=config.allow_frame_skip, tau_init=config.tau,
        )

        self._rng = np.random.default_rng(seed)   # exploration sampling (reproducible)
        self.E_cache = torch.zeros(self.P, self.D)
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_idx = -1
        self.cum_joules = 0.0
        self._bootstrapped = False

    # ------------------------------------------------------------------ utils
    def _encode_all(self, frame: np.ndarray) -> torch.Tensor:
        return _l2norm(self.bb.encode(frame))

    def _fresh_for(self, frame: np.ndarray, idx: np.ndarray,
                   e_full: Optional[torch.Tensor]) -> torch.Tensor:
        if idx.size == 0:
            return torch.empty(0, self.D)
        if e_full is not None:                       # reuse the ground truth we already have
            return e_full[torch.as_tensor(idx, dtype=torch.long)]
        return _l2norm(self.bb.encode(frame, idx))   # true deploy path: encode only the subset

    def _fidelity(self, e_recon: torch.Tensor, e_full: Optional[torch.Tensor]) -> float:
        if e_full is None:
            return float("nan")
        return float((e_recon * e_full).sum(dim=-1).mean())

    def _result(self, n_encoded: int, processed: bool, tau: float, fidelity: float,
                surprisal_mean: float, mask: Optional[np.ndarray]) -> FrameResult:
        frame_flops = self.fm.frame_flops(n_encoded, processed=processed)
        frame_joules = self.fm.joules(frame_flops)
        self.cum_joules += frame_joules
        budget_ok = self.controller.invariant_ok()
        return FrameResult(
            index=self.frame_idx, n_encoded=n_encoded, n_patches=self.P, processed=processed,
            frame_flops=frame_flops, frame_joules=frame_joules, cum_joules=self.cum_joules,
            tau=tau, fidelity=fidelity, surprisal_mean=surprisal_mean, budget_ok=budget_ok,
            encoded_mask=(mask.reshape(self.grid) if (self.collect_masks and mask is not None) else None),
        )

    # ------------------------------------------------------------------ modes
    def step(self, frame: np.ndarray) -> FrameResult:
        """Process one frame in whichever ``mode`` this engine was configured for."""
        self.frame_idx += 1
        if self.cfg.mode == "full":
            return self._step_full(frame)
        if self.cfg.mode == "temporalsim":
            return self._step_temporalsim(frame)
        return self._step_saccade(frame)

    def _step_full(self, frame: np.ndarray) -> FrameResult:
        """Baseline A: encode everything, every frame. Claim: upper bound on 省エネ cost."""
        e_full = self._encode_all(frame)
        self.E_cache = e_full
        self.prev_gray = to_gray(frame, self.cfg.img_size)
        mask = np.ones(self.P, dtype=bool)
        return self._result(self.P, True, self.gate.tau, 1.0, 0.0, mask)

    def _step_temporalsim(self, frame: np.ndarray) -> FrameResult:
        """Baseline B: same-position per-patch frame-difference gate (no motion comp).

        Claim: this is the method that *cannot* skip steady ego-motion — its residual
        is computed at fixed grid positions, so a global pan makes every patch differ.
        """
        gray = to_gray(frame, self.cfg.img_size)
        e_full = self._encode_all(frame) if self.measure_fidelity else None
        if not self._bootstrapped or self.prev_gray is None:
            e = e_full if e_full is not None else self._encode_all(frame)
            self.E_cache = e
            self.prev_gray = gray
            self._bootstrapped = True
            return self._result(self.P, True, self.cfg.temporalsim_tau, 1.0, 0.0,
                                np.ones(self.P, dtype=bool))
        residual = patch_residual(gray, self.prev_gray, self.grid)     # SAME position, no warp
        mask = residual > self.cfg.temporalsim_tau
        idx = np.nonzero(mask)[0]
        fresh = self._fresh_for(frame, idx, e_full)
        e_recon = self.E_cache.clone()
        if idx.size:
            e_recon[torch.as_tensor(idx, dtype=torch.long)] = fresh
        fidelity = self._fidelity(e_recon, e_full)
        self.E_cache = e_recon
        self.prev_gray = gray
        return self._result(int(mask.sum()), True, self.cfg.temporalsim_tau, fidelity,
                            float(residual.mean()), mask)

    def _step_saccade(self, frame: np.ndarray) -> FrameResult:
        """Condition C: the full Saccade pipeline. Claim: 省エネ + 品質保持 + 予算保証."""
        cfg = self.cfg
        gray = to_gray(frame, cfg.img_size)
        e_full = self._encode_all(frame) if self.measure_fidelity else None
        warmup = self.frame_idx < cfg.warmup_frames

        # ---- bootstrap (frame 0 or no history): populate the cache -----------
        if not self._bootstrapped or self.prev_gray is None:
            self.controller.begin_frame()
            plan = self.controller.afford(self.P, n_forced=self.P)
            if not plan.processed:
                self.controller.commit(0, processed=False)
                return self._result(0, False, self.controller.current_tau(),
                                    self._fidelity(self.E_cache, e_full), 0.0,
                                    np.zeros(self.P, dtype=bool))
            n = plan.max_encode
            idx = np.arange(n)
            fresh = self._fresh_for(frame, idx, e_full)
            e_recon = torch.zeros(self.P, self.D)
            if n:
                e_recon[torch.as_tensor(idx, dtype=torch.long)] = fresh
            self.E_cache = e_recon
            self.prev_gray = gray
            self.predictor.reset()
            self._bootstrapped = True
            self.controller.commit(n, processed=True)
            mask = np.zeros(self.P, dtype=bool); mask[idx] = True
            return self._result(n, True, self.controller.current_tau(),
                                self._fidelity(e_recon, e_full), 0.0, mask)

        # ---- refill budget; safety-drop if we cannot even look --------------
        self.controller.begin_frame()
        if self.controller.enabled and self.controller.bucket < self.controller.overhead_j:
            self.controller.commit(0, processed=False)     # anytime degraded output = reuse cache
            self.prev_gray = gray
            return self._result(0, False, self.controller.current_tau(),
                                self._fidelity(self.E_cache, e_full), 0.0,
                                np.zeros(self.P, dtype=bool))

        # ---- pseudo-IMU + motion compensation -------------------------------
        # We always estimate ego-motion, but only *compensate* when there is motion
        # to compensate: below the static threshold the warp would only inject
        # optical-flow noise, so we fall back to same-position reuse (== baseline B on
        # static regions). This keeps the motion-comp residual an honest measure of
        # genuine appearance change rather than warp artefacts.
        motion = estimate_motion(self.prev_gray, gray, self.grid)
        do_warp = cfg.motion_compensation and motion.mag_median >= cfg.static_motion_thresh
        if do_warp:
            e_warp, valid = remap_grid(self.E_cache, motion.patch_disp, self.grid)
            gray_ref = remap_gray(self.prev_gray, motion.patch_disp, self.grid)
        else:
            e_warp, valid = self.E_cache.clone(), torch.ones(self.P, dtype=torch.bool)
            gray_ref = self.prev_gray
        e_warp = _l2norm(e_warp)

        # ---- cheap per-patch appearance residual (gate signal) --------------
        # Computed before prediction because the predictor conditions on it: it is the
        # feature that tells the network whether this patch is in the "nothing happened"
        # regime (output ~0 correction) or the "something changed" regime.
        residual = patch_residual(gray, gray_ref, self.grid)

        # ---- predict next-frame embeddings ----------------------------------
        # Only trust the predictor when there is motion to predict. On static frames
        # the (un-warped) cache is already the exact previous embedding, so adding a
        # learned residual would only inject drift — reuse the cache verbatim instead.
        use_pred_now = cfg.use_predictor and do_warp
        if use_pred_now:
            e_pred = self.predictor.predict(e_warp, motion.patch_disp, motion.global_feat,
                                            residual)
        else:
            e_pred = e_warp

        # ---- proactive skip on quiet, predictable frames (utility) ----------
        if self.controller.should_proactively_skip(motion.mag_median):
            e_recon = e_pred
            self.E_cache = e_recon
            self.prev_gray = gray
            self.controller.commit(0, processed=True)
            self.importance.update(np.array([], dtype=int), np.array([]))
            return self._result(0, True, self.controller.current_tau(),
                                self._fidelity(e_recon, e_full), 0.0,
                                np.zeros(self.P, dtype=bool))

        # ---- surprisal gate on the motion-compensated residual --------------
        forced = (~valid).numpy()
        importance = self.importance.normalized() if cfg.task_aware else None
        self.gate.tau = self.controller.current_tau()
        if warmup:
            decision_mask = np.ones(self.P, dtype=bool)      # warm the cache/predictor fully (still capped)
            tau_used = self.gate.tau
        else:
            decision = self.gate.decide(residual, importance, force_mask=forced)
            decision_mask = decision.encoded_mask
            tau_used = self.gate.tau
        cand_idx = np.nonzero(decision_mask)[0]

        # ---- cap the encode set to the energy budget ------------------------
        n_explore = int(round(cfg.explore_frac * self.P)) if use_pred_now else 0
        plan = self.controller.afford(len(cand_idx) + n_explore, n_forced=int(forced.sum()))
        idx = _cap_selection(cand_idx, residual, min(plan.max_encode, len(cand_idx)), forced)

        # ---- exploration: a few patches the gate would have skipped ---------
        # Encoded purely so the predictor gets ground truth from the population it is
        # actually applied to (skipped, low-residual patches). Only when there is budget
        # to spare — under starvation, real surprises come first.
        explore_idx = np.empty(0, dtype=int)
        if n_explore > 0 and plan.max_encode > len(idx):
            pool = np.setdiff1d(np.arange(self.P), idx)
            k = min(n_explore, plan.max_encode - len(idx), pool.size)
            if k > 0:
                explore_idx = self._rng.choice(pool, size=k, replace=False)
                idx = np.concatenate([idx, explore_idx])
        idx = np.sort(idx)

        # ---- encode the chosen subset; splice fresh + predicted -------------
        fresh = self._fresh_for(frame, idx, e_full)
        e_recon = e_pred.clone()
        if idx.size:
            e_recon[torch.as_tensor(idx, dtype=torch.long)] = fresh
        e_recon = _l2norm(e_recon)

        # ---- self-supervised predictor update on observed patches -----------
        surprisal_mean = 0.0
        if idx.size:
            surp = embedding_surprisal(e_pred[torch.as_tensor(idx, dtype=torch.long)], fresh)
            self.importance.update(idx, surp)
            surprisal_mean = float(surp.mean())
        else:
            self.importance.update(np.array([], dtype=int), np.array([]))

        if use_pred_now:     # only train the predictor on frames where it was used
            trust_mask = torch.from_numpy(np.isin(idx, explore_idx)) if explore_idx.size else None
            if self.train_predictor_on_all and e_full is not None:
                # OFFLINE checkpoint training: supervise on EVERY patch. At deployment we
                # only have ground truth for gate-encoded (i.e. surprising) patches, but
                # the prediction is applied to the un-encoded (unsurprising) ones — training
                # on that biased subset teaches a correction that is wrong where it is used.
                # When e_full is already being computed we can remove the bias entirely.
                # offline: every patch is ground truth, so the trust check is unbiased
                # on the low-residual population without needing exploration
                low = torch.from_numpy(residual <= self.gate.tau)
                self.predictor.observe(torch.arange(self.P), e_full, trust_mask=low)
            elif idx.size:
                self.predictor.observe(torch.as_tensor(idx, dtype=torch.long), fresh,
                                       trust_mask=trust_mask)

        fidelity = self._fidelity(e_recon, e_full)
        self.E_cache = e_recon.detach()
        self.prev_gray = gray
        self.controller.commit(int(idx.size), processed=True)

        mask = np.zeros(self.P, dtype=bool)
        mask[idx] = True
        return self._result(int(idx.size), True, tau_used, fidelity, surprisal_mean, mask)

    # ------------------------------------------------------------------- run
    def run(self, frames: Iterable[np.ndarray]) -> List[FrameResult]:
        """Process a whole stream, returning per-frame results. Claim: end-to-end."""
        return [self.step(f) for f in frames]


def summarize(results: List[FrameResult], label: str, fps: float, warmup: int = 0) -> RunSummary:
    """Aggregate per-frame results into the README metrics. Claim: 省エネ/品質保持/予算保証.

    ``runtime_hours_10kJ`` is the headline edge metric: how long the device could run
    continuously on a 10 kJ energy envelope at this mean power. ``warmup`` drops the
    first N frames so the comparison reflects *steady-state* always-on efficiency (the
    one-time cache/predictor warm-up amortises to zero over an hours-long deployment);
    it is applied identically to every condition so the frames compared stay aligned.
    """
    results = results[warmup:]
    n = len(results)
    total_flops = sum(r.frame_flops for r in results)
    total_joules = sum(r.frame_joules for r in results)
    enc = np.mean([r.encoded_fraction for r in results]) if n else 0.0
    fids = [r.fidelity for r in results if not np.isnan(r.fidelity)]
    mean_fid = float(np.mean(fids)) if fids else float("nan")
    elapsed = n / fps
    mean_power = total_joules / elapsed if elapsed > 0 else 0.0
    runtime_hours = (10000.0 / mean_power / 3600.0) if mean_power > 0 else float("inf")
    violations = sum(0 if r.budget_ok else 1 for r in results)
    return RunSummary(
        label=label, n_frames=n, total_flops=total_flops, total_joules=total_joules,
        mean_encoded_fraction=float(enc), mean_fidelity=mean_fid, mean_power_watts=mean_power,
        runtime_hours_10kJ=runtime_hours, budget_violations=violations,
    )
