"""Energy controller tests. Claim under test: 予算保証 — a hard, provable energy ceiling
with an anytime degraded output, holding for every budget including pathologically small
ones."""

import math

import numpy as np
import pytest

from saccade import EngineConfig, SaccadeEngine, vit_s_like
from saccade.energy_controller import EnergyBudgetController


def _ceiling(B, reserve, dt, frame_idx):
    return B * ((frame_idx + 1) * dt) + reserve * B


@pytest.mark.parametrize("B", [0.05, 0.01, 0.003, 0.0008, 0.0001])
def test_engine_never_exceeds_budget(backbone, stream, B):
    """Independently recompute cum_energy ≤ B·elapsed + E0 at every frame. Claim: 予算保証."""
    frames, _ = stream
    cfg = EngineConfig(mode="saccade", budget_watts=B, fps=10.0, reserve_seconds=0.5)
    res = SaccadeEngine(backbone, cfg, measure_fidelity=False).run(frames)
    dt = 1.0 / cfg.fps
    for r in res:
        assert r.cum_joules <= _ceiling(B, cfg.reserve_seconds, dt, r.index) + 1e-9
        assert r.budget_ok
    assert sum(0 if r.budget_ok else 1 for r in res) == 0


def test_anytime_output_under_starvation(backbone, stream):
    """At a punishing budget the engine still returns a valid, finite output every frame.

    Claim: 予算保証 (anytime). Quality degrades toward zero, but there is always an output
    and the ceiling is never crossed — the defining property of an anytime system.
    """
    frames, _ = stream
    B = 5e-5
    cfg = EngineConfig(mode="saccade", budget_watts=B, fps=10.0)
    res = SaccadeEngine(backbone, cfg, measure_fidelity=True).run(frames)
    dt = 1.0 / cfg.fps
    assert len(res) == len(frames)
    for r in res:
        assert math.isfinite(r.frame_joules)
        assert math.isfinite(r.fidelity)             # a (degraded) output exists every frame
        assert r.cum_joules <= _ceiling(B, cfg.reserve_seconds, dt, r.index) + 1e-9
    # under starvation most frames are safety-dropped (pure reuse) yet still produced
    assert sum(1 for r in res if not r.processed) > 0


def test_higher_budget_spends_more_but_still_bounded(backbone, short_stream):
    """More budget => more encoding (utility up), but the ceiling still holds. Claim: 省エネ+予算保証."""
    frames, _ = short_stream
    enc = {}
    for B in [0.002, 0.02, 0.2]:
        cfg = EngineConfig(mode="saccade", budget_watts=B, fps=10.0)
        res = SaccadeEngine(backbone, cfg, measure_fidelity=False).run(frames)
        enc[B] = np.mean([r.encoded_fraction for r in res])
        assert all(r.budget_ok for r in res)
    assert enc[0.002] <= enc[0.02] <= enc[0.2]


def test_controller_bucket_never_negative():
    """Unit test of the token bucket: it is granted energy only while it can pay. Claim: 予算保証."""
    fm = vit_s_like()
    ctrl = EnergyBudgetController(fm, budget_watts=0.01, fps=10.0, reserve_seconds=0.5)
    rng = np.random.default_rng(0)
    for _ in range(200):
        ctrl.begin_frame()
        if ctrl.bucket < ctrl.overhead_j:
            plan = ctrl.afford(fm.n_patches)
            assert not plan.processed
            ctrl.commit(0, processed=False)
        else:
            desired = int(rng.integers(0, fm.n_patches + 1))
            plan = ctrl.afford(desired)
            n = min(desired, plan.max_encode)
            spent = ctrl.commit(n, processed=True)
            assert spent <= plan.available_joules + 1e-12
        assert ctrl.bucket >= -1e-12
        assert ctrl.invariant_ok()


def test_open_loop_when_no_budget():
    """budget=None runs open-loop: everything affordable, invariant trivially satisfied."""
    fm = vit_s_like()
    ctrl = EnergyBudgetController(fm, budget_watts=None, fps=10.0)
    ctrl.begin_frame()
    plan = ctrl.afford(fm.n_patches)
    assert plan.processed and plan.max_encode == fm.n_patches
    ctrl.commit(fm.n_patches, processed=True)
    assert ctrl.invariant_ok()
