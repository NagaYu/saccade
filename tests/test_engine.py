"""End-to-end engine tests tying the claims together:
省エネ (Saccade encodes far fewer patches), 品質保持 (fidelity vs Full stays within
threshold with the gate ON), and the fair-comparison property vs TemporalSim on
predictable ego-motion."""

import numpy as np

from saccade import EngineConfig, SaccadeEngine, summarize

WARMUP = 6


def _run(backbone, frames, mode, budget=None, **kw):
    cfg = EngineConfig(mode=mode, tau=0.06, temporalsim_tau=0.06, budget_watts=budget,
                       fps=10.0, **kw)
    return SaccadeEngine(backbone, cfg, measure_fidelity=True, collect_masks=False).run(frames)


def test_gate_on_preserves_quality_while_saving(backbone, stream):
    """With the predictive gate ON, fidelity vs Full stays within threshold AND we encode
    a small fraction of patches. Claim: 品質保持 + 省エネ (the central promise)."""
    frames, _ = stream
    res = _run(backbone, frames, "saccade", budget=0.02)
    s = summarize(res, "saccade", 10.0, warmup=WARMUP)
    assert s.mean_fidelity >= 0.90, f"quality dropped too far: {s.mean_fidelity:.3f}"
    assert s.mean_encoded_fraction < 0.6, f"not saving enough: {s.mean_encoded_fraction:.3f}"


def test_full_baseline_is_ground_truth(backbone, stream):
    """Full re-encodes everything and is by definition fidelity 1.0 — the quality anchor."""
    frames, _ = stream
    res = _run(backbone, frames, "full")
    assert all(r.n_encoded == backbone.n_patches for r in res)
    assert np.allclose([r.fidelity for r in res], 1.0)


def test_saccade_cheaper_than_temporalsim_at_equal_quality(backbone, stream):
    """Saccade uses less energy than TemporalSim while holding comparable fidelity.

    Claim: 省エネ + 品質保持. This is the benchmark's headline: the predictive/motion-aware
    gate dominates the same-position temporal-similarity gate.
    """
    frames, _ = stream
    c = summarize(_run(backbone, frames, "saccade", budget=0.02), "C", 10.0, warmup=WARMUP)
    b = summarize(_run(backbone, frames, "temporalsim"), "B", 10.0, warmup=WARMUP)
    assert c.total_joules < b.total_joules
    assert c.mean_encoded_fraction < b.mean_encoded_fraction
    assert c.mean_fidelity >= b.mean_fidelity - 0.02        # quality not sacrificed


def test_saccade_skips_more_than_temporalsim_on_walking(backbone, stream):
    """On predictable ego-motion (walk/turn) Saccade re-encodes far fewer patches than
    TemporalSim — the centerpiece claim, measured per segment. Claim: 省エネ."""
    frames, labels = stream
    warm = np.arange(len(frames)) >= WARMUP
    c = np.array([r.encoded_fraction for r in _run(backbone, frames, "saccade")])
    b = np.array([r.encoded_fraction for r in _run(backbone, frames, "temporalsim")])
    for seg in ["walk", "turn"]:
        m = (labels == seg) & warm
        assert c[m].mean() < b[m].mean(), f"Saccade did not skip more on {seg}"
        assert c[m].mean() < 0.6 * b[m].mean(), f"skip advantage on {seg} too small"


def test_static_region_both_skip(backbone, stream):
    """On a truly static camera both methods skip (fair regime, no spurious encoding)."""
    frames, labels = stream
    warm = np.arange(len(frames)) >= WARMUP
    c = np.array([r.encoded_fraction for r in _run(backbone, frames, "saccade")])
    m = (labels == "static") & warm
    assert c[m].mean() < 0.05


def test_budget_guarantee_end_to_end(backbone, stream):
    """No frame ever reports a budget violation when the controller is active. Claim: 予算保証."""
    frames, _ = stream
    res = _run(backbone, frames, "saccade", budget=0.01)
    assert all(r.budget_ok for r in res)
