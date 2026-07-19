"""Gate tests. Claim under test: 省エネ (threshold controls how much we skip) and the
品質保持 safety valve (task-awareness + forced encodes protect important patches)."""

import numpy as np

from saccade import SurprisalGate
from saccade.gate import ImportanceTracker


def test_higher_tau_encodes_fewer():
    """Monotonicity: raising τ never increases the encode set. Basis of budget control."""
    rng = np.random.default_rng(0)
    residual = rng.random(196).astype(np.float32)
    counts = [SurprisalGate(tau=t, task_aware=False).decide(residual).n_encoded
              for t in [0.1, 0.3, 0.5, 0.7, 0.9]]
    assert all(b <= a for a, b in zip(counts, counts[1:]))
    assert counts[0] > counts[-1]


def test_task_aware_lowers_threshold_for_important_patches():
    """An important patch just below τ gets encoded once task-awareness lowers its bar.

    Claim: 品質保持. Prevents dropping an output-critical patch merely to save FLOPs.
    """
    P = 196
    residual = np.full(P, 0.05, dtype=np.float32)   # everything just under τ=0.06
    importance = np.zeros(P, dtype=np.float32)
    importance[10] = 1.0                            # one very important patch
    plain = SurprisalGate(tau=0.06, task_aware=False).decide(residual)
    aware = SurprisalGate(tau=0.06, task_aware=True, task_lambda=1.5).decide(residual, importance)
    assert plain.n_encoded == 0
    assert aware.encoded_mask[10]
    assert aware.tau_eff[10] < aware.tau_eff[0]


def test_force_mask_always_encoded():
    """Patches the motion warp could not fill must be encoded regardless of residual."""
    P = 50
    residual = np.zeros(P, dtype=np.float32)
    force = np.zeros(P, dtype=bool); force[[1, 7, 40]] = True
    dec = SurprisalGate(tau=0.9).decide(residual, force_mask=force)
    assert dec.encoded_mask[[1, 7, 40]].all()
    assert dec.n_encoded == 3


def test_importance_tracker_decays_and_normalizes():
    it = ImportanceTracker(10, momentum=0.5)
    it.update(np.array([2]), np.array([1.0]))
    assert it.normalized()[2] == 1.0
    before = it.w[2]
    it.update(np.array([], dtype=int), np.array([]))   # decay only
    assert it.w[2] < before
    assert (it.normalized() <= 1.0).all() and (it.normalized() >= 0.0).all()
