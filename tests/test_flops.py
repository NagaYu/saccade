"""FLOP/energy model tests. Claim under test: 省エネ / 予算保証 (the units are sound)."""

import pytest

from saccade import FlopModel, vit_s_like


def test_frame_flops_monotonic_in_encoded():
    """More encoded patches must never cost fewer FLOPs (basis of the energy claim)."""
    fm = vit_s_like()
    vals = [fm.frame_flops(n) for n in range(0, fm.n_patches + 1, 10)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert fm.frame_flops(fm.n_patches) == pytest.approx(fm.full_frame_flops())


def test_dropped_frame_cheaper_than_processed():
    """A fully skipped frame pays no overhead; a processed 0-encode frame still does."""
    fm = vit_s_like()
    assert fm.frame_flops(0, processed=False) == 0.0
    assert fm.frame_flops(0, processed=True) == fm.fixed_overhead_flops
    assert fm.frame_flops(0, processed=True) > fm.frame_flops(0, processed=False)


def test_energy_is_linear_in_flops():
    fm = vit_s_like(j_per_flop=2e-12)
    assert fm.joules(1e12) == pytest.approx(2.0)
    assert fm.frame_joules(5) == pytest.approx(fm.joules(fm.frame_flops(5)))


def test_per_patch_decomposition_positive():
    fm = vit_s_like()
    assert fm.embed_flops > 0 and fm.linear_flops > 0 and fm.attn_flops > 0
    assert fm.per_patch_flops == pytest.approx(fm.embed_flops + fm.linear_flops + fm.attn_flops)


def test_negative_encoded_rejected():
    with pytest.raises(ValueError):
        FlopModel(196, 384, 12, 4.0, 768).frame_flops(-1)
