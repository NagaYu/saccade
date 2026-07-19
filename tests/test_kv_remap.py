"""Motion-compensation tests. Core claim under test: 省エネ — motion-compensated KV
reuse recomputes strictly fewer patches than naive same-position reuse under ego-motion.
This is the mechanism that lets Saccade skip steady motion that TemporalSim cannot."""

import numpy as np
import torch

from saccade.kv_remap import remap_grid


def _positional_grid(gh, gw, d):
    """Embeddings that uniquely encode each cell, so 'did the content move' is detectable."""
    e = torch.zeros(gh * gw, d)
    for i in range(gh):
        for j in range(gw):
            e[i * gw + j, 0] = i
            e[i * gw + j, 1] = j
            e[i * gw + j, 2] = (i * 7 + j * 13) % 5
    return e


def test_remap_reconstructs_translated_content():
    """Under a +1-column world shift, remap pulls each cell's token from its source cell.

    Claim: 省エネ. warped[i,j] must equal cache[i, j-1] (content that moved into (i,j)),
    so the reused token is correct despite the spatial shift.
    """
    gh, gw, d = 8, 8, 4
    cache = _positional_grid(gh, gw, d)
    disp = np.zeros((gh, gw, 2), dtype=np.float32)
    disp[..., 0] = 1.0                               # content moved right by one patch
    warped, valid = remap_grid(cache, disp, (gh, gw))
    warped = warped.reshape(gh, gw, d)
    cache_g = cache.reshape(gh, gw, d)
    for i in range(gh):
        for j in range(1, gw):                       # interior columns have a valid source
            assert torch.allclose(warped[i, j], cache_g[i, j - 1], atol=1e-5)
    valid = valid.reshape(gh, gw)
    assert not valid[:, 0].any()                     # revealed left column is invalid
    assert valid[:, 1:].all()


def test_motion_compensation_reduces_recompute_vs_same_position():
    """The headline unit claim: fewer patches look 'surprising' after compensation.

    Under a pure translation of a textured world, same-position reuse sees almost every
    patch change, while motion-compensated reuse sees change only at the revealed edge.
    """
    gh, gw, d = 12, 12, 8
    torch.manual_seed(0)
    cache = torch.randn(gh * gw, d)                  # 'previous frame' tokens
    # true 'current frame': the same world shifted right by one patch column
    cur = cache.reshape(gh, gw, d).clone()
    cur[:, 1:] = cache.reshape(gh, gw, d)[:, :-1]
    cur[:, 0] = torch.randn(gh, d)                   # newly revealed content
    cur = cur.reshape(gh * gw, d)

    disp = np.zeros((gh, gw, 2), dtype=np.float32); disp[..., 0] = 1.0
    warped, valid = remap_grid(cache, disp, (gh, gw))

    tol = 1e-3
    same_pos_recompute = (cur - cache).norm(dim=-1) > tol          # naive reuse
    motion_recompute = ((cur - warped).norm(dim=-1) > tol) | (~valid)  # compensated reuse
    assert int(motion_recompute.sum()) < int(same_pos_recompute.sum())
    # compensated reuse should only need the revealed edge column (~gh patches)
    assert int(motion_recompute.sum()) <= gh
    assert int(same_pos_recompute.sum()) > 3 * gh


def test_engine_motion_compensation_reduces_total_recompute(backbone, stream):
    """End-to-end: on the walking stream, enabling motion compensation cuts the total
    number of re-encoded patches vs the same engine with it disabled. Claim: 省エネ."""
    from saccade import EngineConfig, SaccadeEngine
    frames, _ = stream

    def total_encoded(mc):
        cfg = EngineConfig(mode="saccade", budget_watts=None, motion_compensation=mc,
                           use_predictor=True, fps=10.0)
        res = SaccadeEngine(backbone, cfg, measure_fidelity=False).run(frames)
        return sum(r.n_encoded for r in res)

    with_mc = total_encoded(True)
    without_mc = total_encoded(False)
    assert with_mc < without_mc
    assert with_mc < 0.6 * without_mc                # substantial, not marginal
