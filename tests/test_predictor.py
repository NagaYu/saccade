"""Predictor tests. Claim under test: 品質保持 (a self-supervised predictor learns to
anticipate patch embeddings, so the reconstructed map stays close to the encoder)."""

import numpy as np
import torch

from saccade import GRUPatchPredictor


def _l2(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def test_online_learning_reduces_prediction_error():
    """Feeding a predictable drifting sequence, the online loss must drop over time.

    Claim: 品質保持. The predictor is trained only on observed (encoded) patches with no
    labels; if learning works, later one-step predictions are markedly better than early
    ones — which is what lets the gate safely reuse un-encoded patches.
    """
    torch.manual_seed(0)
    P, D = 16, 32
    pred = GRUPatchPredictor(embed_dim=D, n_patches=P, hidden=64, lr=2e-2, seed=0)
    v0 = _l2(torch.randn(P, D))
    delta = _l2(torch.randn(P, D)) * 0.15          # smooth, learnable drift direction
    motion = np.zeros(6, dtype=np.float32)
    flow = np.zeros((P, 2), dtype=np.float32)

    losses = []
    prev = v0
    for t in range(1, 120):
        target = _l2(v0 + t * delta)                # predictable straight drift
        pred.predict(prev, flow, motion)            # warped cache == previous truth
        loss = pred.observe(torch.arange(P), target)
        losses.append(loss)
        prev = target

    early = float(np.mean(losses[:15]))
    late = float(np.mean(losses[-15:]))
    assert late < 0.5 * early, f"predictor failed to learn: early={early:.4f} late={late:.4f}"


def test_predict_shapes_and_finite():
    P, D = 20, 24
    pred = GRUPatchPredictor(embed_dim=D, n_patches=P, hidden=32)
    warped = _l2(torch.randn(P, D))
    out = pred.predict(warped, np.zeros((P, 2), np.float32), np.zeros(6, np.float32))
    assert out.shape == (P, D)
    assert torch.isfinite(out).all()
    # output is L2-normalised like the backbone convention
    assert torch.allclose(out.norm(dim=-1), torch.ones(P), atol=1e-4)


def test_observe_noop_without_prior_predict():
    pred = GRUPatchPredictor(embed_dim=8, n_patches=4, hidden=8)
    assert pred.observe(torch.arange(4), _l2(torch.randn(4, 8))) == 0.0


def test_trust_gate_prevents_predictor_from_backfiring(backbone, stream):
    """Enabling the predictor must never be worse than pure motion-compensated reuse.

    Claim: 品質保持. A naive predictor measurably *degraded* fidelity, because ground truth
    only exists for gate-encoded (surprising) patches while the correction is applied to
    skipped (unsurprising) ones. The trust gate — earned via a counterfactual check on
    randomly explored patches — must hold even at a learning rate large enough to be
    destructive without it.
    """
    import numpy as np
    from saccade import EngineConfig, GRUPatchPredictor, SaccadeEngine

    frames, _ = stream

    def fidelity(use_predictor, lr=None):
        cfg = EngineConfig(mode="saccade", budget_watts=None, use_predictor=use_predictor,
                           fps=10.0)
        eng = SaccadeEngine(backbone, cfg, measure_fidelity=True)
        if use_predictor and lr is not None:
            eng.predictor = GRUPatchPredictor(backbone.embed_dim, backbone.n_patches,
                                              lr=lr, seed=0)
        res = eng.run(frames)
        return float(np.nanmean([r.fidelity for r in res])), eng

    warp_only, _ = fidelity(False)
    for lr in [1e-3, 2e-2]:               # 2e-2 is destructive without the trust gate
        fid, eng = fidelity(True, lr)
        assert fid >= warp_only - 1e-4, (
            f"predictor backfired at lr={lr}: {fid:.5f} < warp-only {warp_only:.5f}")
        assert 0.0 <= float(eng.predictor.trust) <= 1.0


def test_trust_decays_when_residual_is_useless():
    """Trust must fall when the 'correction' is pure noise. Claim: 品質保持."""
    import numpy as np
    from saccade import GRUPatchPredictor

    P, D = 12, 16
    pred = GRUPatchPredictor(embed_dim=D, n_patches=P, hidden=16, lr=0.0, trust_init=1.0)
    torch.manual_seed(0)
    pred.head.weight.data.normal_(0, 0.5)      # a deliberately harmful, untrainable head
    pred.head.bias.data.normal_(0, 0.5)
    target = _l2(torch.randn(P, D))
    for _ in range(40):
        pred.predict(target, np.zeros((P, 2), np.float32), np.zeros(6, np.float32))
        pred.observe(torch.arange(P), target)   # warp already equals truth => residual only hurts
    assert float(pred.trust) < 0.2, f"trust stayed high on a harmful head: {float(pred.trust)}"
