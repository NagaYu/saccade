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
