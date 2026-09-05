---
license: mit
library_name: pytorch
tags:
  - edge-ai
  - energy-efficiency
  - video
  - vision-transformer
  - streaming
---

# Saccade patch predictor

A tiny forward predictor for **always-on edge vision**: given a motion-compensated cache of
ViT patch embeddings plus a pseudo-IMU signal, it predicts what each patch embedding will
look like in the *next* frame — so the encoder only has to re-run on patches whose change
the prediction could not explain.

This is the learned component of [**Saccade**](https://github.com/NagaYu/saccade)
(prediction-error gating + energy-budget control for an always-on edge VLM).

- 💻 **Code**: https://github.com/NagaYu/saccade
- 📊 **Benchmark dataset**: https://huggingface.co/datasets/NagaYu/saccade-egomotion-bench
- 🚀 **Interactive demo**: https://huggingface.co/spaces/NagaYu/saccade

## What this is (and honestly, what it is not)

The predictor is *not* what produces most of Saccade's energy saving — motion-compensated
cache reuse is. The predictor's job is to recover the part of the change that the spatial
warp gets wrong. Its value therefore scales with **how lossy the warp is in a given
embedding space**, and we measured exactly that:

| Embedding space | warp only | + predictor (cold, online) | + predictor (this checkpoint) |
|---|---|---|---|
| `SyntheticBackbone` (patch-independent) | 0.96891 | 0.96973 | **0.96974** (+0.0008) |
| `facebook/dinov2-small` (context-mixing) | 0.83288 | 0.84814 | **0.84988** (+0.0170) |

*(mean cosine fidelity of the reconstructed patch-token map vs a Full encoder, on a
held-out ego-motion stream, seed 99.)*

The pattern is the point. Where each patch embedding is an independent function of its own
pixels, a spatial warp is nearly exact, the residual left over is by definition the
*unpredictable* innovation, and there is almost nothing to learn. Where tokens mix global
context through self-attention (any real ViT), warping is a cruder approximation and the
predictor recovers a meaningful chunk of the loss.

## The trust gate — why enabling this can't backfire

An earlier version of this predictor **made things worse**: fidelity decreased monotonically
with learning rate. The cause was a train/apply distribution mismatch — at deployment,
ground truth only exists for patches the gate chose to *encode* (the surprising ones), but
the prediction is applied to the patches it *skipped* (the unsurprising ones). A correction
fitted on the former is wrong for the latter.

Two mechanisms fix it, and both are in this checkpoint:

1. **Residual conditioning** — the gate's cheap per-patch appearance residual is an input
   feature, so the network can tell which regime a patch is in.
2. **An earned trust scalar** — the applied correction is `warped + trust · residual`, where
   `trust ∈ [0,1]` is raised or lowered by a counterfactual check ("would the full residual
   have beaten doing nothing?") evaluated on a handful of randomly *explored* patches drawn
   from the skipped population — i.e. the distribution the prediction is actually used on.

The result: trust settles around 0.35–0.50 where the residual helps and decays toward 0
where it does not, so turning the predictor on is never worse than pure motion-compensated
reuse — even at a learning rate that would otherwise be destructive.

`trust` and `gain_ema` are saved as buffers, so a loaded checkpoint keeps the trust it earned.
The shipped value is **calibrated on the held-out stream**, not on the training streams: during
offline training the head starts untrained, so the counterfactual check correctly drives trust to
0 early on and recovers only slowly. Saving that transient would hand you a predictor that applies
no correction until it re-earns trust online.

## Files

| File | Description |
|---|---|
| `predictor_synthetic.safetensors` | Predictor for the `SyntheticBackbone` embedding space (D=384) |
| `predictor_dinov2-small.safetensors` | Predictor for the `facebook/dinov2-small` token space (D=384) |
| `backbone_synthetic.safetensors` | Frozen random projection of the SyntheticBackbone, so that embedding space is byte-reproducible |
| `config.json` | Geometry, energy model, training provenance and the full eval record |

The GRU's weights are **shared across patches**, so a checkpoint is independent of the patch
grid (196 patches at 14×14 and 256 at 16×16 both load fine). What a checkpoint *is* tied to
is the embedding space it was trained on — use the matching file.

## Usage

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from saccade import GRUPatchPredictor, SaccadeEngine, EngineConfig, make_backbone

backbone = make_backbone("hf:facebook/dinov2-small")
predictor = GRUPatchPredictor(backbone.embed_dim, backbone.n_patches, lr=1e-3)
predictor.load_state_dict(load_file(hf_hub_download(
    "NagaYu/saccade-predictor", "predictor_dinov2-small.safetensors")))

cfg = EngineConfig(mode="saccade", budget_watts=0.02, fps=10.0)
engine = SaccadeEngine(backbone, cfg)
engine.predictor = predictor          # warm start instead of learning from scratch

for frame in your_video_stream:       # HxWx3 uint8 RGB
    r = engine.step(frame)
    print(r.n_encoded, r.encoded_fraction, r.cum_joules, r.budget_ok)
```

Install the package first: `pip install git+https://github.com/NagaYu/saccade`.

## Training

Self-supervised, no labels: targets are simply the vision encoder's own outputs.
The engine trains the predictor online on whichever patches the surprisal gate already paid
to encode; for this published checkpoint we additionally remove the gate's selection bias by
supervising on every patch offline (`train_predictor_on_all=True`, valid only when ground
truth is being computed anyway).

- Streams: 3–5 scripted ego-motion sequences (`synthetic_walking_stream`, seeds 0/2/3[/4/5])
- Objective: `||prediction − true embedding||²`, SGD(momentum=0.9), lr 1e-3
- Held-out evaluation: seed 99, never seen during training

Reproduce with [`hf/export_model.py`](https://github.com/NagaYu/saccade/blob/main/hf/export_model.py).

## Limitations

- Evaluated on **scripted synthetic ego-motion**, not natural video. The regimes (static /
  walk / turn / independently-moving object) are controlled on purpose so the claim is
  reproducible, but they are not a substitute for a real wearable-camera study.
- The predictor is tied to its embedding space; there is no cross-backbone transfer.
- Energy figures throughout the project come from an analytic FLOP model
  (`FLOPs × [J/FLOP]`), not from a power meter.
- The gain on the synthetic space (+0.0007) is small enough to be within run-to-run noise
  for practical purposes — it is reported because the *shape* of the result matters, not
  because that checkpoint is impressive.

## License

MIT
