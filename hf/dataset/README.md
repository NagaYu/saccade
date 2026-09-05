---
license: mit
task_categories:
  - image-feature-extraction
tags:
  - edge-ai
  - energy-efficiency
  - video
  - ego-motion
  - benchmark
size_categories:
  - n<1K
configs:
  - config_name: frames
    data_files: frames/*.parquet
  - config_name: signals
    data_files: signals/*.parquet
  - config_name: measurements
    data_files: measurements/*.parquet
---

# Saccade ego-motion benchmark

The stream, the raw decision signals, and the per-frame measurements behind
[**Saccade**](https://github.com/NagaYu/saccade) — an always-on edge VLM that re-encodes only
the image patches whose change ego-motion cannot explain.

This dataset exists so the central claim can be **checked without running our code**.

- 💻 **Code**: https://github.com/NagaYu/saccade
- 🤖 **Model**: https://huggingface.co/NagaYu/saccade-predictor
- 🚀 **Demo**: https://huggingface.co/spaces/NagaYu/saccade

## The claim, in one number

On the segments with predictable camera motion (`walk` + `turn`), at an identical
threshold τ = 0.06:

| Decision rule | Patches it must re-encode |
|---|---|
| Same-position residual (what temporal-similarity skipping thresholds) | **47.4 %** |
| Motion-compensated residual (what Saccade thresholds) | **0.3 %** |

Same frames, same threshold — only the reference frame differs. A moving camera changes
almost every pixel, so frame-differencing cannot thin the workload during steady motion;
subtract the ego-motion first and a static world's residual collapses to nearly nothing.

Verify it yourself in four lines:

```python
from datasets import load_dataset
import numpy as np

sig = load_dataset("NagaYu/saccade-egomotion-bench", "signals", split="train").to_pandas()
walk = sig[sig.segment.isin(["walk", "turn"])]
print((np.stack(walk.residual_same_position)        > 0.06).mean())  # ~0.474
print((np.stack(walk.residual_motion_compensated)   > 0.06).mean())  # ~0.003
```

## Configs

### `frames` — the stream itself (144 rows)
The scripted ego-motion video, so another method can be benchmarked on the exact same input.

| Column | Type | Description |
|---|---|---|
| `index` | int32 | frame index |
| `image` | Image | 224×224 RGB frame |
| `segment` | string | regime: `static`, `walk`, `turn`, `event` |
| `ego_motion_mag_mean` | float32 | mean optical-flow magnitude (patch units) |
| `ego_motion_mag_median` | float32 | median flow magnitude — the robust ego-motion estimate (a single moving object must not read as camera motion) |

### `signals` — the two competing decision signals (143 rows)
The direct evidence. Per frame, the per-patch residual map under **both** rules.

| Column | Type | Description |
|---|---|---|
| `index`, `segment` | int32/string | as above |
| `residual_same_position` | float32[196] | per-patch residual vs the *co-located* previous patch |
| `residual_motion_compensated` | float32[196] | per-patch residual vs the *motion-warped* previous frame |
| `encode_frac_same_position` | float32 | fraction above τ=0.06 under the first rule |
| `encode_frac_motion_compensated` | float32 | fraction above τ=0.06 under the second |

Residuals are mean absolute pixel differences in [0,1] on a 14×14 patch grid (row-major;
`reshape(14, 14)` to get a map).

### `measurements` — per-frame energy and quality (432 rows = 144 × 3 conditions)

| Column | Type | Description |
|---|---|---|
| `index` | int64 | frame index |
| `condition` | string | `Full`, `TemporalSim` or `Saccade` |
| `n_encoded`, `encoded_fraction` | int/float | patches actually re-encoded |
| `fidelity` | float | cosine of the reconstructed patch-token map vs the Full encoder |
| `frame_flops`, `frame_joules`, `cum_joules` | float | analytic energy model |
| `processed` | bool | `False` = frame dropped entirely (anytime degraded output) |
| `tau` | float | threshold in force (the budget controller adapts it) |
| `budget_ok` | bool | whether `cum_J ≤ B·t + E₀` still held |

Headline (steady-state, excluding 6 warm-up frames, budget 20 mW):

| Condition | re-encoded | fidelity | energy | runtime @10 kJ |
|---|---|---|---|---|
| Full | 100 % | 1.000 | 1.264 J | 30.3 h |
| TemporalSim | 33.9 % | 0.9723 | 0.430 J | 89.1 h |
| **Saccade** | **20.7 %** | **0.9739** | **0.263 J** | **145.8 h** |

## How the stream was made

Deterministically generated (`saccade.stream.synthetic_walking_stream`, seed 0): a window
panning over a large static textured canvas, scripted into four regimes —

- `static` — camera still, world still (both methods should skip; fair region)
- `walk` — smooth translation across a static world (**predictable ego-motion**)
- `turn` — faster translation plus mild rotation (harder ego-motion)
- `event` — camera still, an independently moving object crosses (**genuine change**; every
  method must spend here)

## Limitations — please read

- **This is synthetic video, not natural footage.** It is a controlled apparatus for the
  steady-motion regime, chosen so the result is exactly reproducible. It is deliberately
  *not* evidence about natural scenes with parallax, motion blur, rolling shutter, exposure
  changes or non-rigid motion. Treat conclusions as being about the mechanism, not about
  wearable cameras in the wild.
- `fidelity` is patch-token cosine similarity against a Full encoder — a proxy for
  downstream task quality, not a task metric like VQA accuracy.
- Energy is an analytic FLOP model (`FLOPs × 1 pJ/FLOP`), not a power-meter measurement.
  Relative comparisons and the budget guarantee are sound; absolute Joules are model-dependent.
- `measurements` uses the `SyntheticBackbone` (a frozen random ViT-S/16-class patch encoder),
  chosen because each patch embedding is genuinely independent so subset encoding realizes
  real savings. Numbers on a real ViT differ; see the repo for DINOv2-small results.

## Reproduce

```bash
git clone https://github.com/NagaYu/saccade && cd saccade
pip install -r requirements.txt
python hf/export_dataset.py     # rebuilds every config from scratch
```

## License

MIT
