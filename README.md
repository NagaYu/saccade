# Saccade 👁️⚡

[![ci](https://github.com/NagaYu/saccade/actions/workflows/ci.yml/badge.svg)](https://github.com/NagaYu/saccade/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![release](https://img.shields.io/github/v/release/NagaYu/saccade)](https://github.com/NagaYu/saccade/releases)

**An always-on edge VLM driven by prediction-error gating and an energy budget.**
*Predict what the next frame will look like; spend compute only where you were wrong; never exceed your energy budget.*

Saccade borrows a trick from biological vision — the world stays smoothly predictable while your head moves — and cuts the power draw of an always-on edge VLM by **never re-encoding a patch it could predict**. The key is *ego-motion compensation*: when the camera moves, almost every pixel changes, but **subtract the motion and the residual of a static world is nearly zero**. Existing "frame-similarity" skipping cannot skip steady motion at all; Saccade skips most of it.

---

## TL;DR (synthetic walking stream, CPU, ViT-S/16-class encoder)

| Method | Patches re-encoded | Fidelity vs Full | Estimated energy (J) | Mean power | **Runtime on 10 kJ** |
|---|---|---|---|---|---|
| **(A) Full** (encode everything, every frame) | 100.0% | 1.000 | 1.264 | 91.6 mW | 30.3 h |
| **(B) TemporalSim** (same-position frame-diff skipping) | 33.9% | 0.972 | 0.430 | 31.2 mW | 89.1 h |
| **(C) Saccade** (prediction-error gate + IMU compensation + budget control) | **20.6%** | **0.971** | **0.262** | **19.0 mW** | **146.4 h** |

- **Energy saving**: 4.8× less compute than Full and 1.65× less than TemporalSim,
- **Quality preservation**: quality on par with TemporalSim (fidelity vs Full: 0.971 ≈ 0.972),
- **Budget guarantee**: zero budget violations with the controller on (the strict invariant holds for every budget from 0.05 W down to 0.0001 W).

> Numbers are stored in `benchmarks/results.json`. Energy comes from an analytic FLOP model × `1 pJ/FLOP` (a reasonable figure for efficient mobile inference; change it with `--j-per-flop`).

---

## The centerpiece figure: why temporal similarity can't skip steady motion

![why temporal-similarity fails](figures/why_temporalsim_fails.png)

Two consecutive frames from the walking segment. The camera has merely panned — **the world is static** — yet:
- **Bottom-left (B's decision signal)**: the *same-position* residual is high almost everywhere → **re-encode 36% of patches**.
- **Bottom-right (C's decision signal)**: the *motion-compensated* residual is near zero → **re-encode 0% of patches**.

This is the heart of Saccade: don't ask "did the pixels change?" — ask "**did anything change that motion cannot explain?**"

### Per-frame encoder workload
![encoded fraction over time](figures/encoded_fraction.png)

During `walk`/`turn` (predictable ego-motion) segments, **B is stuck re-encoding 42–64%** while **C thins the load to ~14%**. On `static` (no motion) and `event` (an independently moving object = genuine change) segments the two behave almost identically — i.e. C **still catches real change** (no cheating).

### Quality / energy trade-off
![quality vs energy](figures/quality_energy.png)

Just by varying the budget B, Saccade sweeps a Pareto frontier that **dominates** TemporalSim's single operating point (same quality at lower energy, or higher quality at the same energy).

### Webcam-style demo (B vs C; red = patches re-encoded this frame)
![demo](figures/saccade_demo.gif)

---

## Architecture

```mermaid
flowchart TB
  F["frame<br/>(webcam / mp4 / synthetic)"] --> G[grayscale]
  G --> OF["optical flow<br/><b>pseudo-IMU</b> (motion.py)"]
  CACHE[("PatchEmbeddingCache<br/>previous-frame tokens")]

  OF -- "per-patch displacement" --> KV["<b>kv_remap.py</b><br/>remap cached tokens along ego-motion"]
  CACHE --> KV
  KV -- "warped cache" --> PRED["<b>predictor.py</b><br/>GRU predicts next-frame embeddings"]
  OF -- "ego-motion features" --> PRED

  OF -- "motion-compensated residual (cheap)" --> GATE["<b>gate.py</b><br/>surprisal gate τ + task-aware"]
  IMP["importance<br/>attention × error"] --> GATE
  GATE -- "candidate encode set" --> CAP{"<b>energy_controller.py</b><br/>budget caps the encode count"}
  BUD["token-bucket budget<br/>B [J/s]"] --> CAP

  CAP -- "surprising patches only" --> ENC["ViT patch encoder<br/>(backbone.py)"]
  ENC -- "fresh tokens" --> MIX["splice:<br/>fresh + predicted"]
  PRED -- "predicted tokens" --> MIX
  MIX --> OUT["reconstructed<br/>patch-token map → LLM"]
  MIX -- "self-supervised update on observed patches" --> PRED
  MIX --> CACHE
  ENC -. "FLOPs actually spent" .-> BUD

  classDef core fill:#e8f0fe,stroke:#2166ac,stroke-width:2px;
  class KV,PRED,GATE,CAP core;
```

*The four cores (blue)*: `predictor` (forward prediction), `gate` (surprisal gating), `kv_remap` (motion-compensated reuse), `energy_controller` (budget control).
`engine.py` integrates them every frame and produces all three conditions — `full` / `temporalsim` / `saccade` — from **one engine**, so the comparison differs only in the mechanism under test.

---

## Claims → code (every function's docstring states which claim it demonstrates)

Docstrings tag claims with their Japanese research labels: 省エネ = energy saving, 品質保持 = quality preservation, 予算保証 = budget guarantee.

| Claim | Where it is demonstrated |
|---|---|
| **Energy saving** | `gate.py` (encode only the surprising few), `kv_remap.py` (motion compensation cuts recomputation), `flops.py` (FLOP→J is monotone in encode count) |
| **Quality preservation** | `predictor.py` (predicts values for un-encoded patches), fidelity in `engine.py` (cosine of reconstruction vs Full), the gate's task-aware/forced paths |
| **Budget guarantee** | `energy_controller.py` (token bucket enforces `cum_J ≤ B·t + E0` by construction; on starvation, frame-drop yields an anytime degraded output) |

---

## The four cores in detail

1. **PatchEmbeddingCache + forward predictor** — `predictor.py`, `engine.py`
   Every patch embedding is cached. A tiny GRU (weights shared across patches) predicts each patch's next-frame embedding from the *motion-compensated cache + pseudo-IMU*.
   **Fully self-supervised**: it trains online on `||pred − true||` using the true embeddings of patches the gate chose to encode anyway. No labels.
   The prediction is a *residual on top of the warped cache*, so learning starts from identity (residual 0) and only has to model what ego-motion cannot explain.

2. **Surprisal gate** — `gate.py`
   Per patch, threshold the *motion-compensated appearance residual* (a cheap pixel-space proxy) at τ; re-encode only patches above it, reuse cached KV for the rest.
   *Task-aware variant*: `importance = EMA(recent embedding surprisal)` (a stand-in for "attention × error" in this CPU setting without LLM attention) corrects the threshold as `τ_eff = τ/(1+λ·w)`, lowering the bar for important patches.
   (Chicken-and-egg: true embedding surprisal is only known *after* encoding, so decisions use the cheap proxy while learning/importance updates use the true values of encoded patches — see the docstrings.)

3. **Motion-compensated KV reuse** — `kv_remap.py`
   Bilinearly remap cached tokens to their new spatial positions along the pseudo-IMU displacement *before* reuse (avoiding wasted recomputation from positional misalignment). Newly revealed patches at the frame edge are invalid and force-encoded.
   **Measured effect**: motion compensation cuts total recomputed patches by **~3.3×**.

4. **Energy budget controller** — `energy_controller.py`
   Simple energy model: `J = (encoder + LLM-prefill FLOPs) × [J/FLOP]`. Given a budget `B [J/s]`:
   **(1) soft loop** — proportional control adapts τ so average power tracks B (spend the budget you have; maximize utility);
   **(2) hard cap** — a token bucket (capacity `E0 = reserve·B`, refill `B/fps` per frame) enforces
   `cumulative energy ≤ B·t + E0` for all t **by construction**. When the bucket runs dry the frame is dropped entirely and the previous cache is returned as an **anytime degraded output** (there is always a valid output).

---

## Quick start

```bash
pip install -r requirements.txt

# benchmark (synthetic walking stream; writes figures, the GIF, and results.json)
python benchmarks/run.py

# change the budget / fps
python benchmarks/run.py --budget 0.01 --fps 15

# run on a real webcam or an mp4
python benchmarks/run.py --source webcam
python benchmarks/run.py --source path/to/walk.mp4

# run on real ViT/DINOv2 patch tokens (requires transformers)
python benchmarks/run.py --backbone hf:facebook/dinov2-small

# tests (gate-on quality, zero budget violations, motion comp reduces recompute)
pytest -q
```

### The small-VLM wrapper
`saccade/backbone.py` provides a thin wrapper that intervenes at the **patch embeddings** of a ViT-class vision encoder:

- `SyntheticBackbone` — dependency-free, deterministic, and **genuinely capable of encoding a subset of patches** (a frozen-random ViT-S/16 stand-in). Default for tests and the reproducible benchmark.
- `HFVisionBackbone` — exposes the patch tokens of a real HuggingFace ViT/DINOv2 (e.g. `facebook/dinov2-small`). Because of self-attention, subset encoding is an *accounting* approximation (token-caching; see `flops.py`).
  **Verified**: the full pipeline runs end-to-end on `facebook/dinov2-small` (grid 16×16, D=384; 44-frame short stream; measurements in [benchmarks/results_dinov2.json](benchmarks/results_dinov2.json)). On real DINOv2 features the budget sweep still traces a quality/energy Pareto: against TemporalSim's operating point (0.130 J, fidelity 0.834), **Saccade@0.05 W reaches fidelity 0.900 at 0.149 J** (+0.066 in the same energy band), and @0.1 W reaches 0.971 at 0.303 J (35% less energy than Full's 0.464 J at 97% quality). Zero budget violations in every setting.
  Honest caveat, also measured: real features mix context through self-attention, so cache-reuse error is larger than with the synthetic encoder and **low-budget quality degrades faster** (0.065 J → fidelity 0.719). Low-budget operation on semantic features is future work.
  Vision towers of VLMs such as moondream2 / SmolVLM can be swapped in through the same API (patch-token exposure).

> **Why is the default synthetic?** The algorithms (predictive gating / motion compensation / budget control) are backbone-agnostic. With the synthetic encoder each patch is an independent function of its pixels, so **subset encoding realizes the savings as real computation** — the energy claim is measured without approximation. The real-ViT path shows the same machinery running on genuine semantics (best for checking quality preservation).

---

## Repository layout

```
saccade/
  backbone.py          # ViT patch-embedding wrapper (synthetic / HF)
  motion.py            # optical flow = pseudo-IMU (per-patch displacement + ego-motion features)
  predictor.py         # forward predictor (shared-weight GRU, online self-supervised)
  kv_remap.py          # motion-compensated spatial remap of the cache
  gate.py              # surprisal gate (proxy decision + task-aware + importance)
  energy_controller.py # token-bucket budget control (proportional τ + anytime frame-drop)
  engine.py            # the always-on loop integrating the four cores + full/temporalsim/saccade
  flops.py             # analytic FLOP/energy model
  stream.py            # frame sources (synthetic walking / webcam / mp4)
  types.py             # EngineConfig / FrameResult / RunSummary
benchmarks/
  run.py               # A/B/C comparison, figures, GIF, results.json
  results.json         # latest measurements
figures/               # generated artifacts (PNGs + demo GIF)
tests/                 # pytest (verifies energy saving / quality preservation / budget guarantee)
```

---

## Honest caveats

- **Energy is an analytic model**: estimated as `FLOPs × [J/FLOP]`, not measured Joules. The relative comparison (A/B/C) and the budget-guarantee logic are sound; absolute values depend on the coefficient (`--j-per-flop`).
- **Token-caching approximation**: in a real ViT, self-attention mixes patch tokens, so subset-encode savings are accounted as "recompute only that token, reuse cached KV for the rest". Realizing this on hardware needs block-sparse attention (this prototype measures *decision quality* and *modelled energy*). The synthetic backbone realizes the savings as real computation.
- **Synthetic stream**: the default input is a deterministic synthetic "walking" video (explicitly labelled as such), chosen to evaluate the steady-motion regime reproducibly. Real webcam/mp4 inputs run through the identical pipeline.
- **No reliance on server-side long-video QA**: nothing here depends on or imitates V-Rex / StreamingTOM-style long-form video QA. The focus is the always-on edge setting.
- **Quantization/NPU deferred**: correctness on CPU with small models comes first (everything in this README and the tests reproduces on CPU).

---

## License
MIT — see [LICENSE](LICENSE). Research prototype.
