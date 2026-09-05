---
title: Saccade — Always-On Edge VLM
emoji: 👁️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Re-encode only what ego-motion can't explain — energy-gated edge vision
tags:
  - edge-ai
  - energy-efficiency
  - video
  - vision-transformer
---

# Saccade 👁️⚡ (Gradio version)

> **Note**: Hugging Face now requires a PRO subscription to host Gradio Spaces on
> `cpu-basic`, so the **deployed** demo is the free static build in
> [`hf/space_static/`](../space_static) → https://huggingface.co/spaces/NagaYu/saccade
>
> This directory keeps the fully-working Gradio app for anyone who wants to run it
> locally (`python app.py`) or deploy it on paid hardware. It runs the engine live
> instead of replaying a precomputed sweep.


Interactive demo for [**Saccade**](https://github.com/NagaYu/saccade): an always-on edge VLM
that spends encoder compute only where a forward prediction was wrong, under a hard energy
budget.

Don't ask *"did the pixels change?"* — ask **"did anything change that ego-motion cannot
explain?"**

## What you can try

**Benchmark tab** — run all three conditions on the same stream and compare:

- **Full** — re-encode every patch, every frame (the quality anchor)
- **TemporalSim** — per-patch same-position frame differencing (the conventional skip rule)
- **Saccade** — motion-compensated surprisal gating + forward predictor + energy budget

Two switches are worth flipping:

- Turn **motion compensation off** and watch Saccade collapse toward TemporalSim. That single
  switch is the whole idea.
- Drag the budget down to a few mW and confirm **budget violations stays 0** while quality
  degrades gracefully — the anytime guarantee, enforced by a token bucket rather than by hope.

**Why it works tab** — pick any frame and see the two competing decision signals side by side.
In a walking segment the same-position residual is hot everywhere (so TemporalSim must
re-encode ~60 % of patches) while the motion-compensated residual is nearly black (so Saccade
re-encodes ~1 %).

## Honest notes

- The stream is **deterministic synthetic ego-motion**, not natural video — a controlled
  apparatus so the steady-motion claim is exactly reproducible.
- Energy is an **analytic FLOP model** (`FLOPs × 1 pJ/FLOP`), not a power meter. Relative
  comparisons and the budget guarantee are sound; absolute Joules depend on that coefficient.
- The demo uses the dependency-free `SyntheticBackbone` so it runs in seconds on free CPU.
  Results on a real ViT (DINOv2-small) are in the repo and are less flattering at low budgets.

## Links

- 💻 Code: https://github.com/NagaYu/saccade
- 🤖 Model: https://huggingface.co/NagaYu/saccade-predictor
- 📊 Dataset: https://huggingface.co/datasets/NagaYu/saccade-egomotion-bench
