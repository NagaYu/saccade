---
title: Saccade — Always-On Edge VLM
emoji: 👁️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Energy-gated edge vision via ego-motion prediction
tags:
  - edge-ai
  - energy-efficiency
  - video
  - vision-transformer
---

# Saccade 👁️⚡

Interactive demo for [**Saccade**](https://github.com/NagaYu/saccade): an always-on edge VLM
that spends encoder compute only where a forward prediction was wrong, under a hard energy
budget.

Don't ask *"did the pixels change?"* — ask **"did anything change that ego-motion cannot explain?"**

## What you can try

- **Sweep τ and the energy budget** and watch all three conditions (Full / TemporalSim /
  Saccade) update instantly.
- **Turn motion compensation off** — Saccade collapses toward TemporalSim. That one switch
  is the whole idea.
- **Scrub through frames** to see exactly which patches each method re-encodes.
- **Compare the two decision signals** on any frame: in a walking segment the same-position
  residual is hot everywhere while the motion-compensated residual stays near zero.

## Why this is a static Space

Every configuration in the parameter grid was precomputed offline with the same code the
repository's 30 tests exercise, then shipped as JSON. So there is no backend, no queue and no
cold start — and the numbers are exactly reproducible from the repo.

## Honest notes

- The stream is **deterministic synthetic ego-motion**, not natural video — a controlled
  apparatus so the steady-motion claim is exactly reproducible.
- Energy is an **analytic FLOP model** (`FLOPs × 1 pJ/FLOP`), not a power meter.
- Results use the dependency-free `SyntheticBackbone`. On a real ViT (DINOv2-small) the same
  machinery runs but degrades faster at low budgets — see the repo.

## Links

- 💻 Code: https://github.com/NagaYu/saccade
- 🤖 Model: https://huggingface.co/NagaYu/saccade-predictor
- 📊 Dataset: https://huggingface.co/datasets/NagaYu/saccade-egomotion-bench
