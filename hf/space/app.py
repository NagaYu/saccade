"""Saccade interactive demo (Hugging Face Space).

Lets anyone reproduce the three claims in the browser on free CPU hardware:
  省エネ    — watch the encoder workload collapse on predictable ego-motion,
  品質保持  — check fidelity against the Full encoder at the same time,
  予算保証  — tighten the Joule budget and see the controller hold the ceiling.

Runs entirely on the dependency-free SyntheticBackbone so a run takes seconds and
needs no model download.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import List, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from saccade import (EngineConfig, SaccadeEngine, SyntheticBackbone, estimate_motion,
                     summarize, synthetic_walking_stream, to_gray)
from saccade.gate import patch_residual
from saccade.kv_remap import remap_gray

GRID = (14, 14)
IMG = 224
SEG_COLORS = {"static": "#dfe7ef", "walk": "#ffe8c2", "turn": "#ffd0c2", "event": "#d6f5d6"}
COND_COLORS = {"Full": "#444444", "TemporalSim": "#e08214", "Saccade": "#2166ac"}


@lru_cache(maxsize=8)
def get_stream(seed: int, n_frames: int) -> Tuple[tuple, tuple]:
    frames, labels = synthetic_walking_stream(img_size=IMG, seed=seed)
    return tuple(frames[:n_frames]), tuple(labels[:n_frames])


@lru_cache(maxsize=1)
def get_backbone() -> SyntheticBackbone:
    return SyntheticBackbone(grid=GRID, img_size=IMG, embed_dim=384, seed=0)


def _fig_to_img(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _shade(ax, labels):
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            ax.axvspan(start, i, color=SEG_COLORS.get(labels[start], "#eee"), alpha=0.5, lw=0)
            ax.text((start + i) / 2, 101, labels[start], ha="center", va="top", fontsize=8,
                    color="#555")
            start = i


def run_benchmark(seed: int, n_frames: int, tau: float, budget_mw: float, fps: float,
                  motion_comp: bool, use_predictor: bool, budget_on: bool):
    """Run Full / TemporalSim / Saccade on one stream and report all three claims."""
    frames, labels = get_stream(int(seed), int(n_frames))
    frames, labels = list(frames), list(labels)
    bb = get_backbone()
    budget = (budget_mw / 1000.0) if budget_on else None

    res = {}
    for name, mode in [("Full", "full"), ("TemporalSim", "temporalsim"), ("Saccade", "saccade")]:
        cfg = EngineConfig(mode=mode, tau=tau, temporalsim_tau=tau,
                           budget_watts=(budget if mode == "saccade" else None), fps=fps,
                           img_size=IMG, motion_compensation=motion_comp,
                           use_predictor=use_predictor)
        res[name] = SaccadeEngine(bb, cfg, measure_fidelity=True, collect_masks=True).run(frames)

    warmup = min(6, max(0, len(frames) // 8))
    rows = []
    for name in ["Full", "TemporalSim", "Saccade"]:
        s = summarize(res[name], name, fps, warmup=warmup)
        rows.append([name, f"{s.mean_encoded_fraction*100:.1f}%", f"{s.mean_fidelity:.4f}",
                     f"{s.total_joules:.4f}", f"{s.mean_power_watts*1000:.1f}",
                     f"{s.runtime_hours_10kJ:.1f}", s.budget_violations])

    # per-frame workload
    fig, ax = plt.subplots(figsize=(10, 3.6))
    _shade(ax, labels)
    for name in ["Full", "TemporalSim", "Saccade"]:
        ax.plot([r.encoded_fraction * 100 for r in res[name]], label=name,
                color=COND_COLORS[name], lw=2 if name == "Saccade" else 1.4)
    ax.set_xlabel("frame"); ax.set_ylabel("patches re-encoded (%)"); ax.set_ylim(0, 105)
    ax.set_title("Encoder workload per frame")
    ax.legend(loc="center right"); ax.grid(alpha=0.25)
    workload = _fig_to_img(fig)

    # cumulative energy
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for name in ["Full", "TemporalSim", "Saccade"]:
        ax.plot([r.cum_joules for r in res[name]], label=name, color=COND_COLORS[name], lw=2)
    if budget is not None:
        t = np.arange(len(frames)) / fps
        ax.plot(np.arange(len(frames)), budget * t + 0.5 * budget, "--", color="crimson", lw=1.4,
                label="budget ceiling B·t+E₀")
    ax.set_xlabel("frame"); ax.set_ylabel("cumulative energy (J)")
    ax.set_title("Energy: Saccade stays under the ceiling")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    energy = _fig_to_img(fig)

    saved = 1 - (sum(r.frame_joules for r in res["Saccade"]) /
                 max(1e-12, sum(r.frame_joules for r in res["Full"])))
    vs_b = 1 - (sum(r.frame_joules for r in res["Saccade"]) /
                max(1e-12, sum(r.frame_joules for r in res["TemporalSim"])))
    viol = sum(0 if r.budget_ok else 1 for r in res["Saccade"])
    verdict = (f"**Saccade used {saved*100:.0f}% less energy than Full** and "
               f"**{vs_b*100:.0f}% less than TemporalSim**. "
               f"Budget violations: **{viol}**.")
    return rows, workload, energy, verdict


def explain_frame(seed: int, n_frames: int, frame_idx: int, tau: float):
    """Show the two competing decision signals on one frame — the core of the method."""
    frames, labels = get_stream(int(seed), int(n_frames))
    frames, labels = list(frames), list(labels)
    t = int(np.clip(frame_idx, 1, len(frames) - 1))
    g0, g1 = to_gray(frames[t - 1], IMG), to_gray(frames[t], IMG)
    m = estimate_motion(g0, g1, GRID)
    warped = remap_gray(g0, m.patch_disp, GRID)
    r_same = patch_residual(g1, g0, GRID).reshape(GRID)
    r_mc = patch_residual(g1, warped, GRID).reshape(GRID)
    b, c = (r_same > tau).mean() * 100, (r_mc > tau).mean() * 100
    vmax = float(max(r_same.max(), 0.15))

    fig, axs = plt.subplots(2, 2, figsize=(8.6, 7.6))
    axs[0, 0].imshow(frames[t - 1]); axs[0, 0].set_title("frame t-1"); axs[0, 0].axis("off")
    axs[0, 1].imshow(frames[t]); axs[0, 1].set_title(f"frame t  ({labels[t]})"); axs[0, 1].axis("off")
    im = axs[1, 0].imshow(r_same, cmap="inferno", vmin=0, vmax=vmax)
    axs[1, 0].set_title(f"TemporalSim: same-position residual\n→ re-encode {b:.0f}%", fontsize=10)
    axs[1, 0].axis("off")
    axs[1, 1].imshow(r_mc, cmap="inferno", vmin=0, vmax=vmax)
    axs[1, 1].set_title(f"Saccade: motion-compensated residual\n→ re-encode {c:.0f}%", fontsize=10)
    axs[1, 1].axis("off")
    fig.colorbar(im, ax=axs[1, :], fraction=0.046, pad=0.04, label="per-patch residual")
    img = _fig_to_img(fig)
    note = (f"Segment **{labels[t]}** · ego-motion (median) **{m.mag_median:.3f}** patches/frame. "
            f"Same-position gate would re-encode **{b:.0f}%** of patches; "
            f"motion-compensated gate **{c:.0f}%**.")
    return img, note


with gr.Blocks(title="Saccade — always-on edge VLM") as demo:
    gr.Markdown(
        """
        # Saccade 👁️⚡ — prediction-error gating + energy budgets for an always-on edge VLM

        Don't ask *"did the pixels change?"* — ask **"did anything change that ego-motion cannot explain?"**

        A moving camera changes almost every pixel, so frame-similarity skipping cannot thin the
        workload at all during steady motion. Saccade compensates the motion first, so a static
        world under ego-motion produces a near-zero residual and is skipped wholesale — while
        genuinely new events are still caught.

        [📄 Code](https://github.com/NagaYu/saccade) ·
        [🤖 Model](https://huggingface.co/NagaYu/saccade-predictor) ·
        [📊 Dataset](https://huggingface.co/datasets/NagaYu/saccade-egomotion-bench)
        """
    )

    with gr.Tab("Benchmark (A / B / C)"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Slider(0, 20, value=0, step=1, label="stream seed")
                n_frames = gr.Slider(48, 144, value=96, step=12, label="frames")
                tau = gr.Slider(0.01, 0.20, value=0.06, step=0.005,
                                label="τ — surprisal threshold (higher = skip more)")
                budget_on = gr.Checkbox(value=True, label="energy budget controller ON")
                budget_mw = gr.Slider(1, 100, value=20, step=1,
                                      label="energy budget B (mW)")
                fps = gr.Slider(5, 30, value=10, step=1, label="fps")
                motion_comp = gr.Checkbox(value=True, label="motion compensation (pseudo-IMU)")
                use_pred = gr.Checkbox(value=True, label="forward predictor")
                run_btn = gr.Button("Run benchmark", variant="primary")
            with gr.Column(scale=2):
                verdict = gr.Markdown()
                table = gr.Dataframe(
                    headers=["method", "re-encoded", "fidelity vs Full", "energy (J)",
                             "mean power (mW)", "hours @10kJ", "budget violations"],
                    interactive=False, wrap=True)
                workload_img = gr.Image(label="Encoder workload per frame", type="pil")
                energy_img = gr.Image(label="Cumulative energy", type="pil")
        gr.Markdown(
            "Try: set **motion compensation OFF** and watch Saccade collapse toward TemporalSim — "
            "that single switch is the whole idea. Or drag the budget down to 3 mW and confirm "
            "**budget violations stays 0** while quality degrades gracefully (the anytime guarantee)."
        )
        run_btn.click(run_benchmark,
                      [seed, n_frames, tau, budget_mw, fps, motion_comp, use_pred, budget_on],
                      [table, workload_img, energy_img, verdict])

    with gr.Tab("Why it works"):
        with gr.Row():
            e_seed = gr.Slider(0, 20, value=0, step=1, label="stream seed")
            e_frames = gr.Slider(48, 144, value=144, step=12, label="frames")
            e_idx = gr.Slider(1, 143, value=70, step=1, label="frame index")
            e_tau = gr.Slider(0.01, 0.20, value=0.06, step=0.005, label="τ")
        e_note = gr.Markdown()
        e_img = gr.Image(label="Two competing decision signals on the same frame", type="pil")
        e_btn = gr.Button("Explain this frame", variant="primary")
        e_btn.click(explain_frame, [e_seed, e_frames, e_idx, e_tau], [e_img, e_note])
        gr.Markdown(
            "Move the frame index into a **walk** or **turn** segment (roughly frames 12–75 and "
            "104–136 at seed 0): the left map lights up everywhere while the right map stays dark. "
            "Same frames, same threshold — only the reference differs."
        )

    demo.load(run_benchmark,
              [seed, n_frames, tau, budget_mw, fps, motion_comp, use_pred, budget_on],
              [table, workload_img, energy_img, verdict])

if __name__ == "__main__":
    demo.launch()
