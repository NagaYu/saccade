"""Saccade benchmark: compare Full / TemporalSim / Saccade on one stream.

Claim demonstrated: all three, quantitatively — this is the "目玉" (centerpiece). It
produces the metrics table, the figures (including the one showing *why*
temporal-similarity cannot skip steady motion), and the webcam-style demo GIF.

Usage
-----
    python benchmarks/run.py                         # synthetic walking stream (default)
    python benchmarks/run.py --source webcam         # live webcam (index 0)
    python benchmarks/run.py --source path/to.mp4    # an mp4 file
    python benchmarks/run.py --backbone hf:facebook/dinov2-small   # real ViT patch tokens
    python benchmarks/run.py --budget 0.02           # J/s energy budget for Saccade

Outputs go to ``figures/`` and ``benchmarks/results.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from saccade import (EngineConfig, SaccadeEngine, SyntheticBackbone, make_backbone,  # noqa: E402
                     summarize, synthetic_walking_stream, frames_from_video,
                     to_gray, estimate_motion)
from saccade.gate import patch_residual  # noqa: E402
from saccade.kv_remap import remap_gray  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(ROOT, "figures")

SEG_COLORS = {"static": "#dfe7ef", "walk": "#ffe8c2", "turn": "#ffd0c2", "event": "#d6f5d6",
              "live": "#eeeeee"}
COND_COLORS = {"Full": "#444444", "TemporalSim": "#e08214", "Saccade": "#2166ac"}


# --------------------------------------------------------------------------- io
def build_stream(source: str, img_size: int, n_frames: int) -> Tuple[List[np.ndarray], List[str]]:
    if source == "synthetic":
        frames, labels = synthetic_walking_stream(img_size=img_size, seed=0)
        return frames[:n_frames], labels[:n_frames]
    if source == "webcam":
        return frames_from_video(0, n_frames=n_frames, img_size=img_size)
    return frames_from_video(source, n_frames=n_frames, img_size=img_size, stride=2)


def run_condition(mode: str, frames, backbone, *, budget, fps, tau, img_size,
                  collect_masks=False) -> List:
    cfg = EngineConfig(mode=mode, tau=tau, temporalsim_tau=tau,
                       budget_watts=(budget if mode == "saccade" else None),
                       fps=fps, img_size=img_size)
    eng = SaccadeEngine(backbone, cfg, measure_fidelity=True, collect_masks=collect_masks)
    return eng.run(frames)


# ---------------------------------------------------------------------- figures
def _shade_segments(ax, labels):
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            ax.axvspan(start, i, color=SEG_COLORS.get(labels[start], "#eee"), alpha=0.5, lw=0)
            ax.text((start + i) / 2, ax.get_ylim()[1] * 0.96, labels[start], ha="center",
                    va="top", fontsize=8, color="#555")
            start = i


def fig_encoded_fraction(res: Dict[str, List], labels, warmup, path):
    """Centerpiece: per-frame encoded-patch fraction. B stuck high on walk/turn, C low."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    _shade_segments(ax, labels)
    for name, results in res.items():
        y = [r.encoded_fraction * 100 for r in results]
        ax.plot(y, label=name, color=COND_COLORS[name], lw=2 if name == "Saccade" else 1.5,
                alpha=0.95)
    ax.axvspan(0, warmup, color="#cccccc", alpha=0.35, lw=0)
    ax.text(warmup / 2, 50, "warm-up", rotation=90, va="center", ha="center", fontsize=8, color="#666")
    ax.set_xlabel("frame"); ax.set_ylabel("patches re-encoded (%)")
    ax.set_title("Per-frame encoder workload — Saccade skips predictable ego-motion that TemporalSim cannot")
    ax.set_ylim(0, 105); ax.legend(loc="center right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_energy_vs_time(res, path, fps):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, results in res.items():
        ax.plot([r.cum_joules for r in results], label=name, color=COND_COLORS[name], lw=2)
    ax.set_xlabel("frame"); ax.set_ylabel("cumulative energy (J)")
    ax.set_title("Cumulative estimated energy")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_runtime(summaries, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    names = list(summaries.keys())
    vals = [summaries[n].runtime_hours_10kJ for n in names]
    bars = ax.bar(names, vals, color=[COND_COLORS[n] for n in names])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f} h", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("estimated continuous runtime (hours)")
    ax.set_title("How long can it run on a 10 kJ envelope?")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_quality_energy(res, summaries, sweep, path):
    """Quality vs energy Pareto: Saccade budget sweep dominates the baselines."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name in ["Full", "TemporalSim"]:
        s = summaries[name]
        ax.scatter(s.total_joules, s.mean_fidelity, s=110, color=COND_COLORS[name], zorder=3,
                   label=name, edgecolor="k")
    if sweep:
        xs = [s.total_joules for s in sweep]
        ys = [s.mean_fidelity for s in sweep]
        ax.plot(xs, ys, "-o", color=COND_COLORS["Saccade"], label="Saccade (budget sweep)", zorder=2)
        for s in sweep:
            ax.annotate(f"{s.mean_power_watts*1000:.1f} mW", (s.total_joules, s.mean_fidelity),
                        fontsize=7, xytext=(4, -8), textcoords="offset points", color="#2166ac")
    ax.set_xlabel("energy over stream (J)  —  lower is better")
    ax.set_ylabel("fidelity vs Full encoder  —  higher is better")
    ax.set_title("Quality / energy trade-off (up-and-left is better)")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_why_temporalsim_fails(frames, labels, img_size, path):
    """The single explanatory figure: same-position residual vs motion-compensated residual.

    On a walking (predictable ego-motion) frame pair, the same-position per-patch
    residual (what TemporalSim thresholds) is high everywhere, so it must re-encode
    almost everything; the motion-compensated residual (what Saccade thresholds) is
    near zero, so Saccade skips. This is the crux of the whole method.
    """
    grid = (14, 14)
    # pick a mid "walk" frame
    walk_idx = [i for i, l in enumerate(labels) if l == "walk"]
    t = walk_idx[len(walk_idx) // 2]
    g0 = to_gray(frames[t - 1], img_size)
    g1 = to_gray(frames[t], img_size)
    motion = estimate_motion(g0, g1, grid)
    g0_warp = remap_gray(g0, motion.patch_disp, grid)
    res_same = patch_residual(g1, g0, grid).reshape(grid)
    res_mc = patch_residual(g1, g0_warp, grid).reshape(grid)
    tau = 0.06
    b_enc = (res_same > tau).mean() * 100
    c_enc = (res_mc > tau).mean() * 100
    vmax = float(max(res_same.max(), 0.15))

    fig, axs = plt.subplots(2, 2, figsize=(9.5, 8.2))
    axs[0, 0].imshow(frames[t - 1]); axs[0, 0].set_title("frame t-1"); axs[0, 0].axis("off")
    axs[0, 1].imshow(frames[t]); axs[0, 1].set_title("frame t (camera has panned)"); axs[0, 1].axis("off")
    im2 = axs[1, 0].imshow(res_same, cmap="inferno", vmin=0, vmax=vmax)
    axs[1, 0].set_title(f"TemporalSim: same-position residual\n→ re-encode {b_enc:.0f}% of patches",
                        fontsize=11)
    axs[1, 0].axis("off")
    axs[1, 1].imshow(res_mc, cmap="inferno", vmin=0, vmax=vmax)
    axs[1, 1].set_title(f"Saccade: motion-compensated residual\n→ re-encode {c_enc:.0f}% of patches",
                        fontsize=11)
    axs[1, 1].axis("off")
    fig.colorbar(im2, ax=axs[1, :], fraction=0.046, pad=0.04, label="per-patch residual")
    fig.suptitle("Why temporal-similarity can't skip steady ego-motion", fontsize=13)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return b_enc, c_enc


def make_gif(frames, res_b, res_c, labels, path, img_size, stride=2, max_frames=70):
    """Webcam-style demo GIF: B vs C recompute overlays side by side. Claim: 省エネ (visual)."""
    import cv2
    from PIL import Image
    gif_frames = []

    def overlay(frame, mask, title, enc_pct, cumj):
        img = frame.copy()
        if mask is not None:
            gh, gw = mask.shape          # grid comes from the engine's mask, not hardcoded
            ph, pw = img_size // gh, img_size // gw
            heat = np.zeros_like(img)
            for i in range(gh):
                for j in range(gw):
                    if mask[i, j]:
                        heat[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = (255, 40, 40)
            img = cv2.addWeighted(img, 1.0, heat, 0.35, 0)
        pad = np.full((44, img_size, 3), 245, np.uint8)
        img = np.vstack([pad, img])
        cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(img, f"encode {enc_pct:4.0f}%   {cumj*1000:6.1f} mJ", (6, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 30, 30), 1, cv2.LINE_AA)
        return img

    for i in range(0, len(frames), stride):
        if len(gif_frames) >= max_frames:
            break
        rb, rc = res_b[i], res_c[i]
        left = overlay(frames[i], rb.encoded_mask, "TemporalSim", rb.encoded_fraction * 100, rb.cum_joules)
        right = overlay(frames[i], rc.encoded_mask, "Saccade", rc.encoded_fraction * 100, rc.cum_joules)
        sep = np.full((left.shape[0], 4, 3), 255, np.uint8)
        combo = np.hstack([left, sep, right])
        band = np.full((22, combo.shape[1], 3), 255, np.uint8)
        cv2.putText(band, f"segment: {labels[i]}   (red = patches re-encoded this frame)", (6, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
        combo = np.vstack([combo, band])
        gif_frames.append(Image.fromarray(combo))
    gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:], duration=120, loop=0)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Saccade A/B/C benchmark")
    ap.add_argument("--source", default="synthetic", help="synthetic | webcam | <path.mp4>")
    ap.add_argument("--backbone", default="synthetic", help="synthetic | hf[:model_name]")
    ap.add_argument("--budget", type=float, default=0.02, help="Saccade energy budget (J/s)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--tau", type=float, default=0.06)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--j-per-flop", type=float, default=1e-12,
                    help="energy coefficient [J/FLOP] for the analytic model (default 1 pJ/FLOP)")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip PNG figures (e.g. when benchmarking a secondary backbone)")
    ap.add_argument("--out", default="results.json",
                    help="output metrics filename inside benchmarks/ (default results.json)")
    args = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    print(f"[stream] source={args.source}")
    frames, labels = build_stream(args.source, args.img_size, args.frames)
    print(f"[stream] {len(frames)} frames")

    if args.backbone == "synthetic":
        backbone = SyntheticBackbone(grid=(args.img_size // 16, args.img_size // 16),
                                     img_size=args.img_size, embed_dim=384,
                                     j_per_flop=args.j_per_flop)
    else:
        print(f"[backbone] loading {args.backbone} ...")
        backbone = make_backbone(args.backbone, img_size=args.img_size,
                                 j_per_flop=args.j_per_flop)
        args.img_size = backbone.img_size
        print(f"[backbone] grid={backbone.grid} D={backbone.embed_dim}")

    res: Dict[str, List] = {}
    for label, mode in [("Full", "full"), ("TemporalSim", "temporalsim"), ("Saccade", "saccade")]:
        print(f"[run] {label} ...")
        res[label] = run_condition(mode, frames, backbone, budget=args.budget, fps=args.fps,
                                    tau=args.tau, img_size=args.img_size, collect_masks=True)

    summaries = {k: summarize(v, k, args.fps, warmup=args.warmup) for k, v in res.items()}

    # Saccade budget sweep for the quality/energy Pareto
    sweep = []
    for b in [0.005, 0.01, 0.02, 0.05, 0.1]:
        r = run_condition("saccade", frames, backbone, budget=b, fps=args.fps, tau=args.tau,
                          img_size=args.img_size)
        sweep.append(summarize(r, f"saccade@{b}", args.fps, warmup=args.warmup))

    # ---- figures ----
    b_enc = c_enc = None
    if not args.no_figures:
        print("[figures] writing ...")
        fig_encoded_fraction(res, labels, args.warmup, os.path.join(FIGDIR, "encoded_fraction.png"))
        fig_energy_vs_time(res, os.path.join(FIGDIR, "energy_vs_time.png"), args.fps)
        fig_runtime(summaries, os.path.join(FIGDIR, "runtime_10kJ.png"))
        fig_quality_energy(res, summaries, sweep, os.path.join(FIGDIR, "quality_energy.png"))
        if args.source == "synthetic":
            b_enc, c_enc = fig_why_temporalsim_fails(frames, labels, args.img_size,
                                                     os.path.join(FIGDIR, "why_temporalsim_fails.png"))
    if not (args.no_gif or args.no_figures):
        make_gif(frames, res["TemporalSim"], res["Saccade"], labels,
                 os.path.join(FIGDIR, "saccade_demo.gif"), args.img_size)

    # ---- results.json + markdown table ----
    out = {
        "config": vars(args),
        "conditions": {k: v.as_row() for k, v in summaries.items()},
        "saccade_budget_sweep": [s.as_row() for s in sweep],
        "why_temporalsim_fails": {"walk_frame_same_pos_encode_pct": b_enc,
                                  "walk_frame_motion_comp_encode_pct": c_enc},
    }
    out_path = os.path.join(HERE, os.path.basename(args.out))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    cols = ["label", "encoded_%", "fidelity", "GFLOPs", "Joules", "mean_W", "hrs@10kJ", "budget_viol"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for k in ["Full", "TemporalSim", "Saccade"]:
        row = summaries[k].as_row()
        print("| " + " | ".join(str(row[c]) for c in cols) + " |")
    print(f"\n[done] figures in {FIGDIR}, metrics in {out_path}")


if __name__ == "__main__":
    main()
