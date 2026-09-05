"""Precompute everything the static Saccade Space needs.

A static Space has no backend, so instead of running the engine on demand we sweep the
parameter grid offline and ship the results. The browser then explores them instantly:
no queue, no cold start, and the numbers are exactly the ones the repo's tests reproduce.

Outputs into hf/space_static/:
  sprite.jpg  — all frames in one sheet (JS draws sub-rectangles)
  data.json   — metrics + per-frame re-encode masks + per-patch residual maps
"""
from __future__ import annotations
import base64, json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from saccade import (EngineConfig, SaccadeEngine, SyntheticBackbone, estimate_motion,
                     summarize, synthetic_walking_stream, to_gray)
from saccade.gate import patch_residual
from saccade.kv_remap import remap_gray

HERE = os.path.dirname(os.path.abspath(__file__))
GRID, IMG, FPS, WARMUP = (14, 14), 224, 10.0, 6
THUMB = 160
TAUS = [0.03, 0.06, 0.10, 0.15]
BUDGETS_MW = [0, 5, 10, 20, 50]          # 0 = controller off
MCS = [True, False]


def pack(mask_bool: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask_bool.astype(np.uint8))).decode()


def run(bb, frames, mode, tau, budget=None, mc=True):
    cfg = EngineConfig(mode=mode, tau=tau, temporalsim_tau=tau, budget_watts=budget,
                       fps=FPS, img_size=IMG, motion_compensation=mc)
    res = SaccadeEngine(bb, cfg, measure_fidelity=True, collect_masks=True).run(frames)
    s = summarize(res, mode, FPS, warmup=WARMUP)
    masks = np.stack([(r.encoded_mask.reshape(-1) if r.encoded_mask is not None
                       else np.zeros(196, bool)) for r in res])
    return {
        "enc": [round(r.encoded_fraction, 4) for r in res],
        "fid": [round(float(r.fidelity), 4) for r in res],
        "cumJ": [round(r.cum_joules, 6) for r in res],
        "masks": pack(masks),
        "summary": {"enc": round(s.mean_encoded_fraction, 4), "fid": round(s.mean_fidelity, 4),
                    "J": round(s.total_joules, 5), "mW": round(s.mean_power_watts * 1000, 2),
                    "hrs": round(s.runtime_hours_10kJ, 1), "viol": s.budget_violations},
    }


def main():
    frames, labels = synthetic_walking_stream(img_size=IMG, seed=0)
    bb = SyntheticBackbone(grid=GRID, img_size=IMG, embed_dim=384, seed=0)
    n = len(frames)
    print(f"[stream] {n} frames")

    cols = 12
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * THUMB, rows * THUMB))
    for i, f in enumerate(frames):
        sheet.paste(Image.fromarray(f).resize((THUMB, THUMB), Image.LANCZOS),
                    ((i % cols) * THUMB, (i // cols) * THUMB))
    sheet.save(os.path.join(HERE, "sprite.jpg"), quality=82, optimize=True)
    print(f"[sprite] {sheet.size} -> {os.path.getsize(os.path.join(HERE,'sprite.jpg'))/1e6:.2f} MB")

    # residual maps for the explainer (quantised to uint8 over a fixed 0..0.25 range)
    sig_same, sig_mc, ego = [], [], []
    prev = to_gray(frames[0], IMG)
    for i in range(1, n):
        g = to_gray(frames[i], IMG)
        m = estimate_motion(prev, g, GRID)
        w = remap_gray(prev, m.patch_disp, GRID)
        q = lambda a: base64.b64encode(np.clip(a / 0.25 * 255, 0, 255).astype(np.uint8)).decode()
        sig_same.append(q(patch_residual(g, prev, GRID)))
        sig_mc.append(q(patch_residual(g, w, GRID)))
        ego.append(round(float(m.mag_median), 4))
        prev = g

    data = {"meta": {"n_frames": n, "grid": list(GRID), "thumb": THUMB, "cols": cols,
                     "labels": list(labels), "taus": TAUS, "budgets_mw": BUDGETS_MW,
                     "fps": FPS, "warmup": WARMUP, "residual_scale": 0.25},
            "ego_motion_median": ego,
            "signals": {"same": sig_same, "mc": sig_mc},
            "full": run(bb, frames, "full", 0.06),
            "temporalsim": {}, "saccade": {}}

    for tau in TAUS:
        data["temporalsim"][f"{tau}"] = run(bb, frames, "temporalsim", tau)
        print(f"  [B] tau={tau} enc={data['temporalsim'][f'{tau}']['summary']['enc']*100:.1f}%")
    for tau in TAUS:
        for b in BUDGETS_MW:
            for mc in MCS:
                k = f"{tau}|{b}|{int(mc)}"
                data["saccade"][k] = run(bb, frames, "saccade", tau,
                                         budget=(b / 1000.0 if b else None), mc=mc)
        print(f"  [C] tau={tau} done")

    p = os.path.join(HERE, "data.json")
    json.dump(data, open(p, "w"), separators=(",", ":"))
    print(f"[data] {p} -> {os.path.getsize(p)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
