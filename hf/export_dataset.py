"""Build the Saccade ego-motion benchmark as a Hugging Face dataset.

Claim demonstrated: makes 省エネ / 品質保持 / 予算保証 independently *auditable*. Rather
than asking anyone to trust our plots, the dataset ships the raw evidence:

* ``frames``       — the scripted ego-motion stream itself (image + regime label +
                     measured ego-motion), so another method can be benchmarked on the
                     exact same input.
* ``signals``      — per-patch residual maps for BOTH decision rules on every frame:
                     the same-position residual TemporalSim thresholds and the
                     motion-compensated residual Saccade thresholds. This is the direct
                     evidence for "why temporal-similarity cannot skip steady motion" —
                     verifiable with numpy alone, no need to run our code.
* ``measurements`` — per-frame, per-condition energy/quality records for Full,
                     TemporalSim and Saccade.

Usage
-----
    python hf/export_dataset.py            # build parquet under hf/dataset/
    python hf/export_dataset.py --push     # also push to the Hub
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saccade import (EngineConfig, SaccadeEngine, SyntheticBackbone, estimate_motion,
                     summarize, synthetic_walking_stream, to_gray)
from saccade.gate import patch_residual
from saccade.kv_remap import remap_gray

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "dataset")
REPO_ID = "NagaYu/saccade-egomotion-bench"
GRID = (14, 14)
IMG = 224
FPS = 10.0
WARMUP = 6
TAU = 0.06


def build_signals(frames, labels) -> List[Dict]:
    """Per-frame per-patch residual maps for both decision rules. Claim: 省エネ (evidence).

    ``residual_same_position`` is what a temporal-similarity gate sees;
    ``residual_motion_compensated`` is what Saccade sees after removing ego-motion.
    On the walking segments the first is high almost everywhere while the second
    collapses to ~0 — the whole thesis, in two arrays per frame.
    """
    rows = []
    prev = to_gray(frames[0], IMG)
    for i in range(1, len(frames)):
        gray = to_gray(frames[i], IMG)
        m = estimate_motion(prev, gray, GRID)
        warped = remap_gray(prev, m.patch_disp, GRID)
        r_same = patch_residual(gray, prev, GRID)
        r_mc = patch_residual(gray, warped, GRID)
        rows.append({
            "index": i,
            "segment": labels[i],
            "ego_motion_mag_mean": float(m.mag_mean),
            "ego_motion_mag_median": float(m.mag_median),
            "residual_same_position": r_same.astype(np.float32).tolist(),
            "residual_motion_compensated": r_mc.astype(np.float32).tolist(),
            "encode_frac_same_position": float((r_same > TAU).mean()),
            "encode_frac_motion_compensated": float((r_mc > TAU).mean()),
        })
        prev = gray
    return rows


def build_measurements(backbone, frames) -> List[Dict]:
    """Per-frame, per-condition energy and quality records. Claim: all three."""
    rows = []
    for label, mode, budget in [("Full", "full", None), ("TemporalSim", "temporalsim", None),
                                ("Saccade", "saccade", 0.02)]:
        cfg = EngineConfig(mode=mode, tau=TAU, temporalsim_tau=TAU, budget_watts=budget,
                           fps=FPS, img_size=IMG)
        res = SaccadeEngine(backbone, cfg, measure_fidelity=True).run(frames)
        for r in res:
            rows.append({
                "index": r.index, "condition": label,
                "n_encoded": int(r.n_encoded), "encoded_fraction": float(r.encoded_fraction),
                "fidelity": float(r.fidelity), "frame_joules": float(r.frame_joules),
                "cum_joules": float(r.cum_joules), "frame_flops": float(r.frame_flops),
                "processed": bool(r.processed), "tau": float(r.tau),
                "budget_ok": bool(r.budget_ok),
            })
        s = summarize(res, label, FPS, warmup=WARMUP)
        print(f"  [{label:12s}] enc={s.mean_encoded_fraction*100:5.1f}% "
              f"fid={s.mean_fidelity:.4f} J={s.total_joules:.4f} hrs@10kJ={s.runtime_hours_10kJ:.1f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo", default=REPO_ID)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    from datasets import Dataset, Features, Image as HFImage, Value, Sequence
    from PIL import Image as PILImage

    print("[stream] generating scripted ego-motion stream")
    frames, labels = synthetic_walking_stream(img_size=IMG, seed=0)
    backbone = SyntheticBackbone(grid=GRID, img_size=IMG, embed_dim=384, seed=0)
    print(f"[stream] {len(frames)} frames, segments={sorted(set(labels))}")

    # ---- frames config ----
    def png_bytes(a: np.ndarray) -> bytes:
        buf = io.BytesIO(); PILImage.fromarray(a).save(buf, format="PNG"); return buf.getvalue()

    print("[build] frames")
    prev = to_gray(frames[0], IMG)
    ego = [(0.0, 0.0)]
    for i in range(1, len(frames)):
        g = to_gray(frames[i], IMG)
        m = estimate_motion(prev, g, GRID)
        ego.append((float(m.mag_mean), float(m.mag_median)))
        prev = g
    ds_frames = Dataset.from_dict(
        {
            "index": list(range(len(frames))),
            "image": [{"bytes": png_bytes(f), "path": f"frame_{i:04d}.png"}
                      for i, f in enumerate(frames)],
            "segment": list(labels),
            "ego_motion_mag_mean": [e[0] for e in ego],
            "ego_motion_mag_median": [e[1] for e in ego],
        },
        features=Features({
            "index": Value("int32"), "image": HFImage(), "segment": Value("string"),
            "ego_motion_mag_mean": Value("float32"), "ego_motion_mag_median": Value("float32"),
        }),
    )

    print("[build] signals (per-patch residual maps)")
    sig_rows = build_signals(frames, labels)
    ds_signals = Dataset.from_list(sig_rows, features=Features({
        "index": Value("int32"), "segment": Value("string"),
        "ego_motion_mag_mean": Value("float32"), "ego_motion_mag_median": Value("float32"),
        "residual_same_position": Sequence(Value("float32")),
        "residual_motion_compensated": Sequence(Value("float32")),
        "encode_frac_same_position": Value("float32"),
        "encode_frac_motion_compensated": Value("float32"),
    }))

    print("[build] measurements (Full / TemporalSim / Saccade)")
    meas_rows = build_measurements(backbone, frames)
    ds_meas = Dataset.from_list(meas_rows)

    for name, ds in [("frames", ds_frames), ("signals", ds_signals), ("measurements", ds_meas)]:
        p = os.path.join(OUTDIR, f"{name}.parquet")
        ds.to_parquet(p)
        print(f"[save] {p}  rows={len(ds)}  {os.path.getsize(p)/1e6:.2f} MB")

    # headline evidence, so the card can quote measured numbers
    sig = ds_signals.to_pandas()
    walk = sig[sig.segment.isin(["walk", "turn"])]
    summary = {
        "n_frames": len(frames),
        "segments": {s: int((np.array(labels) == s).sum()) for s in sorted(set(labels))},
        "walk_turn_encode_frac_same_position": float(walk.encode_frac_same_position.mean()),
        "walk_turn_encode_frac_motion_compensated": float(walk.encode_frac_motion_compensated.mean()),
        "grid": list(GRID), "img_size": IMG, "tau": TAU, "fps": FPS,
    }
    json.dump(summary, open(os.path.join(OUTDIR, "summary.json"), "w"), indent=2)
    print(f"[evidence] walk+turn: same-position gate would encode "
          f"{summary['walk_turn_encode_frac_same_position']*100:.1f}% of patches, "
          f"motion-compensated gate {summary['walk_turn_encode_frac_motion_compensated']*100:.1f}%")

    if args.push:
        print(f"[push] {args.repo}")
        ds_frames.push_to_hub(args.repo, config_name="frames")
        ds_signals.push_to_hub(args.repo, config_name="signals")
        ds_meas.push_to_hub(args.repo, config_name="measurements")
        print("[push] done")


if __name__ == "__main__":
    main()
