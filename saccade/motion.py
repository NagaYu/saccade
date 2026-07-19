"""Pseudo-IMU: approximate ego-motion from dense optical flow.

Claim demonstrated: **省エネ** (motion is what lets us reuse cached tokens instead
of re-encoding) — this module is the "IMU we don't have". A wearable/edge camera
would read real inertial data; lacking it, we estimate frame-to-frame self-motion
from OpenCV optical flow and expose it in two forms:

* per-patch displacement (in *patch units*) -> consumed by :mod:`saccade.kv_remap`
  to slide cached tokens to where the world moved, and by the predictor.
* a compact global ego-motion feature vector (mean flow, divergence, rotation,
  magnitude stats) -> extra input to the predictor.

Everything here is pixel-space and cheap; its cost is folded into the
``fixed_overhead_flops`` of the FLOP model so Saccade's energy floor stays honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MotionField:
    """Result of one optical-flow estimate between consecutive frames.

    ``patch_disp`` is (Gh, Gw, 2) mean flow per patch in **patch units** (dx, dy),
    i.e. how far the content in each grid cell moved, measured in grid cells.
    ``global_feat`` is a small fixed-length ego-motion summary for the predictor.
    """

    patch_disp: np.ndarray     # (Gh, Gw, 2) float32, patch units
    global_feat: np.ndarray    # (6,) float32
    mag_mean: float            # mean flow magnitude in patch units
    mag_median: float          # median flow magnitude in patch units (robust ego-motion indicator)

    @property
    def is_static(self) -> bool:
        # Median, not mean: a single independently-moving object must not read as
        # camera motion. Ego-motion is what *most* of the scene does.
        return self.mag_median < 1e-2


def to_gray(frame: np.ndarray, img_size: int) -> np.ndarray:
    """Return a uint8 grayscale image resized to (img_size, img_size)."""
    import cv2
    if frame.ndim == 3 and frame.shape[2] >= 3:
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = frame if frame.ndim == 2 else frame[:, :, 0]
    if gray.dtype != np.uint8:
        gray = np.clip(gray * (255.0 if gray.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    if gray.shape[0] != img_size or gray.shape[1] != img_size:
        gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return gray


def estimate_motion(prev_gray: np.ndarray, gray: np.ndarray, grid: Tuple[int, int]) -> MotionField:
    """Dense Farneback flow -> per-patch displacement + global ego-motion features.

    Claim: 省エネ. The per-patch displacement is exactly the signal that separates
    *predictable* motion (a smooth global shift a moving camera induces) from
    *genuine change*. Motion compensation uses it to avoid re-encoding patches that
    only moved. The global features let the predictor anticipate the next frame.
    """
    import cv2
    gh, gw = grid
    h, w = gray.shape
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )  # (H, W, 2) in pixels

    ph, pw = h / gh, w / gw
    patch_disp = np.zeros((gh, gw, 2), dtype=np.float32)
    for i in range(gh):
        for j in range(gw):
            cell = flow[int(i * ph):int((i + 1) * ph), int(j * pw):int((j + 1) * pw)]
            fx = float(cell[..., 0].mean())
            fy = float(cell[..., 1].mean())
            patch_disp[i, j, 0] = fx / pw     # -> patch units (columns)
            patch_disp[i, j, 1] = fy / ph     # -> patch units (rows)

    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    # Ego-motion summary: translation, curl (rotation), divergence (zoom), spread.
    dfx_dy, dfx_dx = np.gradient(fx)
    dfy_dy, dfy_dx = np.gradient(fy)
    curl = float((dfy_dx - dfx_dy).mean())
    div = float((dfx_dx + dfy_dy).mean())
    global_feat = np.array([
        float(fx.mean()) / pw,
        float(fy.mean()) / ph,
        curl,
        div,
        float(mag.mean()) / ((ph + pw) / 2),
        float(mag.std()) / ((ph + pw) / 2),
    ], dtype=np.float32)
    patch_mag = np.sqrt(patch_disp[..., 0] ** 2 + patch_disp[..., 1] ** 2)
    mag_mean = float(patch_mag.mean())
    mag_median = float(np.median(patch_mag))
    return MotionField(patch_disp=patch_disp, global_feat=global_feat,
                       mag_mean=mag_mean, mag_median=mag_median)


def zero_motion(grid: Tuple[int, int]) -> MotionField:
    """A no-motion field for the very first frame / static bootstrap."""
    gh, gw = grid
    return MotionField(
        patch_disp=np.zeros((gh, gw, 2), dtype=np.float32),
        global_feat=np.zeros(6, dtype=np.float32),
        mag_mean=0.0,
        mag_median=0.0,
    )
