"""Motion-compensated cache remap.

Claim demonstrated: **省エネ** (fewer recomputed patches) and, indirectly, **品質保持**
(reused tokens land in the right place, so reuse stays accurate).

Core idea: a moving camera makes almost every patch's *pixels* change even though
the *world* is unchanged — the content simply slid to a neighbouring grid cell.
A same-position cache reuse would therefore look "surprising" everywhere and force
a full re-encode (this is precisely why the temporal-similarity baseline cannot
skip steady ego-motion). If instead we first slide the cached tokens (and the
cached grayscale used by the gate) *backwards along the estimated motion*, the
content lines up again and the residual collapses to ~0 for the static world.

This module implements that warp on the patch grid: for each destination patch we
gather the cached value from the source cell it came from (``src = dst - disp``),
with bilinear interpolation and an out-of-frame validity mask (newly revealed
patches at the frame edge are marked invalid and must be encoded).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def remap_grid(values: torch.Tensor, patch_disp: np.ndarray,
               grid: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp a per-patch tensor along ego-motion. Returns (warped, valid_mask).

    Claim: 省エネ. ``values`` is (P, D) on a Gh x Gw grid (row-major). ``patch_disp``
    is (Gh, Gw, 2) displacement in patch units: content in destination cell (i, j)
    was located at ``(i - dy, j - dx)`` in the previous frame, so we sample the
    cache there. Bilinear gather; destinations whose source falls outside the grid
    get ``valid=False`` (revealed region -> caller must re-encode those).

    Returns
    -------
    warped : (P, D) tensor — cache slid into the current frame's coordinates.
    valid  : (P,) bool tensor — True where the warp had in-frame support.
    """
    gh, gw = grid
    d = values.shape[1]
    vals = values.reshape(gh, gw, d)

    ii, jj = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    src_i = ii.astype(np.float32) - patch_disp[..., 1]   # dy moves rows
    src_j = jj.astype(np.float32) - patch_disp[..., 0]   # dx moves cols

    valid = (src_i >= 0) & (src_i <= gh - 1) & (src_j >= 0) & (src_j <= gw - 1)

    # clamp for safe gather; invalidity is tracked separately
    ci = np.clip(src_i, 0, gh - 1)
    cj = np.clip(src_j, 0, gw - 1)
    i0 = np.floor(ci).astype(np.int64); i1 = np.minimum(i0 + 1, gh - 1)
    j0 = np.floor(cj).astype(np.int64); j1 = np.minimum(j0 + 1, gw - 1)
    wi = torch.from_numpy((ci - i0).astype(np.float32)).unsqueeze(-1)
    wj = torch.from_numpy((cj - j0).astype(np.float32)).unsqueeze(-1)
    i0t = torch.from_numpy(i0); i1t = torch.from_numpy(i1)
    j0t = torch.from_numpy(j0); j1t = torch.from_numpy(j1)

    def gather(it, jt):
        return vals[it.reshape(-1), jt.reshape(-1)].reshape(gh, gw, d)

    top = gather(i0t, j0t) * (1 - wj) + gather(i0t, j1t) * wj
    bot = gather(i1t, j0t) * (1 - wj) + gather(i1t, j1t) * wj
    warped = (top * (1 - wi) + bot * wi).reshape(gh * gw, d)
    valid_t = torch.from_numpy(valid.reshape(-1))
    return warped, valid_t


def remap_gray(gray: np.ndarray, patch_disp: np.ndarray, grid: Tuple[int, int]) -> np.ndarray:
    """Warp a grayscale frame by the *patch-grid* displacement (upsampled to pixels).

    Claim: 省エネ. The gate compares the current frame to the motion-compensated
    previous frame in pixel space; this produces that compensated reference so the
    residual reflects genuine appearance change rather than ego-motion shift.
    """
    import cv2
    gh, gw = grid
    h, w = gray.shape
    # upsample per-patch displacement to a dense pixel flow, then warp
    disp_x = cv2.resize(patch_disp[..., 0], (w, h), interpolation=cv2.INTER_LINEAR) * (w / gw)
    disp_y = cv2.resize(patch_disp[..., 1], (w, h), interpolation=cv2.INTER_LINEAR) * (h / gh)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (xx - disp_x).astype(np.float32)
    map_y = (yy - disp_y).astype(np.float32)
    return cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
