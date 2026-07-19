"""Frame sources for the always-on setting.

Claim demonstrated: provides the **predictable ego-motion** stream that is the
centre of the whole argument — a moving camera over a *static* world. Under such
motion the temporal-similarity baseline cannot skip (pixels shift everywhere) but
Saccade can (motion-compensated residual ~0). We generate it synthetically so the
benchmark is deterministic and hermetic; ``webcam`` / ``mp4`` sources are opt-in for
a live demo.

The synthetic stream is explicitly labelled as synthetic ego-motion; it is a
controlled stand-in for a wearable/vehicle camera, chosen because it lets us script
the exact regimes that stress the three conditions:

* ``walk``  — smooth translation across a static textured world (predictable).
* ``turn``  — faster translation + mild rotation (still ego-motion, harder to predict).
* ``event`` — camera nearly still while a NEW object moves through (genuine change).
* ``static``— camera still, world still (both baselines can skip; fair region).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _make_canvas(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A large, richly-textured static world to pan across (deterministic)."""
    import cv2
    rng = np.random.default_rng(seed)
    # low-frequency colour field
    low = rng.random((h // 32 + 2, w // 32 + 2, 3)).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    # mid-frequency texture
    mid = rng.random((h // 8 + 2, w // 8 + 2, 3)).astype(np.float32)
    mid = cv2.resize(mid, (w, h), interpolation=cv2.INTER_LINEAR)
    canvas = (0.6 * low + 0.4 * mid)
    canvas = (canvas - canvas.min()) / (np.ptp(canvas) + 1e-6)
    img = (canvas * 255).astype(np.uint8)
    # scatter structural shapes so patches have edges/corners (good for flow)
    for _ in range(140):
        c = tuple(int(x) for x in rng.integers(40, 230, size=3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        kind = rng.integers(0, 3)
        if kind == 0:
            r = int(rng.integers(8, 40))
            cv2.circle(img, (x, y), r, c, -1, lineType=cv2.LINE_AA)
        elif kind == 1:
            s = int(rng.integers(12, 60))
            cv2.rectangle(img, (x, y), (x + s, y + s), c, -1)
        else:
            x2, y2 = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.line(img, (x, y), (x2, y2), c, int(rng.integers(2, 6)), lineType=cv2.LINE_AA)
    return img


def _crop(canvas: np.ndarray, cx: float, cy: float, size: int, angle: float = 0.0) -> np.ndarray:
    """Crop a ``size``x``size`` window centred at (cx, cy), optionally rotated."""
    import cv2
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
    M[0, 2] += size / 2 - cx
    M[1, 2] += size / 2 - cy
    out = cv2.warpAffine(canvas, M, (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    return out


def synthetic_walking_stream(img_size: int = 224, seed: int = 0,
                             segments: Optional[List[Tuple[str, int]]] = None
                             ) -> Tuple[List[np.ndarray], List[str]]:
    """Generate the scripted ego-motion stream. Returns (frames, per-frame segment labels).

    Claim: 省エネ centrepiece. The ``walk``/``turn`` segments are predictable ego-motion
    where Saccade should skip heavily; ``event`` is genuine change where it must spend;
    ``static`` is a fair region where both baselines skip.
    """
    import cv2
    if segments is None:
        segments = [("static", 12), ("walk", 40), ("turn", 24), ("event", 28),
                    ("walk", 32), ("static", 8)]
    ch, cw = img_size * 3, img_size * 4
    canvas = _make_canvas(ch, cw, seed=seed)

    frames: List[np.ndarray] = []
    labels: List[str] = []
    cx, cy = cw * 0.25, ch * 0.5
    angle = 0.0
    rng = np.random.default_rng(seed + 1)
    # a moving object for the "event" segment (independent of the camera): large and
    # high-contrast so BOTH baselines register it -> the "event" region is a fair
    # regime where every method must spend, isolating the walk/turn win.
    obj_pos = np.array([img_size * 0.12, img_size * 0.55])
    obj_vel = np.array([4.5, 1.3])
    obj_color = (20, 230, 255)
    obj_radius = 24

    for name, count in segments:
        for k in range(count):
            if name == "static":
                vx, vy, dang = 0.0, 0.0, 0.0
            elif name == "walk":
                vx, vy, dang = 5.5, 1.8, 0.0
            elif name == "turn":
                vx, vy, dang = 7.0, -1.0, 0.35
            elif name == "event":
                # static camera, independently moving object -> genuine change that
                # every method must catch via same-position residual (fair regime).
                vx, vy, dang = 0.0, 0.0, 0.0
            else:
                vx, vy, dang = 0.0, 0.0, 0.0
            # keep the crop inside the canvas (bounce)
            if not (img_size <= cx + vx <= cw - img_size):
                vx = -vx
            if not (img_size <= cy + vy <= ch - img_size):
                vy = -vy
            cx += vx; cy += vy; angle += dang
            frame = _crop(canvas, cx, cy, img_size, angle)
            if name == "event":
                obj_pos = obj_pos + obj_vel
                if not (obj_radius <= obj_pos[0] <= img_size - obj_radius):
                    obj_vel[0] = -obj_vel[0]
                if not (obj_radius <= obj_pos[1] <= img_size - obj_radius):
                    obj_vel[1] = -obj_vel[1]
                cv2.circle(frame, (int(obj_pos[0]), int(obj_pos[1])), obj_radius, obj_color, -1,
                           lineType=cv2.LINE_AA)
                cv2.circle(frame, (int(obj_pos[0]), int(obj_pos[1])), obj_radius, (10, 10, 10), 2,
                           lineType=cv2.LINE_AA)
            # mild sensor noise for realism
            noise = rng.normal(0, 2.0, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            frames.append(frame)
            labels.append(name)
    return frames, labels


def frames_from_video(source, n_frames: int = 150, img_size: int = 224,
                      stride: int = 1) -> Tuple[List[np.ndarray], List[str]]:
    """Read frames from a webcam index (e.g. ``0``) or an mp4 path. Returns (frames, labels).

    Claim: live-demo path for the same pipeline. Labels are all ``"live"`` (no ground-truth
    segmentation for real video). Raises if the source cannot be opened.
    """
    import cv2
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video source: {source!r}")
    frames: List[np.ndarray] = []
    try:
        grabbed = 0
        while len(frames) < n_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            grabbed += 1
            if grabbed % stride:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from source: {source!r}")
    return frames, ["live"] * len(frames)
