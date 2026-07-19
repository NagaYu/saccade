"""Saccade — 予測誤差ゲート + エネルギー予算制御による常時オンのエッジVLM.

A research prototype demonstrating three claims on real-ish video streams:

* 省エネ   (energy saving)        — encode only the patches that surprise a forward predictor.
* 品質保持 (quality preservation) — the reconstructed patch-token map stays close to a Full encoder.
* 予算保証 (budget guarantee)     — a hard energy ceiling with an anytime degraded output.

Public API mirrors the four cores plus the engine that integrates them.
"""

from .backbone import HFVisionBackbone, SyntheticBackbone, VisionBackbone, make_backbone
from .energy_controller import EnergyBudgetController
from .engine import SaccadeEngine, summarize
from .flops import FlopModel, vit_s_like
from .gate import ImportanceTracker, SurprisalGate, embedding_surprisal, patch_residual
from .kv_remap import remap_grid, remap_gray
from .motion import MotionField, estimate_motion, to_gray, zero_motion
from .predictor import GRUPatchPredictor
from .stream import frames_from_video, synthetic_walking_stream
from .types import EngineConfig, FrameResult, RunSummary

__all__ = [
    "VisionBackbone", "SyntheticBackbone", "HFVisionBackbone", "make_backbone",
    "EnergyBudgetController", "SaccadeEngine", "summarize",
    "FlopModel", "vit_s_like", "SurprisalGate", "ImportanceTracker",
    "embedding_surprisal", "patch_residual", "remap_grid", "remap_gray",
    "MotionField", "estimate_motion", "to_gray", "zero_motion",
    "GRUPatchPredictor", "frames_from_video", "synthetic_walking_stream",
    "EngineConfig", "FrameResult", "RunSummary",
]

__version__ = "0.1.0"
