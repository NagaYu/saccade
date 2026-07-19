"""Vision backbones: the thing Saccade decides *not* to run on every patch.

Claim demonstrated: this is the encoder whose FLOPs the gate (省エネ) suppresses
and whose outputs the fidelity metric (品質保持) compares against.

Two implementations behind one interface:

* :class:`SyntheticBackbone` — a deterministic, dependency-free, frozen
  random-feature patch encoder (ViT-S/16 geometry). It can genuinely encode an
  *arbitrary subset* of patches in isolation, so per-patch gating produces *real*
  compute savings. Used for the hermetic tests and the reproducible benchmark.

* :class:`HFVisionBackbone` — a thin wrapper over a real HuggingFace ViT/DINOv2
  vision tower that exposes the per-patch token embeddings ("patch embeddings").
  A real transformer mixes all patches via self-attention, so a subset cannot be
  encoded in true isolation; we run the tower once for correct fresh token values
  and *account* FLOPs for only the selected subset (the token-caching approximation,
  documented in :mod:`saccade.flops`). This is the wrapper the spec asks for; it is
  opt-in because the energy claim is cleanest on the synthetic encoder.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .flops import FlopModel, vit_s_like


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class VisionBackbone:
    """Interface: turn frame pixels into per-patch token embeddings, selectively.

    Concrete subclasses expose ``grid``, ``embed_dim``, ``flop_model`` and
    ``encode(frame, patch_indices)``. ``encode`` with a subset of indices is the
    single intervention point Saccade exploits.
    """

    grid: Tuple[int, int]
    embed_dim: int
    flop_model: FlopModel

    @property
    def n_patches(self) -> int:
        return self.grid[0] * self.grid[1]

    def patchify(self, frame: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, frame: np.ndarray, patch_indices: Optional[Sequence[int]] = None) -> torch.Tensor:
        raise NotImplementedError


class SyntheticBackbone(VisionBackbone):
    """Frozen random-feature patch encoder — deterministic, fast, subset-encodable.

    Claim: 省エネ. Because each patch embedding is an *independent* function of that
    patch's pixels, encoding only the surprising subset yields real, not merely
    modelled, savings — this is the backbone the energy tests rely on.

    The encoder is a fixed 2-layer MLP on a downsampled patch (frozen random
    weights, seeded) plus a fixed positional code. Frozen random features are a
    legitimate cheap stand-in for a trained ViT: similar patches map to similar
    embeddings (so fidelity is meaningful) and content changes move the embedding
    (so surprisal is meaningful).
    """

    def __init__(self, grid: Tuple[int, int] = (14, 14), img_size: int = 224,
                 embed_dim: int = 384, channels: int = 3, tile: int = 8,
                 seed: int = 0, j_per_flop: float = 1e-12):
        self.grid = grid
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.channels = channels
        self.tile = tile                      # patch is downsampled to tile x tile before projection
        self.patch_h = img_size // grid[0]
        self.patch_w = img_size // grid[1]
        g = torch.Generator().manual_seed(seed)
        in_dim = tile * tile * channels
        hidden = 256
        # Frozen random projection weights (fixed for the life of the process).
        self.W1 = torch.randn(in_dim, hidden, generator=g) / np.sqrt(in_dim)
        self.b1 = torch.zeros(hidden)
        self.W2 = torch.randn(hidden, embed_dim, generator=g) / np.sqrt(hidden)
        self.b2 = torch.zeros(embed_dim)
        # Fixed positional embedding so identical texture at different grid cells
        # is still distinguishable (as in a real ViT).
        self.pos = _l2norm(torch.randn(self.n_patches, embed_dim, generator=g)) * 0.3
        self.flop_model = vit_s_like(
            n_patches=self.n_patches,
            patch_size=self.patch_h,   # square patches on a square image
            channels=channels,
            j_per_flop=j_per_flop,
        )

    # -- pixel helpers --------------------------------------------------------
    def _as_chw(self, frame: np.ndarray) -> torch.Tensor:
        """Return a float32 CxHxW tensor in [0,1] at ``img_size``."""
        import cv2
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], self.channels, axis=2)
        if frame.shape[2] != self.channels:
            frame = frame[:, :, : self.channels]
        if frame.shape[0] != self.img_size or frame.shape[1] != self.img_size:
            frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(frame)).float()
        if t.max() > 1.5:
            t = t / 255.0
        return t.permute(2, 0, 1).contiguous()

    def patchify(self, frame: np.ndarray) -> torch.Tensor:
        """Split a frame into P downsampled patch tensors of shape (P, tile*tile*C).

        Claim: 省エネ (cheap pixel-space representation the gate uses without touching
        the expensive encoder).
        """
        import cv2
        chw = self._as_chw(frame).numpy()
        gh, gw = self.grid
        patches = np.empty((self.n_patches, self.tile * self.tile * self.channels), dtype=np.float32)
        for i in range(gh):
            for j in range(gw):
                p = chw[:, i * self.patch_h:(i + 1) * self.patch_h,
                          j * self.patch_w:(j + 1) * self.patch_w]           # C,ph,pw
                p = np.transpose(p, (1, 2, 0))                                # ph,pw,C
                p = cv2.resize(p, (self.tile, self.tile), interpolation=cv2.INTER_AREA)
                patches[i * gw + j] = p.reshape(-1)
        return torch.from_numpy(patches)

    def encode(self, frame: np.ndarray, patch_indices: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Encode all patches, or only ``patch_indices``, into L2-normalised embeddings.

        Claim: 省エネ. Encoding a subset returns embeddings for exactly those patches,
        so downstream FLOP accounting (|S| * per_patch) reflects real work avoided.
        Returns a tensor of shape (len(indices), D) aligned to ``patch_indices`` order,
        or (P, D) when ``patch_indices`` is None.
        """
        patches = self.patchify(frame)
        if patch_indices is not None:
            idx = torch.as_tensor(list(patch_indices), dtype=torch.long)
            patches = patches[idx]
            pos = self.pos[idx]
        else:
            pos = self.pos
        h = torch.tanh(patches @ self.W1 + self.b1)
        e = h @ self.W2 + self.b2
        e = _l2norm(e + pos)
        return e


class HFVisionBackbone(VisionBackbone):
    """Thin wrapper over a real HuggingFace ViT/DINOv2 vision tower (opt-in).

    Claim: 品質保持 / 省エネ on real features. Exposes the per-patch token embeddings
    ("patch embeddings") of a small pretrained encoder so the exact same Saccade
    machinery runs on genuine semantics. Subset encoding is *accounted* (not truly
    isolated) because self-attention mixes patches — see the token-caching note in
    :mod:`saccade.flops`.
    """

    def __init__(self, model_name: str = "facebook/dinov2-small", img_size: int = 224,
                 device: str = "cpu", j_per_flop: float = 1e-12):
        from transformers import AutoModel, AutoImageProcessor
        self.model_name = model_name
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        cfg = self.model.config
        self.embed_dim = int(cfg.hidden_size)
        patch = int(getattr(cfg, "patch_size", 16))
        # DINOv2/ViT use a square patch grid on a square image.
        proc_size = getattr(self.processor, "crop_size", None) or getattr(self.processor, "size", None)
        if isinstance(proc_size, dict):
            side = proc_size.get("height") or proc_size.get("shortest_edge") or img_size
        else:
            side = img_size
        self.img_size = int(side)
        g = self.img_size // patch
        self.grid = (g, g)
        self._register_tokens = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.flop_model = vit_s_like(
            n_patches=self.n_patches, patch_size=patch, channels=3, j_per_flop=j_per_flop,
        )
        # Rebuild the (frozen) FlopModel from the real config where available.
        self.flop_model = FlopModel(
            n_patches=self.n_patches,
            embed_dim=self.embed_dim,
            depth=int(getattr(cfg, "num_hidden_layers", 12)),
            mlp_ratio=float(getattr(cfg, "mlp_ratio", 4.0)) if getattr(cfg, "mlp_ratio", None) else 4.0,
            patch_pixels=patch * patch * 3,
            j_per_flop=j_per_flop,
            fixed_overhead_flops=1.2e7,
        )

    @torch.no_grad()
    def _all_tokens(self, frame: np.ndarray) -> torch.Tensor:
        from PIL import Image
        if frame.dtype != np.uint8:
            arr = np.clip(frame * (255.0 if frame.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
        else:
            arr = frame
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        img = Image.fromarray(arr[:, :, :3])
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        tok = out.last_hidden_state[0]                 # (1 + reg + P, D)
        tok = tok[1 + self._register_tokens:]          # drop CLS (+ register) tokens
        return _l2norm(tok)

    def patchify(self, frame: np.ndarray) -> torch.Tensor:
        # Reuse the synthetic pixel patchifier purely for the cheap gate signal.
        helper = SyntheticBackbone(grid=self.grid, img_size=self.img_size)
        return helper.patchify(frame)

    def encode(self, frame: np.ndarray, patch_indices: Optional[Sequence[int]] = None) -> torch.Tensor:
        tokens = self._all_tokens(frame)
        if tokens.shape[0] != self.n_patches:
            # some processors pad/resize differently; trim/pad to the declared grid
            if tokens.shape[0] > self.n_patches:
                tokens = tokens[: self.n_patches]
            else:
                pad = self.n_patches - tokens.shape[0]
                tokens = torch.cat([tokens, tokens[-1:].expand(pad, -1)], 0)
        if patch_indices is None:
            return tokens
        idx = torch.as_tensor(list(patch_indices), dtype=torch.long)
        return tokens[idx]


def make_backbone(spec: str = "synthetic", **kwargs) -> VisionBackbone:
    """Factory. ``spec`` is ``"synthetic"`` or ``"hf[:model_name]"``.

    Claim: keeps the four cores backbone-agnostic — they only see embeddings and a
    :class:`FlopModel`, so the same code proves the claims on synthetic or real ViT.
    """
    if spec == "synthetic":
        return SyntheticBackbone(**kwargs)
    if spec.startswith("hf"):
        _, _, name = spec.partition(":")
        name = name or "facebook/dinov2-small"
        return HFVisionBackbone(model_name=name, **kwargs)
    raise ValueError(f"unknown backbone spec: {spec!r}")
