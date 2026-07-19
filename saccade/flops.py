"""Analytic FLOP + energy model for a patch-token vision encoder.

Claim demonstrated: **省エネ (energy saving)** and **予算保証 (budget guarantee)**.

The whole Saccade thesis is "spend compute only on patches that surprise you".
To make that measurable we need a defensible, monotone cost model that maps
`number of patches actually (re)encoded` -> FLOPs -> Joules. Every core module
reports its decisions in units of *patches encoded*; this module turns those
decisions into the energy numbers the benchmark and the budget controller act on.

The model follows the standard ViT accounting, but attributed *per token* so
that encoding a subset ``S`` of the ``P`` patches costs ``|S| * per_patch``:

    patch-embedding projection      : 2 * (patch_pixels) * D                  (per encoded patch)
    per-layer token linear (QKV+O+MLP): L * 2 * (3 D^2 + D^2 + 2 mlp D^2)      (per encoded patch)
    attention (query over P keys)   : L * 2 * (2 * P * D)                      (per encoded query)

This is the *token-caching approximation*: a recomputed patch attends over the
(mostly cached) key set, while unchanged patches keep their cached token. It is
the same approximation used by patch/token-reuse edge-VLM systems and is exactly
what lets per-patch gating translate into per-patch savings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlopModel:
    """Per-frame FLOP/energy accounting for a ViT-style patch encoder.

    Claim: 省エネ / 予算保証. Provides the single source of truth that converts a
    gating decision (how many patches were encoded) into estimated Joules, so the
    energy controller can enforce a hard budget and the benchmark can compare A/B/C
    on identical physical units.
    """

    n_patches: int          # P: total patches per frame
    embed_dim: int          # D
    depth: int              # L transformer layers
    mlp_ratio: float        # MLP hidden expansion
    patch_pixels: int       # patch_h * patch_w * channels (patch-embed input size)
    j_per_flop: float = 1e-12          # ~1 pJ/FLOP, efficient mobile inference
    fixed_overhead_flops: float = 0.0  # per *processed* frame: optical flow + gate + predictor
    # LLM prefill cost per re-encoded vision token. Under token-caching KV reuse, a
    # patch whose token is reused keeps its LLM KV entries too — only *fresh* tokens
    # must be prefilled again, so prefill scales with the same n_encoded the gate
    # controls. Default 0 models the encoder-dominated regime; set to
    # ~2 * 12 * D_llm^2 * L_llm to include a concrete small-LM prefill.
    llm_prefill_flops_per_token: float = 0.0

    # ---- per-patch cost decomposition (FLOPs) -------------------------------
    @property
    def embed_flops(self) -> float:
        """Patch-embedding projection cost for ONE patch (multiply-add = 2 FLOPs)."""
        return 2.0 * self.patch_pixels * self.embed_dim

    @property
    def linear_flops(self) -> float:
        """Per-token linear cost (QKV, attn-out, MLP up/down) across all L layers, ONE token."""
        d = self.embed_dim
        per_layer = 2.0 * (3.0 * d * d      # QKV projection
                           + d * d          # attention output projection
                           + 2.0 * self.mlp_ratio * d * d)  # MLP up + down
        return self.depth * per_layer

    @property
    def attn_flops(self) -> float:
        """Attention cost for ONE query attending over all P keys, across L layers.

        QK^T similarity (2*D per key) + weighted-sum over V (2*D per key) = 4*D per key.
        """
        return self.depth * 2.0 * (2.0 * self.n_patches * self.embed_dim)

    @property
    def per_patch_flops(self) -> float:
        """Total marginal FLOPs to (re)encode one patch this frame.

        Claim: 省エネ. Includes the optional LLM-prefill share: with token-caching KV
        reuse, skipping a patch saves BOTH its encoder pass and its LLM prefill, so
        the whole marginal cost rides on the gate's decision.
        """
        return (self.embed_flops + self.linear_flops + self.attn_flops
                + self.llm_prefill_flops_per_token)

    # ---- frame-level roll-ups ----------------------------------------------
    def frame_flops(self, n_encoded: int, processed: bool = True) -> float:
        """FLOPs for a frame that (re)encoded ``n_encoded`` patches.

        Claim: 省エネ. Monotone non-decreasing in ``n_encoded`` — fewer surprising
        patches strictly costs fewer FLOPs, which is the quantity Saccade minimises.

        ``processed=False`` marks a *fully skipped* frame (frame-drop): even the
        cheap flow/gate/predictor overhead is not paid because we did not look at
        the frame at all (pure cache reuse). ``processed=True`` (default) pays the
        fixed overhead once regardless of how many patches were encoded, so Saccade
        is never modelled as literally free.
        """
        if n_encoded < 0:
            raise ValueError("n_encoded must be >= 0")
        base = self.fixed_overhead_flops if processed else 0.0
        return base + n_encoded * self.per_patch_flops

    def full_frame_flops(self) -> float:
        """Cost of encoding every patch (the Full baseline's per-frame cost)."""
        return self.frame_flops(self.n_patches, processed=True)

    def joules(self, flops: float) -> float:
        """Convert FLOPs to estimated Joules. Claim: 予算保証 (energy is the budgeted unit)."""
        return flops * self.j_per_flop

    def frame_joules(self, n_encoded: int, processed: bool = True) -> float:
        """Estimated Joules for a frame that encoded ``n_encoded`` patches."""
        return self.joules(self.frame_flops(n_encoded, processed=processed))


def vit_s_like(n_patches: int = 196, patch_size: int = 16, channels: int = 3,
               j_per_flop: float = 1e-12) -> FlopModel:
    """A ViT-S/16-like cost model (D=384, L=12, mlp=4) — matches the SyntheticBackbone.

    Fixed overhead is set to a small, realistic value covering dense optical flow,
    the pixel-space gate, and the tiny GRU predictor, so a 0-encode Saccade frame
    still shows a non-zero energy floor (honest accounting).
    """
    d, depth = 384, 12
    # Overhead: Farneback flow (~a few ops/pixel over the frame) + per-patch gate +
    # predictor GRU (~ P * (few) * D * D_h). Order ~1e7 FLOPs; small vs a full frame (~1e9).
    overhead = 1.2e7
    return FlopModel(
        n_patches=n_patches,
        embed_dim=d,
        depth=depth,
        mlp_ratio=4.0,
        patch_pixels=patch_size * patch_size * channels,
        j_per_flop=j_per_flop,
        fixed_overhead_flops=overhead,
    )
