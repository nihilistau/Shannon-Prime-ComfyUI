# Shannon-Prime VHT2: ComfyUI Custom Nodes
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
# Licensed under AGPLv3. Commercial license available.
#
# Provides a ShannonPrimeWanCache node that wraps Wan 2.1/2.2 cross-attention
# K/V linear layers with VHT2-compressed caching. Text context (T5/UMT5 output)
# is identical across all diffusion timesteps, so the K/V projections of that
# context are recomputed ~50× needlessly in a vanilla Wan inference. This node
# intercepts those computations via drop-in nn.Module replacements on
# `block.cross_attn.k` and `block.cross_attn.v` (and `.k_img`/`.v_img` for
# I2V), caching the first computation and returning VHT2-reconstructed values
# on subsequent calls.
#
# Works with ComfyUI's native Wan implementation at comfy.ldm.wan.model.
# Safe to apply twice (idempotent). Cache is keyed per-block per-linear.
# Input-change detection uses the tensor's data_ptr: when the upstream context
# tensor is a new allocation (i.e. a new generation), cache entries for that
# linear are invalidated and refilled.

import os
import sys
import pathlib

import torch
import torch.nn as nn

# Make the shannon-prime submodule importable.
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SP_TOOLS = _REPO_ROOT / "lib" / "shannon-prime" / "tools"
_SP_TORCH = _REPO_ROOT / "lib" / "shannon-prime" / "backends" / "torch"
for p in (_SP_TOOLS, _SP_TORCH):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from shannon_prime_comfyui import VHT2CrossAttentionCache  # noqa: E402


def _input_fingerprint(x: torch.Tensor):
    """
    Cheap content-based fingerprint — identifies same-valued inputs even when
    re-allocated to a new memory address.

    We can't use `data_ptr()` as the identity for cache hits because ComfyUI's
    sampler re-batches the conditioning tensor into a fresh allocation each
    timestep. Pointer-identity invalidated the cache every step and the hits
    documented in the paper never materialised.

    Three flat-index samples + shape + dtype disambiguate "same context across
    timesteps" vs "fresh context for a new generation" without scanning the
    full tensor. On CUDA each `.item()` forces a single-element device→host
    sync; three of those per cross-attn forward is negligible (~10 µs total)
    compared to the compress/decompress work they save.
    """
    flat = x.view(-1) if x.is_contiguous() else x.reshape(-1)
    n = flat.numel()
    # Pick three anchors that spread across the tensor; identical data will
    # fingerprint identically regardless of reallocation.
    i_mid = n // 2 if n > 1 else 0
    i_end = n - 1 if n > 0 else 0
    return (
        tuple(x.shape),
        x.dtype,
        float(flat[0].item()),
        float(flat[i_mid].item()),
        float(flat[i_end].item()),
    )


class _SPCachingLinear(nn.Module):
    """
    Drop-in for cross_attn.k / cross_attn.v (and k_img / v_img).

    Caches the linear's output keyed by (key, content_fingerprint(input)).
    The first call for a given fingerprint computes + compresses; subsequent
    calls with the same fingerprint reconstruct from the VHT2 cache. When the
    fingerprint changes (new generation with a different prompt/seed), the
    stale entry is dropped and refilled on that call.
    """

    def __init__(self, original: nn.Module, cache: VHT2CrossAttentionCache, key: str):
        super().__init__()
        self.original = original
        self._sp_cache = cache
        self._sp_key = key
        self._sp_last_fp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cache = self._sp_cache
        key = self._sp_key
        fp = _input_fingerprint(x)

        # Cache hit: the conditioning content fingerprint matches what we
        # stored last time AND we have an entry. Return the reconstructed K/V.
        if fp == self._sp_last_fp and cache.has(key):
            out, _ = cache.get(key)
            return out

        # Fingerprint changed or no entry — recompute and refill.
        result = self.original(x)

        # Drop any stale entry under the expert-scoped key.
        stored_key = cache._cache_key(key)
        cache._cache.pop(stored_key, None)

        # VHT2CrossAttentionCache.put() stores a (k, v) pair; we use the same
        # tensor for both since each linear is cached independently.
        cache.put(key, result, result)
        self._sp_last_fp = fp
        return result


def _wrap_cross_attn(cross_attn: nn.Module, cache: VHT2CrossAttentionCache,
                     block_idx: int) -> bool:
    """Wrap .k, .v (and .k_img, .v_img if present) on a cross-attn module."""
    if cross_attn is None:
        return False
    wrapped_any = False
    for suffix, attr in (("k", "k"), ("v", "v"), ("kimg", "k_img"), ("vimg", "v_img")):
        lin = getattr(cross_attn, attr, None)
        if lin is None:
            continue
        if isinstance(lin, _SPCachingLinear):
            continue  # already wrapped
        wrapper = _SPCachingLinear(lin, cache, f"block_{block_idx}_{suffix}")
        setattr(cross_attn, attr, wrapper)
        wrapped_any = True
    return wrapped_any


def _iter_wan_blocks(model_obj):
    """Yield (index, block) for each WanAttentionBlock inside a ModelPatcher."""
    inner = getattr(model_obj, "model", model_obj)
    diff = getattr(inner, "diffusion_model", None)
    if diff is None:
        diff = getattr(inner, "model", None)
    if diff is None:
        return
    blocks = getattr(diff, "blocks", None)
    if blocks is None:
        return
    for i, blk in enumerate(blocks):
        yield i, blk


def _parse_bits_csv(s: str, default, width: int):
    try:
        out = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    except ValueError:
        return list(default)
    if not out:
        return list(default)
    # Pad or truncate to the requested width by repeating the last element.
    while len(out) < width:
        out.append(out[-1])
    return out[:width]


class ShannonPrimeWanCache:
    """
    Patches a Wan 2.1/2.2 MODEL to cache cross-attention K/V via Shannon-Prime
    VHT2 compression.

    Cross-attention K/V in Wan are linear projections of T5/UMT5 text
    embeddings. That context is constant across the ~50 diffusion timesteps
    in a generation, so the K/V tensors are invariant — computing them once
    and reusing them is strictly profitable for both compute and VRAM.

    The node monkey-patches block.cross_attn.{k, v} (and .k_img/.v_img on
    I2V) on every WanAttentionBlock it finds. First call through each
    wrapped linear computes + compresses; subsequent calls with the same
    context tensor (same data_ptr) reconstruct from the VHT2 cache.

    Apply once per Wan MODEL before the sampler. For Wan 2.2 MoE, apply
    separately to each expert MODEL — the two caches are naturally partitioned
    because the two experts are separate Python objects.
    """

    CATEGORY = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = ("Wraps Wan cross-attention K/V with VHT2 compressed caching. "
                   "Cross-attn context is constant across timesteps; compute once, "
                   "compress, reconstruct on subsequent calls.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "k_bits": ("STRING", {"default": "5,4,4,3",
                                      "tooltip": "K band bit allocation (4 bands)"}),
                "v_bits": ("STRING", {"default": "5,4,4,3",
                                      "tooltip": "V band bit allocation (4 bands for cross-attn, unlike self-attn flat 3-bit)"}),
                "use_mobius": ("BOOLEAN", {"default": True,
                                           "tooltip": "Möbius squarefree-first reorder on both K and V (cross-attn has no RoPE)"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # The patch isn't deterministic w.r.t. pure input hashing — different
        # applications share cache state on the model. Returning NaN forces
        # re-evaluation each queue.
        return float("nan")

    def patch(self, model, k_bits: str, v_bits: str, use_mobius: bool):
        k_bits_list = _parse_bits_csv(k_bits, [5, 4, 4, 3], 4)
        v_bits_list = _parse_bits_csv(v_bits, [5, 4, 4, 3], 4)

        # Clone the ModelPatcher so we don't mutate the caller's reference.
        # Note: ComfyUI's ModelPatcher.clone() shares the underlying nn.Module,
        # so our patches DO persist on the model object across workflows. For
        # multiple concurrent generations this is benign (same cache state is
        # safely reused); for switching between different Wan models load them
        # as separate UNETLoader instances.
        patched = model.clone()

        blocks = list(_iter_wan_blocks(patched))
        if not blocks:
            print("[Shannon-Prime] ShannonPrimeWanCache: no Wan blocks found on model — passing through")
            return (patched,)

        # Use the first block's head_dim (uniform across Wan architectures).
        head_dim = blocks[0][1].self_attn.head_dim

        cache = VHT2CrossAttentionCache(
            head_dim=head_dim,
            k_band_bits=k_bits_list,
            v_band_bits=v_bits_list,
            use_mobius=bool(use_mobius),
        )

        wrapped = 0
        for i, blk in blocks:
            if _wrap_cross_attn(getattr(blk, "cross_attn", None), cache, i):
                wrapped += 1

        # Stash cache on the ModelPatcher for stats / inspection via a debug node.
        patched._sp_cache = cache

        comp_ratio = cache.compression_ratio()
        print(f"[Shannon-Prime] patched {wrapped}/{len(blocks)} Wan blocks "
              f"(head_dim={head_dim}, K={k_bits_list}, V={v_bits_list}, "
              f"möbius={use_mobius}, compression~{comp_ratio:.2f}x)")

        return (patched,)


class ShannonPrimeWanCacheStats:
    """Reports cache hit/miss stats from a model previously patched by
    ShannonPrimeWanCache. Passes the model through unchanged."""

    CATEGORY = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "report"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")

    def report(self, model):
        cache = getattr(model, "_sp_cache", None)
        if cache is None:
            print("[Shannon-Prime] stats: model has no Shannon-Prime cache attached")
            return (model,)
        s = cache.stats()
        print(f"[Shannon-Prime] cache stats: hits={s['hits']} misses={s['misses']} "
              f"hit_rate={s['hit_rate']:.3f} entries={s['n_entries_cached']} "
              f"compression={s['compression_ratio']:.2f}x")
        return (model,)


NODE_CLASS_MAPPINGS = {
    "ShannonPrimeWanCache": ShannonPrimeWanCache,
    "ShannonPrimeWanCacheStats": ShannonPrimeWanCacheStats,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShannonPrimeWanCache": "Shannon-Prime Wan Cache",
    "ShannonPrimeWanCacheStats": "Shannon-Prime Wan Cache Stats",
}
