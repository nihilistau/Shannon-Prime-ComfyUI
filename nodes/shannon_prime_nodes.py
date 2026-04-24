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


class ShannonPrimeWanCacheSqfree:
    """
    Aggressive variant of ShannonPrimeWanCache using the sqfree+spinor path.

    Wraps Wan cross-attn K/V with the sqfree prime-Hartley basis + Möbius CSR
    predictor + (optional) SU(2) spinor sheet-bit correction. Target regime:
    Q8+ text encoders + bf16 diffusion weights (per CLAUDE.md scaling law).
    Uses `VHT2SqfreeCrossAttentionCache` + `WanSqfreeCrossAttnCachingLinear`
    from the submodule's sqfree tool — these expose a get_or_compute API
    rather than the put/get used by the WHT cache, so cannot be dropped into
    the WHT node without a second wrapper class.
    """

    CATEGORY = "shannon-prime"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = ("Sqfree+spinor aggressive variant of the Wan cross-attention cache. "
                   "Higher compression at equivalent quality on Q8+ backbones.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "band_bits": ("STRING", {"default": "3,3,3,3,3",
                                         "tooltip": "5-band torus-aligned bit allocation (default aggressive 3/3/3/3/3)"}),
                "residual_bits": ("INT", {"default": 3, "min": 1, "max": 4,
                                          "tooltip": "N-bit residual quantization. 3 is the Shannon saturation point."}),
                "use_spinor": ("BOOLEAN", {"default": True,
                                           "tooltip": "Enable SU(2) spinor sheet-bit correction at the causal-mask boundary"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def patch(self, model, band_bits: str, residual_bits: int, use_spinor: bool):
        band_bits_list = _parse_bits_csv(band_bits, [3, 3, 3, 3, 3], 5)

        from shannon_prime_comfyui_sqfree import (  # local import: only needed for sqfree path
            VHT2SqfreeCrossAttentionCache,
            WanSqfreeCrossAttnCachingLinear,
        )

        patched = model.clone()
        blocks = list(_iter_wan_blocks(patched))
        if not blocks:
            print("[Shannon-Prime SQFREE] no Wan blocks found on model — passing through")
            return (patched,)

        head_dim = blocks[0][1].self_attn.head_dim
        max_blocks = len(blocks)

        cache = VHT2SqfreeCrossAttentionCache(
            head_dim=head_dim,
            max_blocks=max_blocks,
            use_spinor=bool(use_spinor),
            residual_bits=int(residual_bits),
            band_bits=band_bits_list,
        )

        wrapped = 0
        for i, blk in blocks:
            cross_attn = getattr(blk, "cross_attn", None)
            if cross_attn is None:
                continue
            for suffix, attr in (("k", "k"), ("v", "v"), ("kimg", "k_img"), ("vimg", "v_img")):
                lin = getattr(cross_attn, attr, None)
                if lin is None or isinstance(lin, WanSqfreeCrossAttnCachingLinear):
                    continue
                wrapper = WanSqfreeCrossAttnCachingLinear(lin, cache, f"block_{i}_{suffix}")
                setattr(cross_attn, attr, wrapper)
            wrapped += 1

        patched._sp_sqfree_cache = cache
        pad_dim = cache._cache.pad_dim  # internal, but useful for the log

        print(f"[Shannon-Prime SQFREE] patched {wrapped}/{len(blocks)} Wan blocks "
              f"(head_dim={head_dim} -> pad_dim={pad_dim}, bands={band_bits_list}, "
              f"residual_bits={residual_bits}, spinor={use_spinor})")

        return (patched,)


class ShannonPrimeWanSelfExtract:
    """
    Phase 12 diagnostic: captures Wan self-attention K vectors during denoising.

    Hooks into `blk.self_attn.k` for every WanAttentionBlock and captures
    the K projection output at the specified denoising step (0 = first/noisiest,
    mid = structural regime, late = texture regime).

    Output saved as .npz compatible with sp_diagnostics.py:
        k_vectors shape: (n_blocks, n_kv_heads, n_tokens, head_dim)

    Usage after running:
        python sp_diagnostics.py --input wan_self_attn.npz --sqfree --layer-period 4

    Connect BEFORE the sampler. The MODEL output is the same model — this node
    is an observer, not a patcher. The .npz is written once the target step
    fires, then hooks are removed automatically.
    """

    CATEGORY = "shannon-prime/diagnostics"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "attach"
    DESCRIPTION = ("Captures Wan self-attention K vectors at a target denoising step "
                   "and saves them as .npz for sp_diagnostics Phase 12 analysis.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "capture_step": ("INT", {
                    "default": 25, "min": 0, "max": 200,
                    "tooltip": "Denoising step at which to capture K (0=first/highest-sigma)"}),
                "max_tokens": ("INT", {
                    "default": 256, "min": 64, "max": 4096,
                    "tooltip": "Max token positions to store per block (caps memory)"}),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory for .npz output (default: ComfyUI output/shannon_prime/)"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")

    def attach(self, model, capture_step=25, max_tokens=256, output_dir=""):
        import folder_paths as _fp
        import numpy as np, json, time

        if not output_dir:
            output_dir = os.path.join(_fp.get_output_directory(), "shannon_prime")
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"wan_self_attn_step{capture_step}.npz")

        patched = model.clone()
        blocks  = list(_iter_wan_blocks(patched))
        if not blocks:
            print("[SP SelfExtract] no Wan blocks found — passing through")
            return (patched,)

        n_blocks  = len(blocks)
        head_dim  = blocks[0][1].self_attn.head_dim

        # Shared mutable state across hooks (dict avoids closure rebind issues)
        state = {
            "step":      0,         # counts how many times block-0's K fired
            "captured":  {},        # block_idx -> torch.Tensor (n_kv_heads, T, head_dim)
            "target":    capture_step,
            "done":      False,
            "handles":   [],
            "max_tok":   max_tokens,
            "head_dim":  head_dim,
        }

        def make_hook(block_idx):
            def _hook(module, _inp, output):
                if state["done"]:
                    return
                # Count steps via block 0 as the "clock" block
                if block_idx == 0:
                    state["step"] += 1

                if state["step"] - 1 == state["target"]:
                    # output: (batch*tokens, n_kv_heads*head_dim)  or
                    #         (batch, tokens, n_kv_heads*head_dim)
                    k = output.detach().cpu().float()
                    if k.dim() == 2:
                        # (B*T, D) — reshape; we don't know B, assume B=1
                        k = k.unsqueeze(0)          # (1, B*T, D)
                    # k: (B, T, D)  where D = n_kv_heads * head_dim
                    T  = k.shape[1]
                    D  = k.shape[2]
                    hd = state["head_dim"]
                    n_kv = max(1, D // hd) if hd > 0 else 1

                    k = k[0]                         # (T, D) — take first batch
                    T_cap = min(T, state["max_tok"])
                    k = k[:T_cap]                    # (T_cap, D)
                    k = k.reshape(T_cap, n_kv, hd)  # (T_cap, n_kv, hd)
                    k = k.permute(1, 0, 2)           # (n_kv, T_cap, hd)
                    state["captured"][block_idx] = k

                # After capturing all blocks at the target step, save
                if (state["step"] - 1 == state["target"]
                        and block_idx == n_blocks - 1
                        and not state["done"]):
                    state["done"] = True
                    _save(state, out_path, n_blocks, capture_step, head_dim)
                    for h in state["handles"]:
                        h.remove()
                    state["handles"].clear()
            return _hook

        # Attach hooks to blk.self_attn.k (try common names)
        n_hooked = 0
        for i, blk in blocks:
            sa = getattr(blk, "self_attn", None)
            if sa is None:
                continue
            k_lin = None
            for attr in ("k", "k_proj", "to_k", "wk", "key"):
                candidate = getattr(sa, attr, None)
                if isinstance(candidate, nn.Module):
                    k_lin = candidate
                    break
            if k_lin is None:
                continue
            h = k_lin.register_forward_hook(make_hook(i))
            state["handles"].append(h)
            n_hooked += 1

        print(f"[SP SelfExtract] hooked {n_hooked}/{n_blocks} self-attn K projections "
              f"(head_dim={head_dim}, capture_step={capture_step}, "
              f"max_tokens={max_tokens})")
        print(f"[SP SelfExtract] will save to: {out_path}")

        return (patched,)


def _save(state, out_path, n_blocks, capture_step, head_dim):
    """Assemble captured K dicts → (n_blocks, n_kv_heads, T, hd) .npz."""
    import numpy as np, json

    captured = state["captured"]
    if not captured:
        print("[SP SelfExtract] Nothing captured — did the generation complete?")
        return

    # Find common n_kv_heads and T from first block
    sample = next(iter(captured.values()))   # (n_kv, T, hd)
    n_kv_heads = sample.shape[0]
    T_cap      = sample.shape[1]

    k_arr = torch.zeros(n_blocks, n_kv_heads, T_cap, head_dim)
    for idx, k in captured.items():
        # k: (n_kv, T_cap, hd) — pad if mismatched (safety)
        nk = min(k.shape[0], n_kv_heads)
        nt = min(k.shape[1], T_cap)
        k_arr[idx, :nk, :nt, :] = k[:nk, :nt, :]

    k_np = k_arr.numpy().astype("float32")

    meta = {
        "source":        "wan_self_attention",
        "capture_step":  capture_step,
        "n_blocks":      n_blocks,
        "n_kv_heads":    n_kv_heads,
        "n_tokens":      T_cap,
        "head_dim":      head_dim,
        "note": ("Wan DiT self-attention K vectors. "
                 "Axis 0 = DiT block (analogous to 'layer'), "
                 "Axis 1 = KV heads, Axis 2 = token positions, Axis 3 = head_dim. "
                 "Run: sp_diagnostics.py --input <this_file> --sqfree --layer-period 4"),
    }

    import numpy as np
    np.savez_compressed(out_path,
                        k_vectors=k_np,
                        metadata=np.array(json.dumps(meta)))

    print(f"[SP SelfExtract] saved {k_np.shape} K vectors -> {out_path}")
    print(f"[SP SelfExtract] Next: python sp_diagnostics.py "
          f"--input {out_path} --sqfree --layer-period 4 --global-offset 3")


NODE_CLASS_MAPPINGS = {
    "ShannonPrimeWanCache":        ShannonPrimeWanCache,
    "ShannonPrimeWanCacheStats":   ShannonPrimeWanCacheStats,
    "ShannonPrimeWanCacheSqfree":  ShannonPrimeWanCacheSqfree,
    "ShannonPrimeWanSelfExtract":  ShannonPrimeWanSelfExtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShannonPrimeWanCache":       "Shannon-Prime: Wan Cross-Attn Cache",
    "ShannonPrimeWanCacheStats":  "Shannon-Prime: Cache Stats",
    "ShannonPrimeWanCacheSqfree": "Shannon-Prime: Wan Cross-Attn Cache (Sqfree)",
    "ShannonPrimeWanSelfExtract": "Shannon-Prime: Wan Self-Attn Extract (Phase 12)",
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShannonPrimeWanCache": "Shannon-Prime Wan Cache",
    "ShannonPrimeWanCacheStats": "Shannon-Prime Wan Cache Stats",
    "ShannonPrimeWanCacheSqfree": "Shannon-Prime Wan Cache (Sqfree+Spinor)",
}
