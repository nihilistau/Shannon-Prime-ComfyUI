# Shannon-Prime ComfyUI — Node Reference

Eight nodes across three categories: production (ship path), experimental, and diagnostics.

---

## Production Nodes

### 1. ShannonPrimeWanCache

**Display:** Shannon-Prime: Wan Cross-Attn Cache
**Category:** shannon-prime

Patches every `WanAttentionBlock.cross_attn` to cache text encoder K/V projections on CPU. T5/UMT5 text embeddings are constant across all denoising steps — the K/V projections are therefore identical on every step. This node computes them once (step 1) and returns the cached result on all subsequent steps.

Cache invalidation uses content fingerprinting (three flat-index samples + shape + dtype) rather than pointer identity, so ComfyUI's per-step tensor reallocations do not break the cache. The patch is idempotent — applying twice does not double-wrap.

Phase 15 LEAN: the cache is stored as raw CPU tensors (fp16). No VHT2 compression, no GPU temporaries on step 1. This eliminates the warm-cache overhead from earlier phases.

| Input | Type | Default | Notes |
|---|---|---|---|
| `model` | MODEL | — | Wan model from any loader |
| `k_bits` | STRING | `"5,4,4,3"` | K band bit allocation (4 VHT2 frequency bands). Currently informational only in Phase 15 LEAN. |
| `v_bits` | STRING | `"5,4,4,3"` | V band bit allocation. |
| `use_mobius` | BOOLEAN | `True` | Möbius squarefree-first reorder. Currently informational only in Phase 15 LEAN. |

**Output:** MODEL (patched)

For MoE models (Wan 2.2 A14B), apply one WanCache per expert MODEL object.

---

### 2. ShannonPrimeWanBlockSkip

**Display:** Shannon-Prime: Wan Block Skip (Phase 15)
**Category:** shannon-prime

The primary performance node. Patches `WanAttentionBlock.forward()` to cache and skip computation at the block level. Three things can be cached per block:

1. **Self-attention output (y)** — always cached when the block's tier window > 0.
2. **Cross-attention output** — cached on the same window as self-attention. Since the text side is frozen, cross-attn output is even more stable than self-attn.
3. **FFN pre-gate output** (TURBO mode) — cached when `cache_ffn=True`.

On a cache-hit step, the block's computation reduces to:
- Recompute adaLN modulation from the current timestep embedding (6 floats, trivial)
- Load cached self-attn y from CPU, multiply by current gate
- Load cached cross-attn output from CPU, add
- (TURBO) Load cached FFN output from CPU, multiply by current gate, add
- Return

All the expensive computation (Q/K/V projections, attention scores, output projection, cross-attention forward, FFN forward) is skipped.

**Block tier map** (derived from sigma-sweep Phase 12 diagnostics on Wan 2.2):

| Tier | Blocks | Stability | Default Window | Streak-miss |
|---|---|---|---|---|
| 0 — Permanent Granite | L00–L03 | cos_sim > 0.95 across 10 steps | 10 | 10 |
| 1 — Stable Sand | L04–L08 | Moderate stability | 3 | 5 |
| 2 — Volatile | L09–L15 | Lower stability | 0 (disabled) | 3 |
| 3 — Deep/Late | L16–L39 | Texture detail | 0 (disabled) | 3 |

| Input | Type | Default | Notes |
|---|---|---|---|
| `model` | MODEL | — | Apply after WanCache |
| `tier_0_window` | INT | `10` | L00–L03. 0=disabled. |
| `tier_1_window` | INT | `3` | L04–L08. 0=disabled. |
| `tier_2_window` | INT | `0` | L09–L15. Try 2–5 for speed. |
| `tier_3_window` | INT | `0` | L16–L39. YOLO: try 2–3 for SVI. |
| `cache_ffn` | BOOLEAN | `False` | TURBO: cache FFN output. Near-zero compute on hits. |
| `cache_dtype` | dropdown | `fp16` | `fp16`, `fp8`, or `mixed`. Mixed = fp16 for tier-0/1, fp8 for tier-2/3. |
| `verbose` | BOOLEAN | `False` | Per-block HIT/MISS console logs. |

**Output:** MODEL (patched)

**Cache storage:** all caches are CPU-resident. fp16 caches at 720p are ~100MB per tensor per block. TURBO with all 40 blocks (3 tensors each) at fp16 = ~12GB CPU. At fp8 or mixed, ~7–8GB. If CPU memory fills, model weights get offloaded to disk and step times degrade catastrophically — reduce tier windows or switch to fp8/mixed.

**fp8 on Turing (RTX 2060):** Turing GPUs have no hardware fp8, so the load path casts fp8→fp16 on CPU first, then transfers to GPU. This avoids a CUDA hang that occurs when doing fp8→fp16 cast on-device on Turing.

---

### 3. ShannonPrimeWanCacheFlush

**Display:** Shannon-Prime: Cache Flush (before VAE)
**Category:** shannon-prime

**Essential for any workflow using BlockSkip.** Place between KSampler output and VAEDecode. Clears all BlockSkip caches (self-attn, cross-attn, FFN) and all cross-attn `_SPCachingLinear` wrapper caches, then calls `torch.cuda.empty_cache()`.

Without this node, cached tensors remain allocated during VAE decode, competing for VRAM and causing massive slowdowns (34s+ overhead at 720p) or OOM at higher resolutions.

| Input | Type | Notes |
|---|---|---|
| `model` | MODEL | The patched model |
| `samples` | LATENT | From KSampler (passed through unchanged) |

**Output:** LATENT (pass-through)

Safe to include even if BlockSkip is not in the workflow — the flush is a no-op on unpatched models.

---

### 4. ShannonPrimeWanCacheStats

**Display:** Shannon-Prime: Cache Stats
**Category:** shannon-prime

Observer node. Reports cache hit/miss stats from a WanCache-patched model. Wire anywhere in the model chain. Pass-through — does not modify the model.

**Output:** MODEL (pass-through)

---

### 5. ShannonPrimeWanCacheSqfree

**Display:** Shannon-Prime: Wan Cross-Attn Cache (Sqfree)
**Category:** shannon-prime

Aggressive compression variant using sqfree prime-Hartley basis + Möbius CSR predictor + optional SU(2) spinor sheet-bit correction. Targets Q8+ backbones where higher compression is valuable.

| Input | Type | Default | Notes |
|---|---|---|---|
| `model` | MODEL | — | Wan model |
| `band_bits` | STRING | `"3,3,3,3,3"` | 5-band torus-aligned allocation |
| `residual_bits` | INT | `3` | N-bit residual quantization (1–4). 3 = Shannon saturation point. |
| `use_spinor` | BOOLEAN | `True` | SU(2) sheet-bit correction at causal boundary |

**Output:** MODEL (patched)

---

## Experimental Nodes

### 6. ShannonPrimeWanSigmaSwitch

**Display:** Shannon-Prime: Sigma Switch (experimental)
**Category:** shannon-prime

Attaches to the model's forward pass via `model_function_wrapper` and adjusts BlockSkip's effective windows at each step based on the current sigma. High sigma → wider windows (early steps are noisier, caching is safer). Low sigma → narrower windows (late steps refine detail, recompute more often).

| Input | Type | Default | Notes |
|---|---|---|---|
| `model` | MODEL | — | BlockSkip-patched model |
| `high_sigma_mult` | FLOAT | `1.5` | Window multiplier at high sigma |
| `low_sigma_mult` | FLOAT | `0.5` | Window multiplier at low sigma |
| `sigma_split_frac` | FLOAT | `0.5` | Fraction of sigma range for HIGH→LOW switch |
| `verbose` | BOOLEAN | `False` | Log sigma and window decisions |

**Output:** MODEL (patched)

---

## Diagnostic Nodes

### 7. ShannonPrimeWanRicciSentinel

**Display:** Shannon-Prime: Ricci Sentinel (diagnostic)
**Category:** shannon-prime

Per-step sigma regime and cache window timeline reporter. Prints `e_mag`, HIGH/LOW regime label, and effective windows at each step. Prints compact summary table at generation boundaries.

### 8. ShannonPrimeWanSelfExtract

**Display:** Shannon-Prime: Wan Self-Attn Extract (diagnostic)
**Category:** shannon-prime/diagnostics

Phase 12 analysis tool. Hooks self-attention K projections and saves `.npz` files for offline analysis with `sp_diagnostics.py`. Used to derive the tier map and measure per-block stability across denoising steps.

| Input | Type | Default | Notes |
|---|---|---|---|
| `model` | MODEL | — | Wan model |
| `capture_step` | INT | `25` | Denoising step to capture |
| `max_tokens` | INT | `256` | Max token positions stored per block |
| `output_dir` | STRING | `""` | Output dir (defaults to `ComfyUI/output/shannon_prime/`) |

**Output:** MODEL (hooks attached, observer only)
