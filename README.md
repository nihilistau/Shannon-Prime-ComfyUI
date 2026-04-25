# Shannon-Prime for ComfyUI

**Accelerate Wan 2.x video generation by caching what doesn't change.**

Shannon-Prime is a set of custom ComfyUI nodes that exploit structural invariants in the Wan DiT architecture to skip redundant computation during video denoising. Cross-attention context (T5/UMT5 text embeddings) is constant across all denoising steps, and many self-attention blocks are geometrically stable for long stretches. Shannon-Prime caches both, re-applying only the cheap adaLN gating from the current timestep so sigma tracking stays accurate.

The result: **7 s/step on hardware that does 32 s/step stock** (RTX 2060 12GB, Wan 2.2 TI2V-5B Q8, 720p, `--lowvram`). No quality loss on stable blocks; configurable aggressiveness for the rest.

---

## Key Features

**Cross-Attention Caching** — text encoder K/V projections are identical every step. Compute once, serve from CPU cache forever. Zero overhead after step 1.

**Block-Level Self-Attention Skip** — for stable DiT blocks, cache the entire self-attention output. On hit steps, skip Q/K/V projections, attention scores, and output projection entirely. Only recompute the adaLN gate (trivial) and add the cached result.

**Cross-Attention Output Caching** — the full cross-attention output (not just K/V) is cached alongside self-attention. Since the text side is frozen, cross-attention output is even more stable than self-attention.

**TURBO Mode** — optionally cache the FFN pre-gate output too. On hit steps, an entire DiT block reduces to: adaLN modulation + three CPU→GPU tensor loads + three additions. Near-zero compute.

**4-Tier Block System** — blocks are grouped by empirically measured stability. Each tier has independent cache windows, so you can be aggressive on granite-stable early blocks and conservative (or off) on volatile late blocks.

**Mixed Precision Caching** — cache in fp16, fp8, or mixed (fp16 for precision-sensitive tiers, fp8 for aggressive tiers). fp8 halves CPU memory and PCIe transfer time. Mixed gives you both: precision where it matters, savings where the approximation is already aggressive.

**SVI Compatible** — works with Step-Video Inference 6-step distilled Wan 2.2 workflows, including non-monotonic sigma schedules and dual hi/lo-noise GGUF loading.

---

## Performance

Measured on RTX 2060 12GB + 32GB system RAM, Wan 2.2 TI2V-5B Q8, 720p 9 frames, `--lowvram`:

| Configuration | Step Time | Notes |
|---|---|---|
| Stock (no Shannon-Prime) | ~32 s/step | Baseline |
| Cross-attn cache only | ~28 s/step | Eliminates redundant text K/V |
| + BlockSkip tier-0/1 | ~15 s/step | Skips self-attn on 9 stable blocks |
| + Cross-attn output cache | ~8.4 s/step | Skips cross-attn compute on hit |
| + TURBO (all 40 blocks, fp8) | ~7.0 s/step | Near-zero compute on cached steps |

Output quality: visually identical at tier-0/1 defaults. Tier-2/3 with TURBO trades minor detail for significant speed. The tradeoff is configurable per-tier.

---

## Supported Models

| Model | Type | Status |
|---|---|---|
| Wan 2.2 TI2V-5B | Dense | Fully tested, primary target |
| Wan 2.2 A14B T2V/I2V | MoE (2 experts) | Full — expert-aware caching |
| Wan 2.1 14B | Dense | Full |
| Wan 2.1 1.3B | Dense | Full |

For MoE models with separate expert MODEL objects, apply `ShannonPrimeWanCache` to each expert independently.

---

## Installation

**Option 1: Clone into custom_nodes (recommended)**
```bash
cd /path/to/ComfyUI/custom_nodes
git clone --recursive https://github.com/nihilistau/shannon-prime-comfyui.git
```

If you already cloned without `--recursive`:
```bash
cd shannon-prime-comfyui
git submodule update --init --recursive
```

**Option 2: Symlink (Windows, elevated prompt)**
```batch
mklink /J C:\ComfyUI\custom_nodes\shannon-prime-comfyui D:\path\to\shannon-prime-comfyui
```

**Dependencies:** None beyond ComfyUI itself. All math lives in the `lib/shannon-prime` submodule (pure Python + optional C/CUDA). No pip packages required.

---

## Quick Start

### Minimal (cross-attention caching only)

Wire one node between your model loader and sampler:

```
UnetLoaderGGUF → ShannonPrimeWanCache → KSampler → VAEDecode
```

Leave all defaults. Eliminates redundant text K/V computation on every step after the first.

### Recommended (cross-attn + block skip)

```
UnetLoaderGGUF
  → ShannonPrimeWanCache          (cross-attn K/V caching)
    → ShannonPrimeWanBlockSkip    (self-attn + cross-attn + optional FFN skip)
      → KSampler
        → ShannonPrimeWanCacheFlush   (free memory before VAE)
          → VAEDecode
            → SaveAnimatedWEBP
```

**CacheFlush is essential** — without it, cached tensors stay allocated during VAE decode and can cause massive slowdowns or OOM.

### FULL SEND (maximum speed)

Set BlockSkip to:
- `tier_0_window=10`, `tier_1_window=3`, `tier_2_window=5`, `tier_3_window=3`
- `cache_ffn=True` (TURBO mode)
- `cache_dtype=mixed` (fp16 for tier-0/1, fp8 for tier-2/3)

This caches all 40 blocks across self-attn, cross-attn, and FFN. On cache-hit steps, entire blocks reduce to adaLN + three tensor additions. Achieves ~7 s/step on hardware that does 32 s/step stock.

**Caveat:** TURBO at high resolutions (10K+ tokens) can exhaust CPU memory. If you see step times degrading across outputs, reduce tier-2/3 windows or disable `cache_ffn`.

---

## Nodes

### ShannonPrimeWanCache

**Cross-attention K/V caching.** Patches every `WanAttentionBlock.cross_attn` to cache text encoder K/V projections on CPU after the first step. Content-based fingerprinting ensures cache correctness across ComfyUI's per-step tensor reallocations.

| Input | Type | Default | Description |
|---|---|---|---|
| `model` | MODEL | — | Wan model from any loader |
| `k_bits` | STRING | `"4,3,3,3"` | K band bit allocation (4 VHT2 frequency bands, low→high). Ignored in LEAN mode (raw CPU cache). |
| `v_bits` | STRING | `"4,3,3,3"` | V band bit allocation. Ignored in LEAN mode. |
| `use_mobius` | BOOLEAN | `False` | Möbius squarefree-first reorder. Off by default — LEAN mode uses raw CPU cache with zero overhead. |

### ShannonPrimeWanBlockSkip

**Block-level computation skip.** The primary performance node. Caches self-attention output, cross-attention output, and optionally FFN output per block. On cache-hit steps, skips all heavy computation and applies only the cheap adaLN gating.

| Input | Type | Default | Description |
|---|---|---|---|
| `model` | MODEL | — | Apply after WanCache |
| `tier_0_window` | INT | `10` | Cache window for L00–L03 (Permanent Granite). Most stable blocks. |
| `tier_1_window` | INT | `3` | Cache window for L04–L08 (Stable Sand). |
| `tier_2_window` | INT | `0` | Cache window for L09–L15 (Volatile). 0=disabled. Try 2–5 for speed. |
| `tier_3_window` | INT | `0` | Cache window for L16–L39 (Deep/late). 0=disabled. YOLO: try 2–3. |
| `cache_ffn` | BOOLEAN | `False` | TURBO: cache FFN pre-gate output. Hit steps become near-zero compute. |
| `cache_dtype` | `fp16`/`fp8`/`mixed` | `fp16` | Cache precision. mixed = fp16 for tier-0/1, fp8 for tier-2/3. |
| `verbose` | BOOLEAN | `False` | Per-block HIT/MISS console logging. |

**Tier map:**

| Tier | Blocks | Stability | Default Window | Streak |
|---|---|---|---|---|
| 0 — Permanent Granite | L00–L03 | cos_sim > 0.95 for 10+ steps | 10 | 10 |
| 1 — Stable Sand | L04–L08 | Moderate stability | 3 | 5 |
| 2 — Volatile | L09–L15 | Lower stability | 0 (off) | 3 |
| 3 — Deep/Late | L16–L39 | Texture detail | 0 (off) | 3 |

### ShannonPrimeWanCacheFlush

**Memory cleanup.** Place between KSampler output and VAEDecode. Clears all BlockSkip caches (self-attn, cross-attn, FFN) and cross-attn `_SPCachingLinear` wrappers, then calls `torch.cuda.empty_cache()`. Without this, cached tensors compete with VAE for VRAM.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | The patched model |
| `samples` | LATENT | Latent from KSampler (passed through unchanged) |

### ShannonPrimeWanCacheStats

**Observer.** Reports cache hit/miss statistics. Wire anywhere in the model chain to see hit rates and compression ratios. Pass-through — does not modify the model.

### ShannonPrimeWanCacheSqfree

**Aggressive compression variant.** Uses sqfree prime-Hartley basis + Möbius CSR predictor + optional SU(2) spinor sheet-bit correction. Higher compression for Q8+ backbones at the cost of more computation.

| Input | Type | Default | Description |
|---|---|---|---|
| `model` | MODEL | — | Wan model |
| `band_bits` | STRING | `"3,3,3,3,3"` | 5-band torus-aligned allocation |
| `residual_bits` | INT | `3` | N-bit residual quantization (1–4) |
| `use_spinor` | BOOLEAN | `True` | SU(2) sheet-bit correction at causal boundary |

### ShannonPrimeWanSigmaSwitch

**Sigma-adaptive cache windows (experimental).** Adjusts BlockSkip windows based on current denoising sigma. High sigma (noisy, early steps) → wider windows. Low sigma (refined, late steps) → narrower windows. Hooks into the model's sigma schedule via `model_function_wrapper`.

### ShannonPrimeWanRicciSentinel

**Diagnostic.** Per-step sigma regime and cache window timeline reporter. Prints `e_mag`, HIGH/LOW regime label, and effective windows at each step. Prints a compact summary table at generation boundaries.

### ShannonPrimeWanSelfExtract

**Phase 12 diagnostic.** Hooks self-attention K projections and saves `.npz` files for offline analysis with `sp_diagnostics.py`. Used to derive the tier map and measure per-block stability.

---

## Tuning Guide

**Start conservative, then open up.** The defaults (tier-0=10, tier-1=3, everything else off) are safe for any Wan model and prompt. From there:

1. **Enable tier-2** (`tier_2_window=3`): adds 7 more blocks to caching. Watch for quality changes in fine detail.
2. **Enable tier-3** (`tier_3_window=2`): caches all 40 blocks. Texture detail may soften slightly.
3. **Enable TURBO** (`cache_ffn=True`): skips FFN on hit steps. Maximum speed. Quality impact depends on schedule length — fine for 6-step SVI, monitor for 20+ steps.
4. **Switch to mixed dtype** (`cache_dtype=mixed`): saves ~40% CPU memory on tier-2/3 caches with negligible quality impact (those tiers are already approximate).
5. **Full fp8** (`cache_dtype=fp8`): maximum memory savings. Precision-sensitive content (text rendering, faces) may show minor artifacts.

**Memory budget:** each cached block stores up to 3 tensors (self-attn y, cross-attn output, FFN output). At 720p in fp16, that's roughly 100MB per tensor. TURBO with all 40 blocks at fp16 would be ~12GB CPU. At fp8 or mixed, roughly 7–8GB. If you see step time degradation across sequential outputs, CPU memory is full and model weights are being offloaded — reduce tier windows or enable fp8.

---

## Bit Allocation Guide

Shannon-Prime's VHT2 compression decomposes attention vectors into spectral bands, with configurable bits per band. The cross-attention node (`WanCache`) defaults to LEAN mode (raw CPU caching, zero VHT2 overhead) — bit settings only apply when `cache_compress=vht2` is enabled on BlockSkip, or when using the `CacheSqfree` node.

**The scaling law** from *Multiplicative Lattice Combined* governs what you can get away with:

    ΔPPL/PPL = exp(4700 · (1 − K_corr)² / (params^1.1 · bits^1.5)) − 1

For Wan 2.2 14B at bf16 (params=14, bits=16), the safe K-corr floor at 3% PPL budget is **0.914** — far more tolerant than small quantized models. This means aggressive bit reduction is viable.

**Recommended configurations for Wan:**

| Config | K bits | V bits | Bands | Compression | Use Case |
|---|---|---|---|---|---|
| Safe | `4,3,3,3` | `4,3,3,3` | 4 | ~3× | Default for VHT2 mode |
| Aggressive | `3,3,3,3` | `3,3,3` | 4/3 | ~3.5× | Wan 14B bf16 — scaling law says viable |
| Maximum | `3,2,2,2` | `3,2,2` | 4/3 | ~4.5× | Wan 14B bf16 — needs validation |
| Ultra | `2,2,2,2` | `2,2,2` | 4/3 | ~5× | Experimental — 2 bits = 4 levels only |

**Sqfree node** (5 bands, torus-aligned): default `3,3,3,3,3` is proven at 3.3× on Qwen3-8B Q8 with the spinor bit. For Wan, `3,2,2,2,2` or even `2,2,2,2,2` may work — the 14B bf16 denominator is enormous.

**Can you go below 3 bits?** Yes. 2-bit quantization gives 4 levels per coefficient. The spectral bands are roughly Gaussian-distributed, so 4 levels capture the sign and rough magnitude. The scaling law predicts that 14B bf16 models can tolerate the K-corr drop. 1-bit (2 levels: sign only) is possible but untested — it preserves spectral structure but loses all magnitude information.

**Band count: 4 vs 5?** For head_dim=128, 4 bands splits into ~32-dim chunks (ship path). 5 bands aligns with the 5 primes {2,3,5,7,11} in the sqfree basis. Both work well. More bands = finer bit allocation control but more overhead per band (scale factors). 4 is optimal for the ship path; 5 for sqfree.

---

## RoPE-ALiBi Frequency Injection

Shannon-Prime includes `sp_inject_freqs.py` — a tool that blends lattice-aligned frequencies into any GGUF model's RoPE schedule. This is a **free win**: zero retraining, zero runtime cost, measured −0.6% to −0.8% PPL improvement across architectures.

**How it works:** standard RoPE uses geometric frequencies θ_j = 10000^(-2j/d). These are 1D — they throw away the arithmetic relationships between positions. Replacing them with a blend of geometric + lattice-drawn integer frequencies makes those relationships accessible to attention. The tool writes a `rope_freqs.weight` tensor into the GGUF file; llama.cpp picks it up automatically.

**Usage:**
```bash
# Show model info
python lib/shannon-prime/tools/sp_inject_freqs.py model.gguf --info

# Inject at optimal alpha (recommended: 0.17)
python lib/shannon-prime/tools/sp_inject_freqs.py model.gguf model_sp.gguf --alpha 0.17

# Analyze without modifying
python lib/shannon-prime/tools/sp_inject_freqs.py model.gguf --analyze --alpha 0.17
```

**Results** (Position Is Arithmetic v8, Dolphin 3.0 Llama 3.2 1B): Q8 −0.82% PPL at α=0.22, Q6_K −0.66% at α=0.17, Q4_K_M −0.61% at α=0.17. Optimal α range: 0.15–0.22 (flat optimum, deployment-robust).

**Note for Wan/DiT models:** Wan uses 3D video RoPE (`get_1d_rotary_pos_embed` with separate temporal/spatial frequencies), not standard 1D RoPE. The `sp_inject_freqs.py` tool targets standard RoPE models loaded via llama.cpp. For Wan, the frequency injection would need to hook into the DiT-specific RoPE path — this is a future work item.

---

## Mixed Precision Per Layer (MoE)

For MoE architectures like Wan 2.2 A14B (2-expert) or Qwen3.6 35B-A3B, different layers have different compression profiles. Shannon-Prime supports per-tier mixed precision via `cache_dtype=mixed`:

- **Dense attention layers** (every layer in Wan 5B, every 4th in Qwen3.6): carry full K/V cache. These benefit from fp8's dynamic range when distributions are smooth.
- **MoE routing layers**: carry expert-routed K/V. These have higher variance and benefit from fp16 precision on early tiers.
- **Gated DeltaNet / SSM layers** (Qwen3.6 hybrid): contribute NO K/V to the cache at all — they use recurrent state instead.

The `cache_dtype=mixed` setting applies fp16 to tier-0/1 (precision-sensitive early blocks) and fp8 to tier-2/3 (aggressive late blocks). For MoE models with separate expert MODEL objects, apply `ShannonPrimeWanBlockSkip` to each expert independently — each expert's tier map is independent.

---

## Workflow Integration

All nodes output standard ComfyUI `MODEL` or `LATENT` types. Compatible with `KSampler`, `SamplerCustomAdvanced`, and any node that accepts MODEL or LATENT.

**Cancel/interrupt cleanup:** on cancel, the `/sp/cleanup` endpoint (auto-registered on startup) unloads models and clears CUDA cache. Alternatively, `patch()` on BlockSkip clears stale caches at the start of every queue execution automatically.

**ComfyUI launch flags for lowVRAM setups:**
```bash
python main.py --lowvram --port 8189
```

---

## How It Works

Shannon-Prime exploits the mathematical structure of the Wan DiT denoising process. The core insight is that the Vilenkin-Hartley Transform (VHT2) decomposes attention tensors into spectral bands whose energy decays predictably — enabling principled compression with known error bounds.

The cross-attention cache works because T5 text embeddings are constant: the same context vector enters every denoising step, producing identical K/V projections. Caching these after step 1 eliminates all redundant text encoder→attention computation.

The block skip works because DiT blocks in the early layers (L00–L08) produce self-attention outputs that change slowly across consecutive denoising steps. The cosine similarity between step N and step N+k remains above 0.95 for k=10 on the most stable blocks. By caching the pre-gate output and re-applying only the cheap adaLN gate from the current timestep, we maintain sigma-accurate brightness/contrast tracking while skipping all the expensive attention computation.

The TURBO mode extends this to FFN, which is also stable across cached windows. On a hit step, an entire DiT block becomes: read 6 floats from the timestep embedding → load 3 cached tensors from CPU → multiply-add → done.

For the mathematical foundations, see the papers in `lib/shannon-prime/docs/`: *Position Is Arithmetic*, *KV Cache Is A View*, and *Multiplicative Lattice Combined*.

---

## Project Structure

```
shannon-prime-comfyui/
├── __init__.py              # ComfyUI entry point, /sp/cleanup endpoint
├── pyproject.toml           # ComfyUI Manager metadata
├── nodes/
│   ├── __init__.py          # Re-exports NODE_CLASS_MAPPINGS
│   └── shannon_prime_nodes.py   # All node definitions
├── lib/
│   └── shannon-prime/       # Core math submodule (VHT2, Möbius, backends)
├── docs/                    # Technical documentation
├── workflows/               # Example ComfyUI workflows
├── web/                     # JS extension for cleanup on cancel
└── LICENSE                  # AGPLv3 + Commercial dual license
```

---

## License

**AGPLv3** for open-source, academic, and non-proprietary use. Derivative works must share alike.

**Commercial license** available for proprietary integration. Contact: Ray Daniels (raydaniels@gmail.com).

Copyright (C) 2026 Ray Daniels. See [LICENSE](LICENSE).
