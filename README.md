# Shannon-Prime for ComfyUI

**Universal spectral compression for video, image, and audio generation.**

Shannon-Prime is a suite of 16 custom ComfyUI nodes that apply the Vilenkin-Hartley Transform (VHT2) — a self-inverse spectral decomposition — to compress KV caches and skip redundant computation across every generative modality. One mathematical framework, three modalities, one set of nodes.

```
                          ┌─────────────────────────────┐
                          │     Shannon-Prime VHT2       │
                          │  Spectral KV Compression     │
                          └──────────┬──────────────────┘
                 ┌───────────────────┼───────────────────┐
           ┌─────┴─────┐     ┌──────┴──────┐     ┌──────┴──────┐
           │   VIDEO    │     │    IMAGE    │     │    AUDIO    │
           │            │     │             │     │             │
           │  Wan 2.x   │     │  Flux/SD    │     │ Stable Audio│
           │  5B / 14B  │     │  DiT/UNet   │     │ Qwen3-TTS  │
           │  MoE A14B  │     │             │     │ Voxtral 4B  │
           └────────────┘     └─────────────┘     └─────────────┘
```

The same VHT2 butterfly decomposition works everywhere because the mathematical property it exploits — RoPE imprints spectral structure on KV vectors, and that structure is compressible — is universal across all transformer architectures that use rotary position embeddings.

---

## Headline Numbers

| Modality | Model | Speedup | Mechanism |
|---|---|---|---|
| Video | Wan 2.2 5B | **4.6× step speed** (32 → 7 s/step) | Block-skip + cross-attn cache + TURBO |
| Video | Wan 2.2 A14B MoE | **3.5× step speed** | Expert-aware block-skip |
| Image | Flux v1/v2 | Block-level skip | Dual-stream + single-stream cache |
| Audio | Stable Audio | Cross-attention cache | VHT2 KV compression |
| TTS | Voxtral 4B | KV compression | Python, Rust, and C implementations |

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nihilistau/shannon-prime-comfyui
cd shannon-prime-comfyui
pip install -e .
```

Restart ComfyUI. Nodes appear under the **shannon-prime** category.

---

## Core Nodes

### ShannonPrimeWanCache

Patches Wan cross-attention to cache text encoder K/V projections. T5/UMT5 text embeddings are constant across denoising steps — K/V projections are identical on every step. This node computes them once (step 1) and returns the cached result thereafter.

Cache invalidation uses content fingerprinting (shape + dtype + three flat-index samples), not pointer identity. The patch is idempotent.

| Input | Default | Purpose |
|---|---|---|
| `model` | — | Wan model from any loader |
| `k_bits` | `"5,4,4,3"` | K band bit allocation (informational in Phase 15 LEAN) |
| `v_bits` | `"5,4,4,3"` | V band bit allocation |
| `use_mobius` | `True` | Möbius squarefree-first reorder |

### ShannonPrimeWanBlockSkip

The primary performance node. Caches and skips block-level computation:

1. **Self-attention output** — cached when the block's tier window > 0
2. **Cross-attention output** — cached on the same window
3. **FFN output** (TURBO mode) — cached when `cache_ffn=True`

On a cache-hit step, the block reduces to: load cached tensors, multiply by current adaLN gate, add, return. All expensive computation (Q/K/V projections, attention scores, FFN forward) is skipped.

**Block tier map** (from sigma-sweep Phase 12 diagnostics on Wan 2.2):

| Tier | Blocks | Stability | Default Window |
|---|---|---|---|
| 0 — Permanent Granite | L00–L03 | cos_sim > 0.95 across 10 steps | 10 |
| 1 — Stable Sand | L04–L08 | Moderate stability | 3 |
| 2 — Volatile | L09–L15 | Lower stability | 0 (disabled) |
| 3 — Deep/Late | L16–L39 | Texture detail | 0 (disabled) |

| Input | Default | Purpose |
|---|---|---|
| `model` | — | Apply after WanCache |
| `tier_0_window` | `10` | L00–L03 cache window |
| `tier_1_window` | `3` | L04–L08 cache window |
| `tier_2_window` | `0` | L09–L15 (try 2–5 for speed) |
| `tier_3_window` | `0` | L16–L39 (YOLO: try 2–3) |
| `cache_ffn` | `False` | TURBO: cache FFN output |
| `cache_dtype` | `fp16` | `fp16`, `fp8`, or `mixed` |
| `verbose` | `False` | Per-block HIT/MISS logs |

**Memory note:** All caches are CPU-resident. TURBO with all 40 blocks (3 tensors each) at fp16 720p = ~12 GB CPU. At fp8/mixed = ~7–8 GB. If CPU memory fills, model weights get offloaded to disk — reduce tier windows or switch to fp8/mixed.

**Turing GPUs (RTX 2060):** No hardware fp8. The load path casts fp8→fp16 on CPU before GPU transfer. This avoids a CUDA hang from on-device fp8→fp16 cast on Turing.

### ShannonPrimeWanCacheFlush

Flushes all Shannon-Prime caches before VAE decode. Essential for freeing GPU memory before the VAE decoder runs.

---

## Advanced Features

### 1D-Circle Granite Reconstruction

Strict 1D-circle reconstruction for tier-0 blocks. Blocks L00–L03 have outputs that lie on a 1D circle in the block's output space. Instead of caching the full output tensor, cache the 1D projection and reconstruct. Enabled via `enable_one_dim_granite`.

### Cross-Tier Energy Borrowing

When lower-stability tiers (2/3) are disabled, their unused energy budgets are redistributed to higher-stability tiers (0/1), allowing longer cache windows. Enabled via `enable_cross_tier_borrow`.

### Per-Token VHT2 Skeleton Fraction

Varies the VHT2 skeleton fraction based on denoising step position. Early steps (high noise) retain fewer coefficients; late steps (fine detail) retain more. Enabled via `enable_per_token_skeleton`.

### Lyapunov Spectrum Measurement

`ShannonPrimeWanLyapunovSnapshot` captures block-level stability data. `scripts/sp_lyapunov_analyze.py` computes Lyapunov exponents for each block, empirically validating the tier assignments.

### Higher-Order Integrators

AB2 (v4), AB3 (v5) Adams-Bashforth integrators for the block-skip prediction. `harmonic_order=3_ab3` uses third-order prediction for cache-hit outputs, reducing the quality cost of long skip windows.

---

## Workflows

Ready-to-use workflow JSON files are in `workflows/`:

### wan22_ti2v_5b_phase12_ship.json

Production Wan 2.2 TI2V-5B workflow. Tested at 1280×720, 9 frames, 20 steps on RTX 2060 6GB.

```
UnetLoaderGGUF → ShannonPrimeWanCache → ShannonPrimeWanBlockSkip
    → KSampler → ShannonPrimeWanCacheFlush → VAEDecode → SaveAnimatedWEBP
```

### Models Required

- `Wan2.2-TI2V-5B-Q8_0.gguf`
- `umt5-xxl-encoder-Q8_0.gguf`
- `wan_2.1_vae.safetensors`

---

## Node Reference

| Node | Category | Purpose |
|---|---|---|
| ShannonPrimeWanCache | Video | Cross-attention K/V caching (Wan 2.x) |
| ShannonPrimeWanBlockSkip | Video | Block-level skip caching with tier map |
| ShannonPrimeWanCacheFlush | Video | Flush caches before VAE decode |
| ShannonPrimeFluxCache | Image | Cross-attention cache for Flux DiT |
| ShannonPrimeFluxBlockSkip | Image | Block-skip for Flux dual/single streams |
| ShannonPrimeAudioCache | Audio | KV cache for Stable Audio |
| ShannonPrimeVoxtralCache | TTS | KV compression for Voxtral 4B |
| ShannonPrimeLyapunovSnapshot | Diagnostic | Capture block stability data |
| ShannonPrimeDashboard | Diagnostic | Real-time compression statistics |

See [docs/NODES.md](docs/NODES.md) for the full node reference with all inputs, outputs, and usage notes.

---

## How It Works

### Video (Wan 2.x)

In Wan 2.1/2.2, cross-attention K/V projections recompute identical T5 text embeddings every denoising step across every DiT block. Shannon-Prime computes them once, caches the result, and returns it on subsequent steps.

Block-skip caching goes further: for blocks whose self-attention outputs barely change between steps (measured by cosine similarity), the entire block computation is skipped and the cached output is reused with the current timestep's adaLN modulation applied.

The combination of cross-attention caching + block-skip + TURBO (FFN caching) eliminates 70–90% of the FLOPs per step for the most stable blocks.

### Image (Flux)

Flux DiT has dual-stream and single-stream blocks. Shannon-Prime caches cross-attention K/V in the dual-stream blocks and applies block-skip to the single-stream blocks. The stability characteristics differ from Wan (Flux has fewer but larger blocks), so the tier map is adjusted.

### Audio (Stable Audio / Voxtral)

Audio generation models with cross-attention (text-conditioned) benefit from the same K/V caching as video. Voxtral 4B TTS additionally gets VHT2 KV compression for the autoregressive decode path — the same compression that powers LLM inference.

---

## Sibling Repositories

| Repository | Role |
|---|---|
| [shannon-prime](https://github.com/nihilistau/shannon-prime) | Core math library. The torch backend runs the VHT2 math for this integration. |
| [shannon-prime-engine](https://github.com/nihilistau/shannon-prime-engine) | Standalone inference engine. The reference implementation. |
| [shannon-prime-llama](https://github.com/nihilistau/shannon-prime-llama) | llama.cpp integration for LM Studio. |

**Voxtral TTS forks** with integrated VHT2 KV compression:
[Python](https://github.com/nihilistau/ComfyUI-FL-VoxtralTTS),
[Rust](https://github.com/nihilistau/voxtral-mini-realtime-rs),
[C](https://github.com/nihilistau/voxtral-tts.c).

---

## License

Copyright (C) 2026 Ray Daniels. All Rights Reserved.

Licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3).
Commercial license available — contact raydaniels@gmail.com.
