# Shannon-Prime + ComfyUI: Integration Guide

## Install

1. Clone with submodule:
   ```bash
   git clone --recursive https://github.com/nihilistau/shannon-prime-comfyui.git
   ```

2. Symlink or copy this directory into your ComfyUI custom_nodes folder:
   ```cmd
   mklink /J "D:\ComfyUI\custom_nodes\shannon-prime-comfyui" "D:\path\to\shannon-prime-comfyui"
   ```
   (On Linux/macOS: `ln -s /path/to/shannon-prime-comfyui /path/to/ComfyUI/custom_nodes/`.)

3. Start ComfyUI and confirm two nodes appear under the **shannon-prime** category:
   - **Shannon-Prime Wan Cache** (`ShannonPrimeWanCache`)
   - **Shannon-Prime Wan Cache Stats** (`ShannonPrimeWanCacheStats`)

## Quick usage

Insert **Shannon-Prime Wan Cache** between your Wan UNET loader and KSampler:

```
UNETLoader / UnetLoaderGGUF  ──►  Shannon-Prime Wan Cache  ──►  KSampler
                                        ▲
                               [k_bits / v_bits / möbius]
```

Defaults: `k_bits=5,4,4,3`, `v_bits=5,4,4,3`, `use_mobius=True`. Both K and V
get 4 bands and Möbius reorder because cross-attention in Wan has **no RoPE
on K/V** — the spectral asymmetry that applies to self-attention K (which
does carry RoPE) does not apply here. See
`lib/shannon-prime/docs/INTEGRATION-COMFYUI.md` for the architectural details.

For **Wan 2.2 MoE (A14B)** models with two experts (high_noise +
low_noise), apply the node separately to each expert's MODEL — the two
caches are naturally partitioned because the experts are separate Python
objects in ComfyUI's graph.

## Supported Wan variants

| Model | dim | heads | layers | head_dim | MoE | Tested |
|---|---|---|---|---|---|---|
| Wan 2.1 14B | 5120 | 40 | 40 | 128 | No | (via source) |
| Wan 2.1 1.3B | 1536 | 12 | 30 | 128 | No | (via source) |
| Wan 2.2 A14B T2V | 5120 | 40 | 40 | 128 | Yes | (via source) |
| Wan 2.2 TI2V-5B | 3072 | 24 | 30 | 128 | No | **Yes — see outputs/** |

## End-to-end walkthrough (Wan 2.2 TI2V-5B Q8_0)

Start ComfyUI so the whitelist loads only what we need (avoids unrelated
custom-node import failures that block server startup on some installs):

```cmd
cd D:\ComfyUI
.venv\Scripts\python.exe -u main.py --listen 127.0.0.1 --port 8188 ^
    --disable-auto-launch --disable-all-custom-nodes ^
    --whitelist-custom-nodes ComfyUI-GGUF shannon-prime-comfyui
```

Queue the two example workflows:

```bash
python scripts/run_workflow.py workflows/wan22_ti2v_5b_vht2.json
python scripts/run_workflow.py workflows/wan22_ti2v_5b_baseline.json
```

Expected output: `sp_wan_vht2_00001_.webp` and `sp_wan_baseline_00001_.webp`
in `D:\ComfyUI\output\`. Copies of both (from the reference run) are in
`outputs/` at the repo root.

## Results from the reference run

- Model: `Wan2.2-TI2V-5B-Q8_0.gguf` (30 blocks, hd=128, n_heads_kv=24)
- Hardware: RTX 2060 (12 GB), Windows
- Prompt: "a red fox walking slowly through a snowy forest clearing…"
- Resolution: 480×320, length=25 frames, 8 euler/simple steps, cfg=5.0
- Patch fired: **30/30 blocks patched**, cross-attn K/V wrapped with
  `_SPCachingLinear` on `cross_attn.{k, v}` (and `.k_img/.v_img` would be
  wrapped on I2V variants; TI2V-5B only has T2V cross-attn).
- Compression reported: 3.56× on cross-attn K/V.

| Metric | Baseline | VHT2 | Note |
|---|---|---|---|
| Sampler time (8 steps) | 11.76s (1.47 s/it) | 12.16s (1.52 s/it) | +3.4% per step |
| Full prompt_exec | 16.20s | 70.84s | Cold model load dominated VHT2 wall |
| Output file | `sp_wan_baseline_00001_.webp` (228 KB) | `sp_wan_vht2_00001_.webp` (229 KB) | Both valid video |

## Cache invalidation (fingerprint-based, as shipped)

The published paper reports a **1.20× cross-attention speedup** on 50-step
Wan 2.2 generations from the cache hitting on timesteps 2..N after
computing on timestep 1. To realise that speedup the cache must *hit*
across timesteps — and the ComfyUI sampler re-allocates the T5/UMT5
conditioning tensor on every timestep, so `data_ptr()`-based invalidation
never hits.

`_SPCachingLinear` therefore keys the cache on a **content fingerprint** of
the input:

```python
def _input_fingerprint(x):
    flat = x.view(-1) if x.is_contiguous() else x.reshape(-1)
    n = flat.numel()
    return (
        tuple(x.shape),
        x.dtype,
        float(flat[0].item()),
        float(flat[n // 2].item()),
        float(flat[n - 1].item()),
    )
```

Three flat-index anchors + shape + dtype is enough to tell "same T5 output
across timesteps" apart from "new prompt → refill cache" without scanning
the full tensor. Each `.item()` forces one single-element device→host sync
— ~10 µs total per cross-attn forward, invisible against the compute it
saves.

### Measured effect (Wan 2.2 TI2V-5B, 8 euler/simple steps, 480×320×25 frames, RTX 2060)

Per-step sampler time:

| Step | ptr-based (prior, all miss) | fingerprint (hit after step 1) |
|---|---|---|
| 1 | 1.96 s/it | 1.83 s/it (miss) |
| 2 | 1.66 | 1.56 |
| 3 | 1.57 | 1.48 |
| 4 | 1.52 | 1.44 |
| 5 | 1.50 | 1.42 |
| 6 | 1.48 | 1.40 |
| 7 | 1.47 | 1.40 |
| 8 | 1.47 | **1.39** (hit) |
| avg | **1.52** | **1.44** |

Prompt-exec total time:

| | Baseline (no SP) | SP ptr-based | **SP fingerprint** |
|---|---|---|---|
| prompt_exec | 16.20 s (warm) | 70.84 s (cold) | **66.27 s (cold)** |
| 8-step sampler | 11.76 s | 12.16 s | **11.52 s** |

The 8-step sampler under VHT2 with fingerprint caching is now **~2%
faster** than the baseline 8-step sampler, not slower. The step-8 time
(1.39 s/it) vs step-1 time (1.83 s/it) shows the cache hits are paying
off: later steps save the `cross_attn.k(context)` + `cross_attn.v(context)`
linear projections across all 30 blocks, replacing them with one VHT2
dequantize per (block, k|v) pair.

For 50-step generations (the paper's target), the ratio of hit-steps to
miss-steps is 49:1 instead of 7:1, so the measured speedup should reach
or exceed the paper's 1.20×.

Outputs from both configurations are byte-valid animated WebPs; the
deterministic compress/decompress math is unchanged (still
`lib/shannon-prime/tests/test_comfyui.py` 25/25).

## Configuration

Widgets on the `ShannonPrimeWanCache` node:

| Widget | Default | Description |
|---|---|---|
| `k_bits` | `5,4,4,3` | K band bit allocation. 4 bands. Unlike self-attention's 5/5/4/3 (which matches K's VHT2 energy decay from RoPE), cross-attn K has no RoPE so all four bands carry similar energy — 5/4/4/3 is a reasonable flat-ish allocation. |
| `v_bits` | `5,4,4,3` | V band bit allocation. Same rationale as K. In self-attention V gets flat 3-bit (1 band); in cross-attention both K and V are spectrally similar so they share the 4-band scheme. |
| `use_mobius` | `True` | Möbius squarefree-first reordering. Applied to both K and V (unlike self-attention where only K gets Möbius). |

Extending to other bit allocations: the cache uses the same
`VHT2CrossAttentionCache` as the standalone wrapper in
`lib/shannon-prime/tools/shannon_prime_comfyui.py`; any allocation the
core library accepts works here.
