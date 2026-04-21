# shannon-prime-comfyui

**Shannon-Prime VHT2 cross-attention caching for ComfyUI video generation**

Compresses and caches cross-attention K/V in Wan 2.1/2.2 video generation
models. Cross-attention from T5/UMT5 text embeddings is identical across all
diffusion timesteps — compute once, compress via the VHT2 spectral transform,
reconstruct on subsequent calls. Two node variants use the same underlying
transform:

- **`ShannonPrimeWanCache`** (ship path): VHT2 + Möbius reorder + 4-band
  quantization. Drop-in, well-tested.
- **`ShannonPrimeWanCacheSqfree`** (aggressive, opt-in): sqfree-padded VHT2
  + Knight mask + Möbius CSR predictor + 3-bit residual + SU(2) spinor sheet
  bit. Targets Q8+ backbones; wraps `VHT2SqfreeCrossAttentionCache` from the
  submodule.

Production results (Wan 2.2 14B, RTX 2060): 1.20× cross-attention speedup,
0.9984 output correlation on the ship path.

## Supported Models

| Model | Type | Support |
|-------|------|---------|
| Wan 2.1 14B / 1.3B | Dense | Full |
| Wan 2.2 A14B T2V | MoE (2 experts) | Full (expert-aware caching) |
| Wan 2.2 A14B I2V | MoE (2 experts) | Full (boundary = 0.900) |
| Wan 2.2 TI2V-5B | Dense | Full |

## Installation

```bash
# Option A — install into an existing ComfyUI
cd /path/to/ComfyUI/custom_nodes
git clone --recursive https://github.com/nihilistau/shannon-prime-comfyui.git

# Option B — point a ComfyUI install at this repo via a junction / symlink
mklink /J /path/to/ComfyUI/custom_nodes/shannon-prime-comfyui \
    /path/to/shannon-prime-comfyui
```

## Usage

### Ship workflow
```bash
python scripts/run_workflow.py workflows/wan22_ti2v_5b_vht2.json
```

### Sqfree+spinor workflow (aggressive, Q8+)
```bash
python scripts/run_workflow.py workflows/wan22_ti2v_5b_sqfree.json
```

### Programmatic use (ship wrapper)
```python
from shannon_prime_comfyui import WanVHT2Wrapper

wrapper = WanVHT2Wrapper(head_dim=128, model_type='wan22_moe', task_type='t2v')

for step, sigma in enumerate(sigmas):
    wrapper.set_expert_from_sigma(sigma)
    for block_idx in range(40):
        k, v = wrapper.get_or_compute(
            f"block_{block_idx}",
            lambda: (block.cross_attn_k(context), block.cross_attn_v(context))
        )

wrapper.reset()  # Between generations
```

## Node Parameters

### `ShannonPrimeWanCache` (ship)
| Input | Default | Description |
|-------|---------|-------------|
| `k_bits` | `"5,4,4,3"` | K band bit allocation (4 bands). Cross-attn-specific — the parent engine's self-attn default is `5,5,4,3` because RoPE concentrates the first K band; here there's no RoPE on the cross-attn K, so K and V ship with matching symmetric profiles. |
| `v_bits` | `"5,4,4,3"` | V band bit allocation — banded for cross-attn (self-attn uses flat-3 instead). |
| `use_mobius` | `True` | Möbius squarefree-first reorder on both K and V |

### `ShannonPrimeWanCacheSqfree` (aggressive)
| Input | Default | Description |
|-------|---------|-------------|
| `band_bits` | `"3,3,3,3,3"` | 5-band torus-aligned allocation over the sqfree-padded skeleton |
| `residual_bits` | `3` | N-bit residual quantization (1–4; 3 is the Pareto point) |
| `use_spinor` | `True` | SU(2) sheet-bit correction at the causal-mask boundary |

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for architecture diagrams and
Wan-specific details.

## License

**AGPLv3** for open-source, academic, and non-proprietary use.
Everyone can use it and benefit. Derivative works must share alike.

**Dual License** — the primary goal is that the work belongs to the
commons and is protected from closure. A commercial license is
available for proprietary integration. See [LICENSE](LICENSE).

Copyright (C) 2026 Ray Daniels.

## Contact

Email: raydaniels@gmail.com
